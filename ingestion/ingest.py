#!/usr/bin/env python3
"""Build the shipped SQLite RAG index from a directory of markdown files."""
import argparse, json, os, sqlite3, struct, subprocess, sys
from pathlib import Path
from typing import Iterator

# --- Long-lived Swift subprocess wrappers -----------------------------------

class SwiftTool:
    def __init__(self, binary: str, subcommand: str):
        self.proc = subprocess.Popen(
            [binary, subcommand],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1,
        )

    def call(self, payload: dict) -> dict:
        line = json.dumps(payload, ensure_ascii=False)
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()
        resp = self.proc.stdout.readline()
        return json.loads(resp)

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()

# --- Chunking ---------------------------------------------------------------

TARGET_TOKENS_PER_CHUNK = 180   # leaves headroom in the 4k Foundation window
MAX_TOKENS_PER_CHUNK = 240      # don't exceed NLEmbedding's ~256 token input
OVERLAP_SENTENCES = 1

def extract_title(md: str) -> str | None:
    for line in md.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return None

def build_chunks(sentences, counter: SwiftTool, doc_path: str):
    """Yield (start_sent, end_sent, start_utf16, end_utf16, text, token_count)."""
    i, n = 0, len(sentences)
    while i < n:
        pieces, tokens = [], 0
        j = i
        while j < n:
            cand = " ".join(s["text"] for s in sentences[i:j+1])
            tc = counter.call({"id": f"{doc_path}:{i}-{j}", "text": cand})["tokenCount"]
            if tc > MAX_TOKENS_PER_CHUNK and j > i:
                break
            tokens = tc
            j += 1
            if tokens >= TARGET_TOKENS_PER_CHUNK:
                break
        j = max(j, i + 1)  # always at least 1 sentence
        start_sent = sentences[i]["index"]
        end_sent = sentences[j-1]["index"]
        start_utf16 = sentences[i]["startUTF16"]
        end_utf16 = sentences[j-1]["endUTF16"]
        text = " ".join(s["text"] for s in sentences[i:j])
        yield (start_sent, end_sent, start_utf16, end_utf16, text, tokens)
        if j >= n: break
        i = max(i + 1, j - OVERLAP_SENTENCES)

def heading_breadcrumb(md: str, utf16_pos: int) -> str:
    """Build a heading chain for the position; prepended to embed-time text."""
    # Cheap: scan line-by-line, track heading stack.
    # (For robust prod use, parse the markdown with mistune or similar.)
    stack, pos = {}, 0
    for line in md.splitlines(keepends=True):
        if pos >= utf16_pos: break
        l = line.lstrip()
        if l.startswith("#"):
            level = len(l) - len(l.lstrip("#"))
            title = l[level:].strip()
            stack[level] = title
            for k in list(stack):
                if k > level: stack.pop(k)
        pos += len(line.encode("utf-16-le")) // 2
    return " / ".join(stack[k] for k in sorted(stack))

# --- SQLite schema ----------------------------------------------------------

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
CREATE TABLE IF NOT EXISTS documents(
    doc_id INTEGER PRIMARY KEY, path TEXT UNIQUE NOT NULL,
    title TEXT, char_count INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS chunks(
    chunk_id INTEGER PRIMARY KEY, doc_id INTEGER NOT NULL REFERENCES documents,
    start_offset INTEGER NOT NULL, end_offset INTEGER NOT NULL,
    start_sent INTEGER NOT NULL, end_sent INTEGER NOT NULL,
    token_count INTEGER NOT NULL, text TEXT NOT NULL,
    embedding BLOB NOT NULL, embedding_bits BLOB);
CREATE INDEX IF NOT EXISTS chunks_doc_idx ON chunks(doc_id);
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

def pack_float32(vec):
    return struct.pack(f"<{len(vec)}f", *vec)

def pack_bits(vec):
    # Sign-based binary quantization: bit=1 if v_i > 0 else 0. 512 bits = 64 bytes.
    out = bytearray(len(vec) // 8)
    for i, v in enumerate(vec):
        if v > 0:
            out[i >> 3] |= (1 << (i & 7))
    return bytes(out)

# --- Pipeline ---------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ragtool", default="./ragingest")
    args = ap.parse_args()

    seg = SwiftTool(args.ragtool, "segment")
    emb = SwiftTool(args.ragtool, "embed")
    cnt = SwiftTool(args.ragtool, "count")

    db = sqlite3.connect(args.out)
    db.executescript(SCHEMA)
    meta = [
        ("embedding_model","NLEmbedding.sentenceEmbedding(.english)"),
        ("embedding_dim","512"),
        ("embedding_dtype","float32"),
        ("normalized","1"),
        ("schema_version","1"),
    ]
    db.executemany("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)", meta)

    md_dir = Path(args.md_dir)
    for i, path in enumerate(sorted(md_dir.rglob("*.md"))):
        text = path.read_text(encoding="utf-8")
        rel = str(path.relative_to(md_dir))
        title = extract_title(text)
        cur = db.execute("INSERT INTO documents(path,title,char_count) VALUES(?,?,?)",
                         (rel, title, len(text)))
        doc_id = cur.lastrowid
        sentences = seg.call({"id": rel, "text": text})["sentences"]
        for (ss, es, so, eo, chunk_text, tc) in build_chunks(sentences, cnt, rel):
            breadcrumb = heading_breadcrumb(text, so)
            embed_input = f"{breadcrumb}\n{chunk_text}" if breadcrumb else chunk_text
            vec = emb.call({"id": f"{rel}:{ss}-{es}", "text": embed_input})["vector"]
            db.execute(
                """INSERT INTO chunks(doc_id,start_offset,end_offset,start_sent,end_sent,
                   token_count,text,embedding,embedding_bits)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (doc_id, so, eo, ss, es, tc, chunk_text,
                 pack_float32(vec), pack_bits(vec)))
        print(f"[{i+1:3d}] {rel}  ({len(sentences)} sentences)")
        db.commit()

    seg.close(); emb.close(); cnt.close()
    # Pre-optimize the DB for read-only shipping.
    db.execute("VACUUM")
    db.execute("ANALYZE")
    db.close()
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
