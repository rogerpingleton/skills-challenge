
# Linux and command-line tools

## File & data inspection

At the core of AI work is wrangling data. `cat`, `head`, and `tail` let you quickly inspect datasets. `wc -l` gives you row counts instantly. `less` lets you page through large files. For structured data, `jq` is indispensable for parsing and filtering JSON (model outputs, API responses, configs):

```bash
# Preview first 5 rows of a dataset
head -n 5 dataset.csv

# Count training samples
wc -l train.jsonl

# Extract a specific field from model output
cat outputs.json | jq '.[].response'
```

---

## Text processing

`grep`, `sed`, and `awk` form a powerful trio for transforming and searching text. You'll use these constantly when cleaning datasets, filtering logs, or extracting patterns from model outputs:

```bash
# Find all lines with "error" in logs
grep -i "error" training.log

# Strip whitespace/artifacts from a dataset
sed 's/\r//' messy_data.txt

# Compute average loss from training logs
awk '/loss:/ {sum+=$2; n++} END {print sum/n}' train.log
```

---

## Process & job management

Training runs are long. `tmux` and `screen` let you detach from sessions so jobs survive disconnects. `nohup` keeps processes running after logout. `ps`, `top`, and `htop` let you monitor CPU/memory usage:

```bash
# Start a detachable training session
tmux new -s training
python train.py

# Detach and leave it running: Ctrl+B, then D
# Reattach later:
tmux attach -t training
```

---

## GPU monitoring

`nvidia-smi` is your best friend when running on GPUs — it shows utilization, memory usage, and running processes. For continuous monitoring:

```bash
# Watch GPU stats every 2 seconds
watch -n 2 nvidia-smi

# Log GPU stats to a file during training
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used \
  --format=csv -l 5 >> gpu_log.csv
```

---

## Environment & dependency management

`pip`, `conda`, and `virtualenv` manage Python environments, but at the shell level, `which`, `env`, and `export` help you debug path and environment variable issues — critical when managing CUDA versions or API keys:

```bash
# Check which Python is active
which python

# Set an API key for a session
export ANTHROPIC_API_KEY="sk-..."

# Inspect all environment variables
env | grep CUDA
```

---

## File transfer & remote work

`ssh` for remote servers, `scp` and `rsync` for moving datasets and model checkpoints. `rsync` is especially useful because it's resumable and only syncs changed files:

```bash
# Sync a checkpoint directory to a remote server
rsync -avz --progress ./checkpoints/ user@gpu-server:/runs/exp1/
```

---

## Disk & storage

Models and datasets are large. `df -h` shows disk usage, `du -sh` shows directory sizes, and `find` locates files by name or pattern:

```bash
# Find the 10 largest files in your project
du -ah . | sort -rh | head -10

# Find all checkpoint files
find . -name "*.pt" -type f
```

---

## Scripting & automation

`bash` scripting ties everything together — automating hyperparameter sweeps, evaluation pipelines, or data preprocessing. `xargs` and `parallel` let you run jobs in parallel:

```bash
# Run eval on multiple checkpoints in parallel
ls checkpoints/*.pt | xargs -P 4 -I{} python eval.py --ckpt {}
```

---

## curl & httpie

For testing API endpoints (your own or third-party), `curl` is essential. `httpie` (`http`) is a friendlier alternative with colored output:

```bash
# Test your model's REST endpoint
curl -X POST https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-sonnet-4-20250514","max_tokens":100,"messages":[{"role":"user","content":"Hello"}]}'
```

---

## Profiling & performance

`time` measures how long a command takes. `strace` traces system calls (useful for debugging hanging processes). For Python specifically, `py-spy` and `line_profiler` integrate well with shell workflows.

```bash
# Time a preprocessing script
time python preprocess.py --input raw/ --output clean/
```

---

## Putting it together: a typical AI Engineer workflow

Here's roughly how these tools chain in practice:

!ai_engineer_cli_workflow.svg|697

---

A few tools worth calling out as particularly high-leverage for Python-focused AI Engineers:

`tmux` — if you only learn one tool from this list beyond the basics, make it this one. Remote training jobs disconnecting mid-run is one of the most painful experiences in the field, and `tmux` eliminates it entirely.

`jq` — JSON is everywhere in AI (model outputs, API responses, dataset formats), and `jq` makes it queryable and transformable from the shell without writing a Python script.

`nvidia-smi` + `watch` — understanding your GPU utilization at a glance is essential for diagnosing underutilization (batches too small, data loading bottleneck) vs. OOM errors.

`rsync` — moving large checkpoints and datasets efficiently between machines is a daily task, and rsync's resumability saves real time.

### Show me examples of grep, sed, awk for AI data preprocessing

Here are practical, real-world examples for each tool — the kind of commands you'll actually reach for when working with datasets.

---

### `head` — quick dataset previews

Before loading anything into Python, `head` tells you what you're dealing with: encoding, delimiter, field names, sample values.

```bash
# Preview a CSV dataset
head -n 5 train.csv

# Preview a JSONL file (one JSON object per line — common for LLM fine-tuning)
head -n 3 finetune.jsonl

# Preview without the header row (useful for wc -l counting later)
tail -n +2 train.csv | head -5

# Peek at a large compressed dataset without decompressing
zcat dataset.jsonl.gz | head -n 5
```

---

### `wc` — counting samples

```bash
# Count total training examples (lines in JSONL)
wc -l train.jsonl

# Count across train/val/test splits at once
wc -l data/*.jsonl

# Count tokens (rough estimate — word count as proxy)
wc -w train.txt

# Check file size alongside line count
wc -lc train.jsonl
```

A common pattern when preparing fine-tuning data:

```bash
# Verify your split ratios add up correctly
wc -l data/train.jsonl data/val.jsonl data/test.jsonl
# Output:
#   8000 data/train.jsonl
#   1000 data/val.jsonl
#   1000 data/test.jsonl
#  10000 total
```

---

### `jq` — inspecting and transforming JSON datasets

`jq` is where things get powerful. AI datasets are almost always JSON or JSONL, and `jq` lets you slice, filter, and reshape them without writing Python.

---

### Combining all three: a typical dataset audit

This is the sequence you'd actually run when receiving a new dataset:

```bash
# 1. What format is it?
head -n 1 dataset.jsonl

# 2. How many examples?
wc -l dataset.jsonl

# 3. What fields exist?
head -n 1 dataset.jsonl | jq 'keys'

# 4. Any missing completions?
jq 'select(.completion == null or .completion == "")' dataset.jsonl | wc -l

# 5. Class distribution
jq -r '.label' dataset.jsonl | sort | uniq -c | sort -rn

# 6. Sample a few records to eyeball quality
jq '.' dataset.jsonl | head -n 60
```

This whole audit takes about 30 seconds and tells you far more than loading the dataset into a notebook first. You'll often catch encoding issues, schema mismatches, or severe class imbalance before ever touching Python.

One `jq` flag worth memorizing: `-r` (raw output) strips the JSON quotes from strings, which is what you want whenever you're piping output to `sort`, `uniq`, `wc`, or `grep`.

### Show me examples of grep, sed, awk for AI data preprocessing

Here are the patterns you'll actually reach for when cleaning and preparing datasets.

---

### `grep` — filtering and finding

`grep` is your first pass: pull out lines that match (or don't match) a pattern.

```bash
# Remove lines containing common boilerplate/artifacts
grep -v "^$" dataset.jsonl                    # drop empty lines
grep -v "^\s*$" dataset.jsonl                 # drop whitespace-only lines
grep -iv "click here\|unsubscribe\|copyright" scraped.txt  # drop web junk

# Find potentially problematic records before training
grep -n "TODO\|FIXME\|PLACEHOLDER" dataset.jsonl   # -n shows line numbers
grep -c "toxic\|harmful" labels.jsonl              # -c counts matches

# Extract only matching lines into a new file
grep '"label": "positive"' dataset.jsonl > positives.jsonl

# Check encoding issues (finds non-ASCII bytes)
grep -P "[\x80-\xFF]" dataset.txt | head -5
```

A very common pattern — subsample a large dataset randomly using `grep` + `shuf`:

```bash
# Randomly sample 10,000 lines from a large JSONL
shuf dataset.jsonl | head -n 10000 > sample_10k.jsonl
```

---

### `sed` — stream editing text

`sed` works line by line, making substitutions and deletions. It's the right tool for systematic text cleanup.

```bash
# Remove Windows line endings (\r) — a silent training killer
sed 's/\r//' messy.txt > clean.txt

# Strip leading/trailing whitespace from each line
sed 's/^[[:space:]]*//; s/[[:space:]]*$//' data.txt

# Remove HTML tags from scraped training data
sed 's/<[^>]*>//g' scraped.html > stripped.txt

# Normalize quotes (curly → straight) for tokenizer consistency
sed "s/['']/'/g; s/[\"\"]/\"/g" data.txt

# Add a field prefix — e.g., prepend "### Instruction:\n" to each line
sed 's/^/### Instruction: /' prompts.txt

# Delete lines matching a pattern (in-place edit with -i)
sed -i '/^\s*#/d' data.txt      # drop comment lines
sed -i '/^.\{0,10\}$/d' data.txt # drop lines under 10 chars (too short)
```

`-i` edits the file in place. On macOS, you need `-i ''` instead of just `-i`.

---

### `awk` — structured field processing

`awk` treats each line as a record with fields. It's the right tool when your data has structure — CSVs, TSVs, space-delimited logs.

```bash
# Extract columns from a TSV (tab-separated) dataset
awk -F'\t' '{print $1}' dataset.tsv          # first column only
awk -F'\t' '{print $2 "\t" $4}' data.tsv    # reorder columns 2 and 4

# Filter rows by a field value
awk -F',' '$3 == "positive"' labeled.csv > positives.csv
awk -F',' '$3 != "neutral"' labeled.csv     # exclude a class

# Check for rows with wrong number of fields (data corruption check)
awk -F'\t' 'NF != 3 {print NR": "NF" fields: "$0}' dataset.tsv

# Compute average response length from a length column
awk -F'\t' '{sum += length($2); n++} END {print "avg length:", sum/n}' data.tsv

# Deduplicate on a key field (first occurrence wins)
awk -F'\t' '!seen[$1]++' dataset.tsv > deduped.tsv

# Add a line number as an ID field
awk '{print NR "\t" $0}' data.txt > data_with_ids.txt

# Split into train/val/test (80/10/10) without shuffling
awk 'NR % 10 <= 7' data.jsonl > train.jsonl
awk 'NR % 10 == 8' data.jsonl > val.jsonl
awk 'NR % 10 >= 9' data.jsonl > test.jsonl
```

---

### Putting it together:  a preprocessing pipeline

!grep_sed_awk_pipeline.svg|697

This is what the full pipeline looks like as a single chained command — common in Makefiles or preprocessing shell scripts:

```bash
cat raw_scraped.txt \
  | grep -v "^\s*$" \                          # drop blank lines
  | grep -iv "unsubscribe\|click here" \       # drop boilerplate
  | sed 's/\r//; s/<[^>]*>//g' \              # strip \r and HTML tags
  | sed "s/['']/'/g; s/[\"\"]/\"/g" \        # normalize quotes
  | awk 'length($0) > 20' \                   # drop very short lines
  | awk '!seen[$0]++' \                       # deduplicate
  > cleaned.txt

wc -l raw_scraped.txt cleaned.txt             # check how many lines survived
```

---

A few things worth internalizing:

`grep -v` (invert match) is often more useful than `grep` itself in preprocessing — you're usually trying to throw things away, not find them.

`awk '!seen[$0]++'` is a classic one-liner for deduplication. It works by building a hash of lines seen — when a line appears a second time, `seen[$0]` is already truthy, `!` flips it to false, and `awk` skips printing it.

The train/val/test split via `awk 'NR % 10'` is deterministic (same split every run) and works on files too large to load into memory — something a Python script can't always say.

### How do I use tmux and nohup to manage long AI training runs?

Both tools solve the same core problem — your training run dying when your connection drops — but in different ways and at different levels of power.

---

### The problem they solve

When you SSH into a remote GPU server and run `python train.py`, that process is a child of your shell. When your SSH connection drops (network blip, laptop closes, timeout), the shell gets a `SIGHUP` signal and kills everything under it — including your training run.

---

### `nohup` — the lightweight option

`nohup` ("no hangup") simply tells a process to ignore `SIGHUP`. It's one command, no setup.

```bash
# Basic usage — output goes to nohup.out by default
nohup python train.py &

# Better: redirect stdout and stderr to a named log file
nohup python train.py > runs/exp1.log 2>&1 &

# The & sends it to the background immediately
# You'll see something like: [1] 48291  (that's the PID)

# Check it's running
ps aux | grep train.py

# Follow the log in real time
tail -f runs/exp1.log

# Kill it if needed
kill 48291
```

`nohup` is fine for fire-and-forget jobs where you don't need to interact with the process again. Its weakness: you can't reattach to it, see live output interactively, or run multiple panes.

---

### `tmux` — the full solution

`tmux` is a terminal multiplexer. It runs a persistent server that keeps sessions alive independent of any SSH connection. You attach to a session, do work, detach, disconnect — the session keeps running. You can reattach from anywhere.

The mental model:

````
tmux server (always running on the remote host)
  └── session: "training"
        ├── window 0: train.py running
        ├── window 1: nvidia-smi watch
        └── window 2: log tail
```---

### A real AI training workflow with tmux

This is the setup pattern most AI engineers settle into:

```bash
# 1. SSH into your GPU server
ssh user@gpu-server

# 2. Create a session named for the experiment
tmux new -s exp1

# 3. Window 0: start training
python train.py \
  --config configs/gpt2_finetune.yaml \
  --output-dir runs/exp1 \
  2>&1 | tee runs/exp1/train.log

# 4. Open a second window (Ctrl+B, c) for monitoring
watch -n 2 nvidia-smi

# 5. Open a third window (Ctrl+B, c) for log tailing
tail -f runs/exp1/train.log | grep -E "loss|epoch|step"

# 6. Detach and walk away (Ctrl+B, d)
# Your training continues. Close your laptop.

# 7. Reconnect hours later
ssh user@gpu-server
tmux attach -t exp1
# You're back — all three windows intact, training still running
````

---

!tmux_training_cheatsheet.html

---

### Combining both: `tmux` + `nohup`

You don't usually need both, but there's one case where it makes sense: you want the process to survive even if someone accidentally kills the tmux session.

```bash
# Inside a tmux session, launch with nohup for double protection
nohup python train.py > runs/exp1.log 2>&1 &
echo $! > runs/exp1.pid   # save the PID for later

# Now even if tmux dies, the process lives
# Kill it cleanly when done:
kill $(cat runs/exp1.pid)
```

---

### A reusable training launch script

Worth keeping as `launch_training.sh` in your project:

```bash
#!/bin/bash
SESSION="train_$(date +%Y%m%d_%H%M)"
LOG="runs/$SESSION.log"

mkdir -p runs

tmux new-session -d -s "$SESSION"   # -d = detached (don't attach yet)

tmux send-keys -t "$SESSION" \
  "python train.py --config $1 2>&1 | tee $LOG" Enter

echo "Training started in tmux session: $SESSION"
echo "Log: $LOG"
echo ""
echo "Attach with:  tmux attach -t $SESSION"
echo "Follow log:   tail -f $LOG"
```

Then use it as:

```bash
chmod +x launch_training.sh
./launch_training.sh configs/my_experiment.yaml
```

This creates a uniquely named session per run, logs everything, and prints the attach command — so you never have to remember the session name from a week ago.

---

The one habit that pays off most: always name your sessions after the experiment (`tmux new -s exp_llama_lora_v2`), not something generic like `tmux new -s work`. When you have three runs going simultaneously, `tmux ls` becomes much more useful.

### How do I monitor GPU usage and training logs on the command line?

GPU monitoring is one of those skills where knowing three or four commands really well beats knowing twenty commands shallowly. Here's what actually gets used.

---

### `nvidia-smi` — the core tool

```bash
# Snapshot: current state of all GPUs
nvidia-smi

# Continuous watch — refreshes every 2 seconds
watch -n 2 nvidia-smi

# Compact one-liner output (great for logging)
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,\
memory.used,memory.total,temperature.gpu,power.draw \
  --format=csv,noheader,nounits

# Log GPU stats to CSV during a training run
nvidia-smi \
  --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,temperature.gpu \
  --format=csv \
  -l 5 \                     # sample every 5 seconds
  >> runs/exp1/gpu_stats.csv &
```

That last command — logging to CSV in the background while training runs — is worth making a habit. You can plot it afterward to diagnose exactly when OOM errors occurred or where utilization dropped.

---

### Reading `nvidia-smi` output

This is the part that trips people up early on:

!nvidia_smi_annotated.html

---

### Diagnosing common problems from `nvidia-smi`

```bash
# GPU util stuck near 0% after starting training?
# → Process may have crashed. Check:
ps aux | grep train.py
tail -n 50 runs/exp1/train.log

# Memory near 100% but training hasn't started yet?
# → A zombie process is hogging VRAM. Find and kill it:
nvidia-smi          # note the PID in the Processes section
kill -9 <PID>

# Multiple GPUs — check all at once
nvidia-smi topo -m  # shows GPU interconnect topology (NVLink etc.)
```

---

### Monitoring training logs

`tail -f` is the basic tool, but filtering its output is what makes it useful:

```bash
# Follow raw log
tail -f runs/exp1/train.log

# Filter to only loss/metric lines (most frameworks print these consistently)
tail -f runs/exp1/train.log | grep -E "loss|acc|epoch|step|eval"

# Show loss lines with a timestamp prefix
tail -f runs/exp1/train.log \
  | grep --line-buffered "loss" \
  | awk '{print strftime("[%H:%M:%S]"), $0}'

# Watch for errors without stopping the stream
tail -f runs/exp1/train.log | tee >(grep -i "error\|nan\|inf\|cuda" >&2)
```

That last one tees the log into two streams simultaneously — the full log passes through to your terminal, while a second process watches for NaN/Inf/CUDA errors and prints them to stderr. Useful for catching silent training failures mid-run.

---

### A complete monitoring layout in tmux

This is the pane layout worth setting up at the start of every long run:

```bash
# Start a new session
tmux new -s monitor

# Pane 1 (top-left): GPU stats refreshing every 2s
watch -n 2 nvidia-smi

# Split horizontally (Ctrl+B, ") → Pane 2 (bottom-left): filtered log
tail -f runs/exp1/train.log | grep -E "loss|step|epoch"

# Split pane 2 vertically (Ctrl+B, %) → Pane 3 (bottom-right): system resources
htop
```

Or script the entire layout so you can launch it in one command:

```bash
#!/bin/bash
# monitor.sh <log_file>
LOG=${1:-"runs/latest/train.log"}
SESSION="monitor"

tmux new-session -d -s $SESSION -x 220 -y 50

tmux send-keys -t $SESSION "watch -n 2 nvidia-smi" Enter

tmux split-window -v -t $SESSION
tmux send-keys -t $SESSION "tail -f $LOG | grep -E 'loss|step|epoch|eval'" Enter

tmux split-window -h -t $SESSION
tmux send-keys -t $SESSION "htop" Enter

tmux select-pane -t $SESSION:0.0
tmux attach -t $SESSION
```

```bash
chmod +x monitor.sh
./monitor.sh runs/exp1/train.log
```

---

### Useful one-liners for post-run analysis

After a run completes, these help you extract key metrics from logs without opening a notebook:

```bash
# Plot loss curve as ASCII (requires gnuplot)
grep "loss:" train.log \
  | awk '{print NR, $NF}' \
  | gnuplot -p -e "plot '-' with lines title 'loss'"

# Extract final eval metrics
grep "eval_" train.log | tail -20

# Find the step with the lowest validation loss
grep "val_loss" train.log \
  | awk '{print NR, $NF}' \
  | sort -k2 -n \
  | head -1

# Check if any NaN losses appeared (silent training killer)
grep -c "nan\|NaN\|inf\|Inf" train.log
```

The NaN check is worth running on every completed log. A training run can produce `NaN` loss for hundreds of steps before diverging visibly — catching it early in the log tells you exactly when it started.

### Show me how to use curl and rsync in an AI model evaluation pipeline

Both tools handle the "edges" of your evaluation pipeline — `curl` talks to model APIs and serves endpoints, `rsync` moves artifacts (checkpoints, datasets, results) efficiently between machines. Here's how they fit in practice.

---

### `curl` — calling model APIs during eval

The most common use: sending prompts to a model endpoint and capturing responses.

```bash
# Basic Anthropic API call
curl https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 512,
    "messages": [{"role": "user", "content": "Summarize: The cat sat on the mat."}]
  }'

# Pipe response through jq to extract just the text
curl -s https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-sonnet-4-20250514","max_tokens":256,
       "messages":[{"role":"user","content":"What is 2+2?"}]}' \
  | jq -r '.content[0].text'
```

The `-s` flag (silent) suppresses the progress bar — essential when scripting, otherwise your log fills with `###` progress indicators.

---

### Evaluating a local model endpoint

When you're serving your own fine-tuned model (e.g. via vLLM, TGI, or a FastAPI wrapper):

```bash
# Test a locally served model endpoint
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Translate to French: Hello, how are you?",
    "max_tokens": 64,
    "temperature": 0.0
  }'

# Check your server is alive before starting eval
curl -sf http://localhost:8000/health && echo "Server ready" || echo "Server down"

# Time a single inference request (useful for latency benchmarking)
time curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 100}'
```

---

### Batch eval loop with `curl`

The most useful pattern — loop over a JSONL eval set, call the API for each prompt, collect results:

```bash
#!/bin/bash
# eval_loop.sh — runs curl against each prompt in eval.jsonl

INPUT="data/eval.jsonl"
OUTPUT="results/eval_outputs.jsonl"
MODEL="claude-sonnet-4-20250514"

mkdir -p results
> "$OUTPUT"   # clear output file

while IFS= read -r line; do
  PROMPT=$(echo "$line" | jq -r '.prompt')
  ID=$(echo "$line" | jq -r '.id')

  RESPONSE=$(curl -s https://api.anthropic.com/v1/messages \
    -H "x-api-key: $ANTHROPIC_API_KEY" \
    -H "anthropic-version: 2023-06-01" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"$MODEL\",
      \"max_tokens\": 512,
      \"messages\": [{\"role\": \"user\", \"content\": $(echo "$PROMPT" | jq -R '.')}]
    }")

  # Append id + response as one JSONL record
  echo "$RESPONSE" \
    | jq --arg id "$ID" --argjson orig "$line" \
        '{id: $id, prompt: $orig.prompt, response: .content[0].text}' \
    >> "$OUTPUT"

  sleep 0.5   # respect rate limits

done < "$INPUT"

echo "Done. $(wc -l < $OUTPUT) results written to $OUTPUT"
```

```bash
chmod +x eval_loop.sh
./eval_loop.sh
```

---

### `rsync` — moving checkpoints, datasets, and results

`rsync` copies only what has changed, is resumable, and works over SSH. For large model files it's far more practical than `scp`.

```bash
# Copy a checkpoint directory to a remote server
rsync -avz --progress \
  runs/exp1/checkpoints/ \
  user@gpu-server:/workspace/runs/exp1/checkpoints/

# Flags:
#   -a  archive mode (preserves permissions, timestamps, symlinks)
#   -v  verbose (shows files being transferred)
#   -z  compress during transfer (helps for text files, skip for already-compressed)
#   --progress  shows per-file progress bar

# Pull results back from remote server after eval
rsync -avz \
  user@gpu-server:/workspace/results/eval_outputs.jsonl \
  ./results/

# Dry run first — see what WOULD be transferred without doing it
rsync -avzn \
  runs/exp1/ \
  user@gpu-server:/workspace/runs/exp1/
```

---

### Key `rsync` patterns for AI workflows

```bash
# Sync only new checkpoints (skip ones already on remote)
rsync -avz --ignore-existing \
  runs/exp1/checkpoints/ \
  user@gpu-server:/workspace/checkpoints/

# Exclude optimizer states to save bandwidth (often 2-3x the model size)
rsync -avz \
  --exclude="optimizer.pt" \
  --exclude="scheduler.pt" \
  runs/exp1/checkpoints/ \
  user@gpu-server:/workspace/checkpoints/

# Mirror a dataset to remote — delete files on remote that no longer exist locally
rsync -avz --delete \
  data/processed/ \
  user@gpu-server:/workspace/data/processed/

# Resume an interrupted transfer (partial files picked up automatically)
rsync -avz --partial --progress \
  large_model_weights/ \
  user@gpu-server:/workspace/weights/

# Sync between two remote servers (without pulling to local first)
rsync -avz \
  user@server-a:/workspace/checkpoints/ \
  user@server-b:/workspace/checkpoints/
```

`--delete` is the flag to be careful with — it removes files on the destination that don't exist on the source. Essential for keeping a clean mirror, but destructive if you point it at the wrong directory.

---

### A complete evaluation pipeline

Here's how `curl`, `rsync`, and the tools from earlier in the conversation compose into a full eval pipeline:

!eval_pipeline_curl_rsync.svg|697

As a single script that wires it all together:

```bash
#!/bin/bash
# run_eval.sh <checkpoint_name>
set -e   # exit immediately on any error

CKPT=$1
REMOTE="user@gpu-server"
REMOTE_CKPT="/workspace/runs/$CKPT"
LOCAL_CKPT="./checkpoints/$CKPT"
EVAL_DATA="data/eval.jsonl"
RESULTS="results/${CKPT}_outputs.jsonl"
METRICS="results/${CKPT}_metrics.json"

echo "=== 1. Pull checkpoint via rsync ==="
mkdir -p "$LOCAL_CKPT"
rsync -avz --exclude="optimizer.pt" \
  "$REMOTE:$REMOTE_CKPT/" "$LOCAL_CKPT/"

echo "=== 2. Start model server ==="
python serve.py --checkpoint "$LOCAL_CKPT" &
SERVER_PID=$!
sleep 10   # wait for server to be ready
curl -sf http://localhost:8000/health || { echo "Server failed"; kill $SERVER_PID; exit 1; }

echo "=== 3. Run eval loop ==="
mkdir -p results
> "$RESULTS"
while IFS= read -r line; do
  PROMPT=$(echo "$line" | jq -r '.prompt')
  ID=$(echo "$line" | jq -r '.id')
  RESPONSE=$(curl -s http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"prompt\": $(echo "$PROMPT" | jq -R '.'), \"max_tokens\": 256, \"temperature\": 0}")
  echo "$RESPONSE" \
    | jq --arg id "$ID" --argjson orig "$line" \
        '{id: $id, expected: $orig.label, response: .choices[0].text}' \
    >> "$RESULTS"
done < "$EVAL_DATA"

echo "=== 4. Score results ==="
python score.py --results "$RESULTS" --output "$METRICS"
cat "$METRICS"

echo "=== 5. Push results to remote ==="
rsync -avz "$RESULTS" "$METRICS" "$REMOTE:/workspace/results/"

echo "=== 6. Shut down server ==="
kill $SERVER_PID

echo "Done. Results: $METRICS"
```

```bash
chmod +x run_eval.sh
./run_eval.sh exp1_checkpoint_step8000
```

---

A few things worth emphasizing:

The `curl -sf` health check before starting the eval loop is important — `-f` makes curl return a non-zero exit code on HTTP errors, so `set -e` will abort the script if your server isn't ready rather than running 1,000 requests against a dead endpoint.

`rsync --exclude="optimizer.pt"` is a practical habit on checkpoints. Optimizer states are often 2–3× the model weights and you typically don't need them for inference during eval — skipping them cuts transfer time significantly on large models.

The `sleep 0.5` rate-limit pause in the eval loop matters more than it seems. At scale, even modest eval sets (a few hundred examples) can hit API rate limits fast without it.