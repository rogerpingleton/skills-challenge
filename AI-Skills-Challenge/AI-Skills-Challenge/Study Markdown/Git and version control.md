# Git and version control

A comprehensive reference covering essential commands, real-world examples, nuances, pitfalls, and modern workflows — including worktrees and GitHub CI/CD.
## 1. Configuration & Setup

### `git config`

Sets user identity, editor preferences, and behavioral defaults. Configuration exists at three levels: `--system` (all users), `--global` (your user), and `--local` (this repo only). Local overrides global, which overrides system.

```bash
# Identity (required before your first commit)
git config --global user.name "Ada Lovelace"
git config --global user.email "ada@example.com"

# Use a per-repo identity for work vs. personal projects
cd ~/work/ml-pipeline
git config --local user.email "ada@company.com"

# Set default branch name for new repos
git config --global init.defaultBranch main

# Useful aliases
git config --global alias.lg "log --oneline --graph --all --decorate"
git config --global alias.st "status -sb"

# Store credentials so you don't re-enter passwords
git config --global credential.helper store   # plaintext file (~/.git-credentials)
git config --global credential.helper cache   # in-memory, expires in 15 min

# View all settings and where they come from
git config --list --show-origin
```

**Nuances:** The `credential.helper store` saves passwords in plaintext. On macOS, use `osxkeychain`; on Linux, consider `libsecret` or a credential manager. For SSH-based workflows (recommended for AI work with large repos), configure SSH keys instead.

**Pitfall:** Forgetting to set your email means your commits won't be linked to your GitHub profile, breaking contribution graphs and audit trails.

---

## 2. Repository Basics

### `git init`

Creates a new Git repository in the current directory by generating a `.git/` folder.

```bash
mkdir model-training && cd model-training
git init

# Initialize with a specific default branch
git init --initial-branch=main
```

**Pitfall:** Running `git init` inside an existing repo doesn't destroy anything — it's idempotent. But running it in your home directory by accident will track everything.

### `git clone`

Copies a remote repository to your local machine, including the full history.

```bash
# Standard clone
git clone https://github.com/org/llm-finetuning.git

# Shallow clone — only the latest commit (great for large AI repos)
git clone --depth 1 https://github.com/huggingface/transformers.git

# Clone a specific branch
git clone --branch v4.38.0 --single-branch https://github.com/huggingface/transformers.git

# Clone with submodules initialized
git clone --recurse-submodules https://github.com/org/project.git
```

**Nuances:** Shallow clones (`--depth 1`) are invaluable for AI work. Repos like `transformers` have massive histories. A shallow clone can reduce clone time from minutes to seconds. However, you cannot push from or rebase within a shallow clone without "unshallowing" first: `git fetch --unshallow`.

**Pitfall:** `git clone` automatically sets up a remote called `origin`. If you later fork the repo, add the original as `upstream` — see Remote Repositories.

---

## 3. Staging & Committing

### `git add`

Moves changes from the working directory into the staging area (the "index").

```bash
# Stage a specific file
git add train.py

# Stage all changes (tracked and untracked)
git add .

# Stage only tracked files (skip new untracked files)
git add -u

# Interactive staging — choose specific hunks to stage
git add -p train.py
```

**Nuances:** `git add -p` (patch mode) is one of the most underused yet powerful features. It lets you stage individual chunks of a file, enabling atomic commits. This is critical when you've made multiple logical changes in one file — e.g., you fixed a bug in data loading AND added a new feature flag. Stage them separately, commit separately.

**Pitfall:** `git add .` stages everything, including large data files, model checkpoints, and `.env` files with API keys. Always set up `.gitignore` first.

### `git commit`

Records staged changes as a snapshot in the repository history.

```bash
# Commit with an inline message
git commit -m "Add learning rate scheduler to training loop"

# Stage all tracked changes and commit in one step
git commit -am "Fix tokenizer padding bug"

# Amend the previous commit (message or content)
git commit --amend -m "Fix tokenizer padding bug for batch inference"

# Amend without changing the message
git commit --amend --no-edit

# Create an empty commit (useful for triggering CI)
git commit --allow-empty -m "Trigger CI pipeline"
```

**Nuances:** Commit messages matter enormously for AI projects. Use conventional commits for clarity: `feat:`, `fix:`, `refactor:`, `data:`, `experiment:`. A message like "updates" is useless when you're tracing why model accuracy dropped after 200 commits.

**Pitfall:** `git commit --amend` rewrites the last commit's hash. Never amend a commit that has already been pushed to a shared branch — it will cause divergence for other collaborators. If you must, you'll need `git push --force-with-lease`.

### `git status` and `git diff`

```bash
# Concise status
git status -sb

# Diff between working directory and staging area
git diff

# Diff between staging area and last commit
git diff --staged

# Diff specific file
git diff HEAD -- config.yaml

# Word-level diff (useful for config files and notebooks)
git diff --word-diff
```

---

## 4. Branching

### `git branch` and `git switch` / `git checkout`

Branches are lightweight pointers to commits. They're cheap to create and essential for parallel experimentation — the bread and butter of AI development.

```bash
# List local branches
git branch

# List all branches (including remote-tracking)
git branch -a

# Create a new branch
git branch feature/add-lora-adapter

# Switch to a branch (modern way)
git switch feature/add-lora-adapter

# Create and switch in one step
git switch -c experiment/lr-sweep

# Legacy equivalent (still works, more common in scripts)
git checkout -b experiment/lr-sweep

# Delete a merged branch
git branch -d feature/add-lora-adapter

# Force-delete an unmerged branch
git branch -D experiment/failed-approach

# Rename the current branch
git branch -m new-name
```

**Nuances:** `git switch` was introduced in Git 2.23 to reduce the overloaded nature of `git checkout`, which handles branches, files, detached HEADs, and more. Prefer `git switch` for branches and `git restore` for files — the separation is clearer and less error-prone.

**Pitfall:** Deleting a branch only removes the pointer, not the commits. However, orphaned commits will be garbage-collected after ~30 days. If you realize you need those commits, use `git reflog` to recover them before GC runs.

**AI-specific tip:** Adopt a naming convention for experiment branches: `experiment/<hypothesis>`, `data/<dataset-version>`, `model/<architecture-change>`. This makes it trivial to find and compare experiments later.

---

## 5. Merging & Rebasing

### `git merge`

Combines two branches by creating a merge commit (or fast-forwarding if possible).

```bash
# Merge a feature branch into main
git switch main
git merge feature/add-lora-adapter

# Force a merge commit even if fast-forward is possible
git merge --no-ff feature/add-lora-adapter

# Abort a merge that has conflicts
git merge --abort

# Merge but squash all commits into one staged change
git merge --squash feature/add-lora-adapter
git commit -m "Add LoRA adapter support"
```

**Nuances:** `--no-ff` is recommended for feature branches. It preserves the history that a branch existed, making it easier to revert an entire feature later. Many teams enforce this via branch protection rules on GitHub.

`--squash` is useful when a branch has a messy history of WIP commits, but you want the final result as a single clean commit on `main`.

### `git rebase`

Replays commits from the current branch on top of another branch, rewriting history to create a linear sequence.

```bash
# Rebase current branch onto main
git rebase main

# Interactive rebase — squash, reorder, edit, drop commits
git rebase -i HEAD~5

# Abort a rebase
git rebase --abort

# Continue after resolving conflicts
git rebase --continue
```

**Nuances:** The golden rule of rebase: **never rebase commits that have been pushed to a shared branch.** Rebasing rewrites commit hashes. If others have based work on those hashes, they'll face painful divergence.

Interactive rebase (`-i`) is extremely powerful. In the editor, you can:

- `pick` — keep a commit as-is
- `squash` (or `s`) — combine with the previous commit
- `fixup` (or `f`) — squash but discard the commit message
- `reword` (or `r`) — change the commit message
- `edit` (or `e`) — pause to amend the commit
- `drop` (or `d`) — remove the commit entirely

**Pitfall:** If you get into a messy rebase state, `git rebase --abort` always takes you back to where you started. Don't panic.

### Merge vs. Rebase — When to Use Which

|Scenario|Use|
|---|---|
|Integrating a shared branch (e.g., `main` into your feature)|`git merge` or `git rebase` (team preference)|
|Cleaning up local commits before a PR|`git rebase -i`|
|Preserving exact history for auditing|`git merge --no-ff`|
|Linear, clean history|`git rebase`|

---

## 6. Remote Repositories

### `git remote`, `git fetch`, `git pull`, `git push`

```bash
# View remotes
git remote -v

# Add a remote (e.g., after forking)
git remote add upstream https://github.com/original/repo.git

# Fetch updates without merging
git fetch origin
git fetch --all   # from all remotes

# Pull = fetch + merge
git pull origin main

# Pull with rebase instead of merge
git pull --rebase origin main

# Push a branch
git push origin feature/new-tokenizer

# Push and set upstream tracking
git push -u origin feature/new-tokenizer

# Force push safely (won't overwrite others' work)
git push --force-with-lease

# Delete a remote branch
git push origin --delete old-feature-branch
```

**Nuances:** Always prefer `--force-with-lease` over `--force`. The former checks that the remote branch hasn't been updated since your last fetch, preventing accidental data loss. Raw `--force` overwrites unconditionally.

`git pull --rebase` keeps your local commits on top of upstream changes, avoiding unnecessary merge commits. Many teams configure this as the default: `git config --global pull.rebase true`.

**Pitfall:** `git pull` on a branch with local commits can create unexpected merge commits. If you want a clean history, use `git fetch` + `git rebase` manually, or set `pull.rebase true`.

**Fork workflow (common in open-source AI):**

```bash
# 1. Fork on GitHub, clone your fork
git clone https://github.com/you/transformers.git
cd transformers

# 2. Add original repo as upstream
git remote add upstream https://github.com/huggingface/transformers.git

# 3. Keep your fork in sync
git fetch upstream
git switch main
git merge upstream/main
git push origin main
```

---

## 7. Stashing

### `git stash`

Temporarily shelves changes so you can switch context without committing half-done work.

```bash
# Stash working directory and staged changes
git stash

# Stash with a descriptive message
git stash push -m "WIP: data augmentation for code generation"

# Stash including untracked files
git stash -u

# Stash only specific files
git stash push -m "config changes" config.yaml settings.json

# List stashes
git stash list

# Apply the most recent stash (keeps it in the stash list)
git stash apply

# Apply and remove the most recent stash
git stash pop

# Apply a specific stash
git stash apply stash@{2}

# View what's in a stash
git stash show -p stash@{0}

# Drop a specific stash
git stash drop stash@{1}

# Clear all stashes
git stash clear
```

**Nuances:** Stashes are stored as commits on a special ref — they're actually part of the object graph. This means they survive branch switches but can be lost on operations like `git gc` if not applied.

**Pitfall:** Stash conflicts. If the code has changed significantly since you stashed, `git stash pop` can produce merge conflicts. In that case, the stash is NOT dropped — you still need to resolve and then `git stash drop` manually.

**AI-specific tip:** Stash is invaluable during long training runs. You start an experiment, realize you need to quickly test something else on another branch — stash your current changes, switch, test, switch back, pop.

---

## 8. Inspecting History

### `git log`

```bash
# Compact one-line log with graph
git log --oneline --graph --all --decorate

# Log for a specific file
git log --follow -- src/model.py

# Log with diff (patch)
git log -p -3   # last 3 commits with diffs

# Search commit messages
git log --grep="learning rate"

# Search for changes to a string in code
git log -S "CrossEntropyLoss" --oneline

# Commits by a specific author
git log --author="ada" --oneline

# Commits in a date range
git log --since="2025-01-01" --until="2025-03-01" --oneline

# Show which files changed per commit
git log --stat --oneline
```

**Nuances:** `git log -S` (the "pickaxe") is incredibly useful for AI engineers. Want to find when someone introduced or removed a particular loss function, hyperparameter, or model class? Pickaxe finds it.

`--follow` is essential when a file has been renamed, which is common during refactors (e.g., `model.py` → `models/transformer.py`).

### `git blame`

Shows who last modified each line of a file and when.

```bash
# Standard blame
git blame src/train.py

# Blame with ignored whitespace changes
git blame -w src/train.py

# Blame a specific range of lines
git blame -L 50,75 src/train.py

# Ignore specific revisions (e.g., formatting commits)
git blame --ignore-rev abc1234 src/train.py

# Ignore revisions listed in a file
echo "abc1234" >> .git-blame-ignore-revs
git config blame.ignoreRevsFile .git-blame-ignore-revs
git blame src/train.py
```

**Nuances:** `.git-blame-ignore-revs` is a game-changer. When someone runs a linter or formatter across the whole codebase, blame becomes useless. Adding that commit's hash to the ignore file restores meaningful blame output. GitHub also honors this file in the web UI.

### `git show`

```bash
# Show a specific commit
git show abc1234

# Show a file at a specific commit
git show HEAD~3:src/config.yaml

# Show just the files changed in a commit
git show --stat abc1234
```

---

## 9. Undoing Changes

### `git restore` (modern replacement for parts of `git checkout`)

```bash
# Discard changes in working directory (restore to staged version)
git restore src/train.py

# Unstage a file (move from staging back to working directory)
git restore --staged src/train.py

# Restore a file to a specific commit
git restore --source=HEAD~2 src/config.yaml
```

### `git reset`

Moves the branch pointer and optionally modifies the staging area and working directory.

```bash
# Soft reset — uncommit but keep changes staged
git reset --soft HEAD~1

# Mixed reset (default) — uncommit and unstage, but keep working directory
git reset HEAD~1

# Hard reset — uncommit, unstage, AND discard working directory changes
git reset --hard HEAD~1

# Reset a specific file from staging
git reset HEAD src/train.py
```

**Nuances:** The three modes are about where changes end up after the reset:

|Mode|HEAD moves?|Staging area|Working directory|
|---|:-:|:-:|:-:|
|`--soft`|Yes|Unchanged|Unchanged|
|`--mixed`|Yes|Reset|Unchanged|
|`--hard`|Yes|Reset|Reset|

**Pitfall:** `git reset --hard` is destructive for uncommitted changes. There is no undo unless you had stashed or committed. Always double-check with `git status` before running it.

### `git revert`

Creates a new commit that undoes a previous commit — safe for shared branches because it doesn't rewrite history.

```bash
# Revert a single commit
git revert abc1234

# Revert without auto-committing (stage the revert for review)
git revert --no-commit abc1234

# Revert a merge commit (specify which parent to keep)
git revert -m 1 abc1234
```

**Nuances:** Use `git revert` on `main` or any shared branch. Use `git reset` only on local/private branches. This distinction prevents headaches for the entire team.

---

## 10. Tagging

### `git tag`

Tags mark specific points in history — perfect for marking model releases, dataset versions, and paper submission snapshots.

```bash
# Lightweight tag
git tag v1.0.0

# Annotated tag (recommended — includes metadata)
git tag -a v2.0.0 -m "Production release: GPT-based summarizer"

# Tag a specific past commit
git tag -a v1.5.0-rc1 abc1234 -m "Release candidate 1"

# List tags
git tag -l "v2.*"

# Push a specific tag
git push origin v2.0.0

# Push all tags
git push origin --tags

# Delete a local tag
git tag -d v1.0.0-beta

# Delete a remote tag
git push origin --delete v1.0.0-beta
```

**Nuances:** Annotated tags (`-a`) store the tagger name, date, and message, and can be GPG-signed. Use annotated tags for anything official. Lightweight tags are just pointers — use them for local bookmarks.

**AI-specific tip:** Tag model releases with metadata: `git tag -a model-v3.2 -m "F1=0.94, trained on dataset-v7, 3 epochs, lr=2e-5"`. This creates a permanent, searchable record linking code state to model performance.

---

## 11. Git Worktrees

### `git worktree`

Worktrees allow you to have **multiple working directories** linked to the same repository. Each worktree can have a different branch checked out simultaneously — no more stashing or committing half-done work to switch context.

```bash
# Add a new worktree for a branch
git worktree add ../experiment-lora feature/lora-finetuning

# Add a worktree with a new branch
git worktree add ../hotfix -b hotfix/fix-inference-crash

# Add a detached worktree at a specific commit
git worktree add ../review-v2 v2.0.0 --detach

# List all worktrees
git worktree list

# Remove a worktree (after you're done)
git worktree remove ../experiment-lora

# Prune stale worktree metadata
git worktree prune
```

**How It Works:** All worktrees share the same `.git` object database. This means they share commits, branches, and history — but each has its own working directory, staging area, and HEAD. Checking out a branch in one worktree locks it from being checked out in another.

**Why AI Engineers Should Care:**

1. **Parallel experiments:** Run a training job in one worktree while developing the next experiment in another. No need to wait for the run to finish or stash changes.
2. **Code review while developing:** Check out a PR branch in a separate worktree, review it, test it — all without touching your development branch.
3. **Comparing outputs:** Have `main` and your experiment branch in side-by-side directories. Run inference on both and diff the outputs directly.
4. **Long-running processes:** Your training script is running from `/home/user/project-main`. You create `/home/user/project-fix`to patch a bug. The training continues uninterrupted.

**Example AI Workflow:**

```bash
# Main development is in ~/ml-project (on branch 'main')
cd ~/ml-project

# Start a training run
python train.py --config configs/baseline.yaml &

# Meanwhile, create a worktree to test a different architecture
git worktree add ~/ml-project-transformer experiment/transformer-arch

# Work on the experiment without touching the running training
cd ~/ml-project-transformer
vim src/model.py
python train.py --config configs/transformer.yaml

# When done, merge and clean up
cd ~/ml-project
git merge experiment/transformer-arch
git worktree remove ~/ml-project-transformer
```

**Nuances:**

- A branch can only be checked out in one worktree at a time. Attempting to check out the same branch in two worktrees will fail.
- Worktrees share the reflog, so `git reflog` in any worktree shows the full history.
- Worktrees are tracked in `.git/worktrees/`. If you manually delete a worktree directory, run `git worktree prune` to clean up stale entries.

**Pitfall:** Don't confuse worktrees with clones. Clones are fully independent copies. Worktrees share the same repo — a commit in one is immediately visible in another. This is the whole advantage (no duplicate data), but it means you must be mindful that operations like `git gc` affect all worktrees.

---

## 12. Submodules

### `git submodule`

Submodules let you embed one Git repository inside another. Common in AI projects for pinning dependencies, datasets, or shared libraries.

```bash
# Add a submodule
git submodule add https://github.com/org/shared-utils.git libs/shared-utils

# Clone a repo with submodules
git clone --recurse-submodules https://github.com/org/project.git

# Initialize and update submodules (after a plain clone)
git submodule update --init --recursive

# Update submodules to their latest remote commits
git submodule update --remote

# Remove a submodule
git submodule deinit libs/shared-utils
git rm libs/shared-utils
rm -rf .git/modules/libs/shared-utils
```

**Pitfall:** Submodules are notoriously confusing. They pin to a specific commit, not a branch. If someone updates the submodule, you must explicitly run `git submodule update`. Forgetting this leads to "works on my machine" bugs where one developer has a different version of the submodule. Consider alternatives like `git subtree` or package managers (pip, npm) when possible.

---

## 13. Large File Storage (Git LFS)

### `git lfs`

Git is not designed for large binary files. Git LFS replaces them with lightweight pointers in the repo and stores the actual file content on a separate server.

```bash
# Install LFS (once per machine)
git lfs install

# Track file patterns
git lfs track "*.h5"
git lfs track "*.safetensors"
git lfs track "*.parquet"
git lfs track "datasets/**"

# Check what's tracked
git lfs track

# See LFS file status
git lfs ls-files

# Ensure .gitattributes is committed
git add .gitattributes
git commit -m "Configure LFS for model and dataset files"
```

**Nuances:** LFS is essential for AI repos that store model weights, checkpoints, or datasets alongside code. Without it, every `git clone` downloads the entire history of every binary file — potentially hundreds of gigabytes.

**Pitfall:** LFS has bandwidth and storage quotas on GitHub (1 GB free storage, 1 GB/month bandwidth for free accounts). For large-scale AI assets, consider DVC (Data Version Control) or cloud storage (S3, GCS) with pointers in the repo. Hugging Face Hub also provides native Git LFS with generous quotas for model and dataset hosting.

**Pitfall:** If you accidentally committed large files without LFS, you'll need to rewrite history with `git filter-branch` or the faster BFG Repo-Cleaner. Simply adding LFS tracking after the fact won't retroactively shrink the repo.

---

## 14. .gitignore & .gitattributes

### `.gitignore`

Prevents files and directories from being tracked. This is your first line of defense against committing secrets, data, and build artifacts.

```gitignore
# Python
__pycache__/
*.pyc
*.pyo
.venv/
venv/
*.egg-info/

# Environment & secrets
.env
.env.local
*.pem
secrets.yaml

# AI/ML artifacts
*.h5
*.onnx
*.safetensors
*.pt
*.pth
checkpoints/
model_outputs/
wandb/
mlruns/

# Data
data/raw/
data/processed/
*.csv
*.parquet
*.arrow

# Jupyter
.ipynb_checkpoints/
*.ipynb  # optional: use jupytext instead

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db
```

**Nuances:** `.gitignore` only affects untracked files. If a file is already tracked, adding it to `.gitignore` won't stop Git from tracking changes. You must untrack it first: `git rm --cached secrets.yaml`.

**Pitfall:** Accidentally committed API keys or credentials are in the history forever — even if you delete the file in the next commit. If this happens, you must rotate the key immediately, then use BFG or `git filter-repo` to purge the history.

### `.gitattributes`

Controls how Git handles specific file types — merge strategies, diff behavior, line endings, and LFS.

```gitattributes
# LFS tracking
*.h5 filter=lfs diff=lfs merge=lfs -text
*.safetensors filter=lfs diff=lfs merge=lfs -text

# Ensure consistent line endings
*.py text eol=lf
*.sh text eol=lf
*.bat text eol=crlf

# Mark generated files (excluded from GitHub stats and diffs)
generated/ linguist-generated
*.lock linguist-generated

# Custom diff driver for notebooks
*.ipynb diff=jupyternotebook
```

---

## 15. Advanced: Bisect, Cherry-Pick, Reflog

### `git bisect`

Uses binary search to find the commit that introduced a bug. Incredibly powerful when model accuracy suddenly dropped somewhere in the last 50 commits.

```bash
# Start bisecting
git bisect start

# Mark the current commit as bad (bug exists here)
git bisect bad

# Mark a known-good commit
git bisect good abc1234

# Git checks out a middle commit. Test it, then:
git bisect good   # or
git bisect bad

# Repeat until Git identifies the culprit

# Automate with a test script
git bisect start HEAD abc1234
git bisect run python tests/test_accuracy.py

# Done — reset to original state
git bisect reset
```

**Nuances:** Automated bisect (`git bisect run`) is a superpower. Write a script that returns exit code 0 for "good" and 1 for "bad", and Git will find the offending commit in O(log n) steps. For AI: write a script that runs a quick eval and checks if accuracy is above a threshold.

### `git cherry-pick`

Applies a specific commit from one branch onto another, without merging the entire branch.

```bash
# Apply a single commit
git cherry-pick abc1234

# Apply multiple commits
git cherry-pick abc1234 def5678

# Cherry-pick without committing (stage only)
git cherry-pick --no-commit abc1234

# Abort if conflicts arise
git cherry-pick --abort
```

**Nuances:** Cherry-picking creates a new commit with a new hash — it duplicates the changes, it doesn't move them. If you later merge the source branch, Git is usually smart enough to avoid conflicts from the duplicate, but not always.

**AI-specific tip:** You ran an experiment on a feature branch, and one commit contains a utility function that's useful everywhere. Cherry-pick that single commit onto `main` without merging the whole experimental branch.

### `git reflog`

Records every change to HEAD — even ones that `git log` doesn't show, like resets, rebases, and branch switches. This is your safety net.

```bash
# View reflog
git reflog

# Recover a "lost" commit after a hard reset
git reflog
# Find the hash of the commit before the reset
git reset --hard abc1234   # restore to that state

# Recover a deleted branch
git reflog
git switch -c recovered-branch abc1234
```

**Nuances:** Reflog entries expire (default: 90 days for reachable, 30 for unreachable). After expiry, commits can be garbage-collected. If you need to recover something, do it sooner rather than later.

**Pitfall:** Reflog is local only. It doesn't transfer with `clone`, `fetch`, or `push`. Each worktree has its own reflog.

---

## 16. GitHub in a CI/CD Environment

GitHub Actions is the native CI/CD platform for GitHub repositories. It executes workflows defined in YAML files in response to events like pushes, pull requests, schedules, or manual triggers.

### Core Concepts

**Workflows** are YAML files stored in `.github/workflows/`. Each workflow contains one or more **jobs**, which run on **runners**(GitHub-hosted or self-hosted VMs). Jobs contain **steps** — individual commands or reusable **actions**.

### Workflow for an AI/ML Project

```yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 6 * * 1'   # Weekly retraining every Monday at 6 AM UTC
  workflow_dispatch:        # Manual trigger from the GitHub UI
    inputs:
      run_full_eval:
        description: 'Run full evaluation suite'
        type: boolean
        default: false

env:
  PYTHON_VERSION: '3.11'
  MODEL_REGISTRY: 's3://models/production'

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Cache pip dependencies
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
          restore-keys: ${{ runner.os }}-pip-

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install ruff pytest

      - name: Lint
        run: ruff check src/ tests/

      - name: Unit tests
        run: pytest tests/ -v --tb=short

  train-and-evaluate:
    needs: lint-and-test
    runs-on: ubuntu-latest
    # Use a GPU runner for real training (self-hosted)
    # runs-on: [self-hosted, gpu, linux]
    steps:
      - uses: actions/checkout@v4
        with:
          lfs: true        # Pull LFS files (model weights, data)

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run training
        env:
          WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          python train.py \
            --config configs/ci-small.yaml \
            --output-dir artifacts/model

      - name: Evaluate model
        run: |
          python evaluate.py \
            --model-path artifacts/model \
            --output artifacts/eval_results.json

      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: model-and-results
          path: artifacts/
          retention-days: 30

      - name: Comment PR with results
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const results = JSON.parse(
              fs.readFileSync('artifacts/eval_results.json', 'utf8')
            );
            const body = `## Evaluation Results\n` +
              `| Metric | Value |\n|--------|-------|\n` +
              Object.entries(results.metrics)
                .map(([k, v]) => `| ${k} | ${v} |`).join('\n');
            github.rest.issues.createComment({
              owner: context.repo.owner,
              repo: context.repo.repo,
              issue_number: context.issue.number,
              body
            });

  deploy:
    needs: train-and-evaluate
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    environment: production   # Requires manual approval if configured
    steps:
      - uses: actions/checkout@v4

      - name: Download model artifact
        uses: actions/download-artifact@v4
        with:
          name: model-and-results
          path: artifacts/

      - name: Deploy to production
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        run: |
          aws s3 cp artifacts/model/ ${{ env.MODEL_REGISTRY }}/ --recursive
          # Trigger model serving infrastructure to reload
          curl -X POST https://api.internal/models/reload \
            -H "Authorization: Bearer ${{ secrets.API_TOKEN }}"
```

### Key CI/CD Patterns for AI Projects

**1. Secrets Management**

Never hardcode API keys. Store them in GitHub repository settings under Settings → Secrets and variables → Actions. Access them via `${{ secrets.SECRET_NAME }}`.

```yaml
env:
  OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
```

**2. Self-Hosted GPU Runners**

GitHub-hosted runners don't have GPUs. For training and evaluation that requires GPU, configure self-hosted runners on your own machines or cloud instances.

```yaml
jobs:
  train:
    runs-on: [self-hosted, gpu, linux]
    steps:
      - uses: actions/checkout@v4
      - run: nvidia-smi   # verify GPU access
      - run: python train.py --device cuda
```

**3. Matrix Strategies for Testing**

Test across Python versions, frameworks, or configurations.

```yaml
jobs:
  test:
    strategy:
      matrix:
        python-version: ['3.10', '3.11', '3.12']
        framework: ['pytorch', 'jax']
      fail-fast: false   # don't cancel other jobs if one fails
    runs-on: ubuntu-latest
    steps:
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -r requirements-${{ matrix.framework }}.txt
      - run: pytest tests/
```

**4. Caching for Faster Builds**

AI dependencies are heavy (PyTorch alone is ~2 GB). Caching dramatically speeds up CI.

```yaml
- uses: actions/cache@v4
  with:
    path: |
      ~/.cache/pip
      ~/.cache/huggingface
    key: ${{ runner.os }}-deps-${{ hashFiles('requirements.txt') }}
```

**5. Branch Protection Rules**

Configure these in GitHub repository settings to enforce quality:

- Require PR reviews before merging to `main`
- Require status checks to pass (lint, tests, evaluation)
- Require branches to be up-to-date before merging
- Require signed commits
- Restrict who can push to `main`

**6. Model Performance Gates**

Add a step that fails the pipeline if model metrics drop below a threshold:

```yaml
- name: Check model quality gate
  run: |
    python -c "
    import json, sys
    results = json.load(open('artifacts/eval_results.json'))
    f1 = results['metrics']['f1']
    if f1 < 0.90:
        print(f'FAILED: F1 score {f1} is below threshold 0.90')
        sys.exit(1)
    print(f'PASSED: F1 score {f1}')
    "
```

**7. Reusable Workflows**

Factor common steps into reusable workflows to avoid duplication across repos.

```yaml
# .github/workflows/reusable-eval.yml
on:
  workflow_call:
    inputs:
      model-path:
        required: true
        type: string
    secrets:
      WANDB_API_KEY:
        required: true

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: python evaluate.py --model-path ${{ inputs.model-path }}
        env:
          WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
```

Call it from another workflow:

```yaml
jobs:
  eval:
    uses: ./.github/workflows/reusable-eval.yml
    with:
      model-path: artifacts/model
    secrets:
      WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
```

### Common CI/CD Pitfalls

1. **LFS files not pulled in CI:** Add `lfs: true` to the checkout action, or the pipeline will operate on pointer files instead of actual model weights.
2. **Timeout on training jobs:** GitHub-hosted runners have a 6-hour limit. Use self-hosted runners for longer jobs, or split training from evaluation.
3. **Secrets not available in forks:** For security, GitHub doesn't expose secrets to workflows triggered by PRs from forks. Use `pull_request_target` cautiously, or run evaluation on maintainer-triggered events only.
4. **Large artifacts:** `actions/upload-artifact` has a 5 GB limit. For model weights, push directly to S3, GCS, Hugging Face Hub, or a model registry within the pipeline.
5. **Non-deterministic tests:** AI tests involving floating-point operations can be flaky across environments. Use tolerances (`np.allclose` with `atol`/`rtol`) rather than exact equality.

---

## Quick Reference: Command Cheat Sheet

|Task|Command|
|---|---|
|Initialize repo|`git init`|
|Clone (shallow)|`git clone --depth 1 <url>`|
|Stage interactively|`git add -p`|
|Amend last commit|`git commit --amend --no-edit`|
|Switch branch|`git switch <branch>`|
|Create + switch|`git switch -c <branch>`|
|Merge (no fast-forward)|`git merge --no-ff <branch>`|
|Interactive rebase|`git rebase -i HEAD~n`|
|Safe force push|`git push --force-with-lease`|
|Stash with message|`git stash push -m "msg"`|
|Find bug via bisect|`git bisect start` / `run`|
|Recover lost commit|`git reflog` → `git reset --hard <hash>`|
|Parallel worktree|`git worktree add <path> <branch>`|
|Track large files|`git lfs track "*.h5"`|
|Search code changes|`git log -S "search term"`|

---

_Last updated: April 2026_