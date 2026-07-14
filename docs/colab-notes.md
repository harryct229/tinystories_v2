# Colab Run Notes (training via the `colab` CLI)

Project-specific gotchas hit running the real training on Colab. General CLI
usage is in `colab skill`; this captures the fixes that were **not** obvious.
The SFT run used the one-command bootstrap `scripts/sft_colab.py` — adapt the
same pattern for the RM/GRPO runs. (First captured 2026-07-12 from the issue 03
SFT run: 800 steps on an L4, model at `hf://congthanh991/tinystories-v2-sft`.)

## Before you run

- **Push `main` to GitHub first.** The VM clones from
  `github.com/harryct229/tinystories_v2`, so any local-only commits are missing
  on the VM. `git push origin main` before provisioning.
- **Secrets via upload, not exec.** `colab upload .env /content/tinystories_v2/.env`.
  Don't pass tokens inside a `colab exec` command — they get written to the CLI
  history logs. `load_env()` reads `.env`; on the notebook path, the Colab
  Secrets cell (`os.environ[...] = userdata.get(...)`) also works and the `!`/
  subprocess inherits it.
- **Private Hub repos need `HF_TOKEN`** (tokenizer, data, checkpoints are all
  private). The bootstrap downloads only `tokenizer.json` + `splits/sft.jsonl`
  (single files), not the whole multi-hundred-MB data repo.

## Setup on the VM

- **CWD shadows the package.** `exec`/`console` default to `/content`, and the
  cloned repo dir `/content/tinystories_v2` **shadows** the installed
  `tinystories_v2` package (src layout) when you import from `/content` — you
  get an empty namespace with no `__version__`. Always run from inside the repo:
  `cd /content/tinystories_v2 && python …`. The bootstrap already does this.
- **`pip install -e .` keeps Colab's CUDA torch** (2.11.0+cu128, satisfies our
  `>=2.6`) — it does not downgrade to a CPU build. `[track]` is unnecessary for
  W&B: Colab preinstalls `wandb`, so a dashboard appears automatically once
  `WANDB_API_KEY` is in `.env` (the pretrain run just didn't have the key set).

## Driving the CLI reliably

- **Use `colab exec` + Python `subprocess.run` for shell ops.** Piping shell
  through `colab console` (tmux) was unreliable here and its output carries
  terminal-control bytes; `exec` with captured `subprocess` output is clean.
- **Never run a multi-minute command inline in one `exec`.** With
  `capture_output=True` it prints nothing until it returns, and the CLI's reply
  window times out and disconnects (the kernel keeps running, so you're blind).
  Background it with a log + exit marker, then poll:
  ```python
  subprocess.Popen(
      "cd /content/tinystories_v2 && "
      "(python scripts/sft_colab.py > /content/run.log 2>&1; "
      "echo EXIT_$? >> /content/run.log) &", shell=True)
  ```
  Poll `run.log` for the `EXIT_` marker with short exec calls.
- **`TimeoutError: Timeout waiting for reply`** = your poll payload slept too
  long on the VM (kernel busy longer than the CLI's ~30–60 s reply window).
  Keep any VM-side wait to < ~20 s per exec call and poll repeatedly instead.
- **`ConnectionResetError` / `Connection aborted`** on `exec`/`upload` are
  intermittent. Wrap CLI calls in a retry loop (3–6 tries, short backoff) — one
  fired on the pip launch this run.

## Preemption & completion

- **The VM dies when idle.** L4s are reaped once the kernel goes idle (and
  preempt ~hourly regardless). This run's VM was reclaimed *right after* the SFT
  run finished — local `/content/...` artifacts vanished and `exec` returned
  404/401 ("session lost").
- **The Hub is the source of truth, not the VM.** Checkpoints sync every
  `checkpoint_every` steps; `manifest.json` (with `final_step`) is written only
  on a clean finish. To know where a run got to, list the Hub repo's
  `checkpoints/step_*.pt` + read `manifest.json` — don't rely on the VM being
  alive. (Locally: `HF_HUB_DISABLE_XET=1 python -c "…HfApi().list_repo_files(…)"`.)
- **Recover by re-running.** On a fresh VM, `python scripts/sft_colab.py` (which
  runs `ts2-sft --resume`) pulls the last Hub checkpoint and continues —
  idempotent, so just re-run after any preemption.
- **Always stop the VM.** `colab stop -s <name>` when done, or verify
  `colab sessions` shows nothing. A reaped session lingers as an orphan `[?]`
  (already released; `colab stop` on it says "not found").

## Hub downloads: Xet is unreliable, and `HF_HUB_DISABLE_XET=1` is not a
   complete fix (issue 07 eval run, 2026-07-14)

- **Symptoms seen:** Xet-backed downloads hung indefinitely on a 359MB
  checkpoint (fixed once by setting `HF_HUB_DISABLE_XET=1`); later, a
  *different* download (a `transformers.from_pretrained` sharded checkpoint
  fetch) still routed through Xet and 403'd on an expired presigned
  `xet-bridge` URL — with `HF_HUB_DISABLE_XET=1` set in the same process.
- **Why the env var alone isn't enough:** `HF_HUB_DISABLE_XET` is read by
  `huggingface_hub.file_download`'s own `hf_hub_download`/`snapshot_download`
  path, but not every caller goes through that exact function — a sharded
  `from_pretrained` load can route through a different internal path that
  doesn't consult the same flag. Verifying `is_xet_available()` returns
  `False` in a standalone script on the VM is not proof every download in
  the actual job respects it.
- **The reliable fix: make Xet physically unavailable.**
  `pip uninstall -qy hf-xet hf_xet` (and `rm -rf
  /root/.cache/huggingface/xet` to clear any stale cached state) right after
  installing the package's extras, before running the stage. With the
  package gone, `is_package_available("hf_xet")` is `False` everywhere,
  independent of which code path checks it.
- Plain-HTTP fallback downloads at ~18 MB/s (not the ~100 MB/s Xet promises
  when it works) — a 16GB judge model takes ~15 min. Budget for it; a
  resumable stage (cached completions, streamed judgments — see
  `eval.py`'s `resume=True`) matters more than raw download speed once Xet
  is out of the picture.
