# k2loop 0.2 — a fully local, always-on Tree-of-Thoughts coding agent

[Türkçe README](README.tr.md)

Designed for a single RTX 3080 Ti (12 GB) and 32 GB of RAM. You write a goal. The agent first **searches the web for
an existing solution**, then plans and codes with **Tree of Thoughts / Self-Consistency**, and you watch every step
live in Open WebUI. Nothing leaves your machine (the internet is used only for search).

```
 Open WebUI :3000 (live agent)   Phoenix :6006 (ToT trace tree)   LDR :5000 (research)   llama-swap :8080/ui
        │ OpenAI API                     ▲ OpenTelemetry              ▲                          │
   k2loop serve :8090 — goal queue: RESEARCH → PLAN (ToT) → CODE (ToT+SC) → REVIEW               │
        │                                   │                          │                         │
   GPU (3080 Ti) ─ llama-swap ─────────────────────────────────────────────────────────────────────┘
     K2 Horizon 7B: code candidates (Aider)  ⇄  strong model: plan / escalation / final review
   CPU ─ helper model (Qwen3 4B, always on) · Laya + SemIf (decisions) · check.sh · MoE experts · SearXNG · LDR
```

| Component | Role | Runs on |
|---|---|---|
| **K2 Horizon 7B** | Code candidates, fix hypotheses, review votes | GPU |
| **Strong model** | Plan (3 alternatives + judge), escalation, final review. `k2loop strong moe\|dense` | GPU + RAM, swapped in for K2 when needed |
| ↳ `moe`: **Qwen3.6-35B-A3B** UD-IQ4_XS | 35B parameters, ~3B active per token: experts in RAM/CPU, the rest on GPU → fast | recommended |
| ↳ `dense`: **Qwen3.8 27B** Q4_K_M | does not fit in 12 GB; some layers on CPU → ~4 tokens/s | old default |
| **Helper model** (Qwen3 4B) | Goal classification, hypothesis scoring, research summaries, chat titles | CPU, alongside K2 |
| **Laya** | Test-output classification, ToT branch pruning, retry/escalate decisions | CPU |
| **llama-swap** | All models behind one address; swaps K2 ⇄ Qwen on the GPU, keeps the CPU model loaded | local |
| **Local Deep Research + SearXNG** | Multi-round web research, sourced reports, scheduled subscriptions | CPU |
| **Open WebUI** | Chat UI; every agent step streams into it live | docker |
| **Phoenix** | Every candidate / branch / Laya decision / model call as a nested trace tree | local |
| **Aider** | K2's file-editing tool | local |
| **Bernstein 3.20.0** | (optional) runs multi-task plans in parallel worktrees with `k2loop go` | local |

## Install

```bash
git clone https://github.com/ELNEMO-cmd/k2loop.git && cd k2loop
./install.sh                 # strong model defaults to MoE (--strong dense: the old Qwen3.8 27B)
```

To update: `git pull && ./install.sh --skip-root` (finished steps are skipped, services restart with the new settings).
Change history: [CHANGELOG.md](CHANGELOG.md). Takes 30–60 min and ~55 GB of disk. Log: `~/k2loop-install.log`.

After installing:
1. Open `http://127.0.0.1:3000` → create the first Open WebUI account → pick the model **k2agent**.
2. (Recommended) Open `http://127.0.0.1:5000` → create a Local Deep Research account and put it in
   `~/.config/k2loop/secrets.env` as `LDR_USER=...` and `LDR_PASSWORD=...`, then
   `systemctl --user restart k2-agent`. Without it, research falls back to plain SearXNG search.
3. Run `k2loop doctor --full` — everything should show ✓.

Options: `--strong moe|dense` · `--no-local` (no strong model) · `--no-web` (no SearXNG/LDR/Open WebUI) ·
`--hybrid` (old mode with Claude Code) · `--quant Q5_K_M`.

The same package runs on Ubuntu 24.04 under WSL2 on Windows; see the Windows section in [README.tr.md](README.tr.md).

## Usage

Type to the **k2agent** model in Open WebUI:

- `add a command to ~/myproject that draws a chart from CSV, with tests` → **code** goal
- `compare the best code-search tools for local LLMs` → **research** goal
- `make a 3-month study plan for German B1` → **plan** goal (any topic)
- `/durum` (status) · `/iptal` (cancel) · `/izle` (attach to the running goal) · `/yardim` (help)

A **plan** goal: research the topic → 3 planners (Pragmatic, Comprehensive, Risk-first) each write a plan with K2 →
they **debate**: each reads the others' plans, critiques them in a short visible message and revises its own → a judge
(Qwen) scores them → the winner is merged with the best ideas of the others. The debate transcript and all candidates
are saved under `~/.local/share/k2loop/agent/plans/`. Settings: `config.yaml` → `plan.*`.

Goals keep running if you close the browser; extra messages queue up. From the terminal: `k2loop ask "goal"`.

A **code** goal:
1. **Research:** is there an existing project/library, what license, what can be reused? → `.k2loop/plans/*-research.md`.
2. **Plan (ToT):** Qwen proposes three routes — *use existing* / *adapt* / *write from scratch* — and a judge picks one.
3. **Each step (ToT + Self-Consistency):** K2 writes 3 candidates (the previous one's tests run on the CPU while the next
   is generated) → the smallest passing diff wins; if none pass, the failure is classified (Laya #1) → 3 fix hypotheses →
   the helper model scores them → 2 are chosen (Laya #2) → tried; between levels: continue / escalate / ask a human
   (Laya #3); escalate to Qwen if needed.
4. **Review:** 3 votes + a final check by Qwen. The winning diff is committed to a `k2loop/<date>-<goal>` branch.
   You merge it into main yourself.

The agent will not start if the project has uncommitted changes, and it generates `.k2loop/check.sh` if missing.
**`.k2loop/check.sh` is the system's referee** — edit and commit it as you like.

Older commands still work: `k2loop init`, `k2loop solve "single task"`, `k2loop plan/run/go` (Bernstein),
`k2loop mode`, `k2loop laya-eval`, `laya-export`, `laya-mode`.

## Monitoring

| What | Where |
|---|---|
| Live agent stream | Open WebUI `http://127.0.0.1:3000` |
| ToT tree, model calls, timings | Phoenix `http://127.0.0.1:6006` (project: k2loop) |
| Loaded model, token speeds, server logs | llama-swap `http://127.0.0.1:8080/ui` |
| Research reports, subscriptions | Local Deep Research `http://127.0.0.1:5000` |
| Raw event log | `~/.local/share/k2loop/agent/events.jsonl` |
| All files of one task | `~/.local/share/k2loop/runs/<time>-.../` (Aider logs, check output, patches) |

## Decisions: rule vs Laya vs SemIf

Three decisions in ToT debugging (failure type, branch pruning, continue/escalate/human) have three candidate deciders:
- **rule**: deterministic; makes the decision at the start;
- **Laya**: a trained decision model (CPU service);
- **SemIf** ([SemIf](https://github.com/theoleecj/semif)): the CPU helper model sees the options as letters and the
  probability of each option is read in a single pass (no text generation).

At first everything runs in **shadow mode**: the rule decides while Laya and SemIf predict the same decision and log
it; outcomes (did the branch pass its tests, did escalation help) become labels automatically. `k2loop laya-eval`
compares all three (accuracy + threshold sweep); `--apply` activates, per decision, whichever beats the rule most
(at least 150 labels). Manually: `k2loop laya-mode retry_escalate semif` · `... active` (Laya) · `... shadow` (rule).
`k2loop laya-export` exports fine-tuning data.

## Configuration

`~/.config/k2loop/config.yaml` (re-read for every goal). Key sections: `tot.*` (candidates/branches/votes, timeouts,
`overlap_checks`), `roles.*` (side jobs on GPU or CPU), `agent.*`, `research.*`. Model command lines live in
`~/.config/k2loop/llama-swap.yaml` (after editing: `systemctl --user restart k2-swap`). Keys go in `secrets.env`.

## Troubleshooting

- **Models:** `journalctl --user -u k2-swap -f` or `http://127.0.0.1:8080/ui`. To re-measure K2's context:
  `rm ~/.local/share/k2loop/llm_state.env && systemctl --user restart k2-swap`. If your monitor is plugged into the
  3080 Ti, the desktop also uses VRAM; plug it into the motherboard (iGPU) if you can.
- **Agent:** `journalctl --user -u k2-agent -f`; crash details in `~/.local/share/k2loop/agent/crash-*.txt`.
- **Open WebUI does not show k2agent:** Settings → Connections → `http://127.0.0.1:8090/v1`, key `AGENT_API_KEY` from `secrets.env`.
- **Research returns nothing:** `docker ps` (is k2-searxng running?), `systemctl --user status k2-ldr`, the LDR account in `secrets.env`.
- **CPU overloaded:** set `K2LOOP_HELPER_THREADS=6` for the helper model in `llama-swap.yaml`.

## Known limits

- The 7B model is good at small, focused steps; planner steps are deliberately kept small.
- K2 Horizon often answers Aider's edit requests with its own tool-call format (`<ifm|tool_call>…`) or whole-file blocks,
  which Aider cannot apply. `k2adapter` sits between K2 and Aider and converts the replies to Aider's SEARCH/REPLACE
  format (`config.yaml` → `aider.k2_adapter: true`). With it off, K2's candidates come back empty.
- Every time Qwen loads, K2 leaves the GPU (a few minutes). Qwen runs half in RAM, so it is slow; the per-task budget
  is 200 min. With a second graphics card you can put both in separate groups in `llama-swap.yaml` and drop the swap.
- Text from the internet is marked as untrusted data and handed to the planner that way; still, review the branches the
  agent produces before merging.
- For embedded projects, verification is limited to builds + host tests.
- K2 uses the MBZUAI-IFM fork of llama.cpp. Bernstein is beta software (pinned to 3.20.0).

Uninstall: `./uninstall.sh`

## License

k2loop's own code is [MIT](LICENSE) licensed. Tools and models that `install.sh` downloads are not in this repo and
come with their own licenses, e.g. Aider, Open WebUI, llama.cpp, llama-swap, SearXNG, Local Deep Research (open
source), Arize Phoenix (Elastic License 2.0), and model weights (each model's own license, e.g. K2 Horizon, Qwen).
Check them if you redistribute.
