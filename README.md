# spot-llm-inspection

**불확실성 인지 기반 선택적 에스컬레이션을 통한 LLM-인간 협력 공장 점검 의사결정 프레임워크**

A Boston Dynamics SPOT robot patrols a simulated factory in Isaac Sim. When it
detects an anomaly, an LLM makes a two-tier decision grounded in Korean
industrial-safety law and, based on its own confidence, escalates the
high-stakes half to a human operator.

The claim is not that the LLM is more accurate — it is that **at the same
amount of human effort, confidence-guided delegation is safer**, and that it
still works on anomaly types nobody wrote a rule for.

## Framework

```
SPOT patrol → anomaly detected
  ↓
[Tier 1 — observation stance]   APPROACH · WAIT_AND_OBSERVE · KEEP_DISTANCE
  ↓  (SPOT physically executes the chosen stance)
[Tier 2 — resolution]           EMERGENCY · MAINTENANCE · CONTINUE · FALSE_ALARM
  ├─ fixed      rule-based
  ├─ llm_full   LLM decides directly, never escalates
  └─ llm_hitl   low confidence → escalate to the human oracle
```

All three conditions are scored on the same events in one run, so the
comparison is paired. SPOT's motion follows the `llm_hitl` decision. Legal
principles (화재예방법 제40조, 소방기본법 제20조 등) are injected as *prompt
principles* rather than lookup rules — that is what lets the LLM generalize to
unseen anomaly types.

Full spec, severity formulas and GT rules: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Results

4 LLMs (Haiku 4.5, Sonnet 4.6, GPT-4o mini, Llama 3.1) × 10 laps = 196 decision
events. 4 in-distribution anomaly types + 3 held out as OOD.

| | Fixed (SOP) | LLM-Full | LLM+HiTL |
|---|---|---|---|
| Tier 2 dangerous under-response | 44.9 % | 5.6 % | **1.5 %** |
| EMERGENCY miss rate | 76.0 % | 7.0 % | **0.0 %** |
| EMERGENCY miss on OOD types | **100 %** | 9.4 % | **0.0 %** |

Confidence ranking reaches ≥ 98 % safety at 25 % human help, 100 % at 65 %.
Raw confidence is badly overconfident (ECE 30.5 %); temperature scaling at
T = 7.0 fixes the mean (ECE 4.7 %) but, being monotonic, leaves the delegation
curve unchanged.

**Caveats**: ground truth is an author-written rule set, not expert labels; the
oracle is a *perfect* human, so LLM+HiTL is an upper bound; N = 196, single
seed; the LLM's binary self-escalation flag fires far too often (85 %) — use
the confidence *ranking* with a calibrated threshold instead.

Details: [`docs/PAPER_SUMMARY.md`](docs/PAPER_SUMMARY.md), [`docs/PAPER_DOSSIER.md`](docs/PAPER_DOSSIER.md).

## Setup

**Analysis only** — regenerates every table and figure from a metrics CSV. No
GPU, no Isaac Sim, no API keys:

```bash
pip install -r requirements-analysis.txt
python tests/smoke_test_v2.py
```

**Full simulation** — additionally needs [Isaac Sim 5.0](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/)
(`carb`, `omni`, `pxr` come from it, not pip):

```bash
pip install -r requirements.txt
cp .env.example .env      # then fill in API keys
```

| Alias | Provider | Needs |
|---|---|---|
| `haiku`, `sonnet` | Anthropic | `ANTHROPIC_API_KEY` |
| `gpt4o` | OpenAI | `OPENAI_API_KEY` |
| `llama` | Ollama | `ollama serve && ollama pull llama3.1` |

## Run

```bash
# Simulation — every anomaly event scores all three conditions
python src/spot_factory_v2.py --models haiku sonnet gpt4o llama --laps 10 --seed 42

# Analysis (run from the repo root)
python analysis/make_paper_figures.py --input results/metrics_raw.csv
python analysis/export_excel.py       --input results/metrics_raw.csv
```

Other flags: `--headless`, `--follow-cam`, `--speed {1,2,3,4}`, `--record`
(one PNG per frame — several GB per run), `--output`, `--log-dir`.

## Layout

```
src/           anomaly_config · llm_decision · oracle · metrics_collector · spot_factory_v2
analysis/      CSV in, tables and figures out — no Isaac Sim
sim/           standalone Isaac Sim scenes, not part of the pipeline
legacy/        v1 of the project, kept for reference
tests/         smoke_test_v2.py — validates configs, emits a synthetic CSV
policies/      pretrained SPOT locomotion policy + env config
docs/          architecture, paper dossier, results guide
```

`results/` and recordings are generated output and are not tracked.

## License

MIT — see [`LICENSE`](LICENSE).
