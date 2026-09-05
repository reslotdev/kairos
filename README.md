# Kairos: A Foundation Model for the Language of Slot Play

<p align="center">
  <a href="https://reslot.dev">reslot.dev</a> ·
  <a href="https://reslot.dev/kairos">Kairos</a> ·
  <a href="https://reslot.dev/blog/kairos-a-player-world-model-for-slot-play">Paper</a> ·
  <a href="https://reslot.dev/blog">Research blog</a> ·
  <a href="https://reslot.dev/simulator/par-sheet">PAR Sheet simulator</a> ·
  <a href="https://reslot.dev/simulator/weight-table">Weight Table simulator</a>
</p>

Kairos is a player world model for slot engines. It treats the round stream a slot engine emits as a language (discrete tokenization, coarse-then-fine autoregression, pretrain-then-finetune, sampled multi-path inference) and learns the one part of that stream that is genuinely uncertain: **the player**, not the RNG.

Each round is written as `[control] → [action] → [outcome]`. Kairos models the action pair conditioned on everything before it; a deterministic engine supplies the outcome under any candidate configuration. Composed together they form a calibrated economy simulator that answers how retention responds to *where and when* giveaway is spent, and a backbone whose embeddings replace hand-written player tags, exit-hazard rules and lifetime-value heuristics.

## Read

- 📄 **Paper** — [Kairos: A Foundation Model for the Language of Slot Play](https://reslot.dev/blog/kairos-a-player-world-model-for-slot-play)
- 📊 **Simulation studies** — [What the Tables Decide, and What the Players Decide](https://reslot.dev/blog/what-the-tables-decide-and-what-the-players-decide) · [The Population Is the Parameter](https://reslot.dev/blog/the-population-is-the-parameter)
- 🧪 **Experiments** — scripts, the v2 behavioural generator and results behind both studies, in [`experiments/`](experiments/)
- 🧭 **Background** — [Loop Engineering for Self-Improving Slot Agents](https://reslot.dev/blog/loop-engineering-for-self-improving-slot-agents)
- 🎛 **Simulators** — [PAR Sheet](https://reslot.dev/simulator/par-sheet) · [Weight Table](https://reslot.dev/simulator/weight-table), run in the browser, nothing uploaded

> **Status: pre-release.** This repository holds the tokenization pipeline, the synthetic pre-training generator, the model and the evaluation protocol. No production weights are published yet; the model zoo below lists target configurations, not released checkpoints.

---

## Highlights

- **Learns the player, not the payout.** Outcome tokens enter the context as inputs and carry no loss. The entire gradient goes to the player's next action, factorized coarse-then-fine the way a player decides: first whether to stop or continue, then how much.
- **Composes with the engine.** Rollouts alternate a sampled action from Kairos with an outcome drawn by the engine under the configuration being tested, so any control table can be evaluated counterfactually.
- **Attribution-aware by construction.** Under deterministic dynamic RTP control, treatment assignment depends only on logged state; carrying that state in context makes the learned conditional valid wherever the data covers the configuration.
- **Two granularities.** A spin-level model (2048-round context) for sessions and bets; a day-level model over per-player-day "K-lines" for D2, D7 and lifetime value.
- **Pooling across games and currencies.** Per-player, per-session normalization (bet over denomination, balance over session-opening balance, gaps in log space) lets every title an operator runs enter one model; absolute bet and balance are kept alongside, because suppression bands are written in currency.
- **Evaluation that ends in experiment replay.** Four suites, from next-action likelihood to simulator fidelity; the final one holds out a completed split experiment and scores the model's off-policy lift estimate against the measured lift.

---

## Architecture

```
                 ┌──────────────────────────────────────────────────────┐
  round t        │  control c_t     action a_t          outcome o_t     │
                 │  template, tag,  coarse: continue /  multiplier,     │
                 │  spin buckets,   raise / lower /     round type,     │
                 │  rolling RTP,    stop / buy bonus…   control flag,   │
                 │  config version  fine: bet bucket ×  pre-suppress,   │
                 │                  gap bucket          balance ratio   │
                 └──────┬───────────────┬───────────────────┬───────────┘
                        │               │                   │
                        ▼               ▼                   ▼
                 ┌──────────────────────────────────────────────────────┐
                 │   field embeddings (summed) + calendar embeddings    │
                 │   decoder-only transformer · RoPE · pre-LN RMSNorm   │
                 └──────────────────────────┬───────────────────────────┘
                                            │ h_t
                     ┌──────────────────────┴──────────────────────┐
                     ▼                                             ▼
             coarse action head                       cross-attn(q = emb(â_c), kv = h_t)
             p(a_c | h_<t, c_t)                       fine action head  p(a_f | ·, a_c)
                     │                                             │
                     └──────────── sampled action ─────────────────┘
                                            │
                                            ▼
                         engine.draw(action, control, config) → o_t
                                            │
                                            └──── append, repeat ────▶
```

Loss is computed on `a_c` and `a_f` only. The fine head trains on the model's own sampled coarse token so training matches multi-step rollout.

Multi-task heads on the backbone embedding (stage E): session-exit hazard, D2, lifetime value, a CATE head with a doubly-robust target, and unsupervised player clusters.

---

## Repository layout

```
kairos/
├── kairos/
│   ├── tokenize/        # ledger → token groups: bucketing, normalization, session gaps
│   ├── data/            # datasets, player/time splits, config-version alignment
│   ├── synth/           # parameter-randomized behavioral families + engine driver
│   ├── model/           # Kairos transformer, dual head, day-level encoder, task heads
│   ├── engine/          # outcome sampler over MAIN tables and the control state machine
│   ├── rollout/         # world-model × engine rollouts, multi-path sampling
│   └── eval/            # fidelity, hazard AUC, CRPS, discriminative score, experiment replay
├── configs/             # model sizes, training stages A–E, bucket definitions
├── scripts/
│   ├── build_corpus.py
│   ├── pretrain_synthetic.py
│   ├── pretrain_provider.py
│   ├── finetune_game.py
│   ├── finetune_intervention.py
│   └── evaluate.py
├── schema/              # ledger schema (payout, login, group assignment, config versions)
└── docs/
```

---

## Installation

```bash
git clone https://github.com/reslotdev/kairos.git
cd kairos
pip install -e .
```

Python ≥ 3.10, PyTorch ≥ 2.2. Training scripts use `torchrun` for multi-GPU.

---

## Ledger schema

Kairos consumes the same records a Provider delivers for phase-one control optimization. Required fields per round:

| Field | Type | Notes |
|---|---|---|
| `round_id` | string | idempotency key |
| `player_id` | string | pseudonymized |
| `game_id` | string | |
| `timestamp` | ISO-8601 with offset | daily resets depend on the platform's midnight |
| `bet_amount`, `payout_amount` | decimal | payout is post-suppression, bonus included |
| `currency`, `denomination` | string, decimal | |
| `round_type` | `BASE` / `BONUS` / `BUY_BONUS` | |
| `bonus_source` | `NATURAL` / `BUY` / `RICH_CARD` | when `round_type ≠ BASE` |
| `is_free_round` | 0/1 | |
| `template_id`, `tag_id` | int | template and tag in force for **this** round |
| `adjust_flag` | `NON` / `CUMULATIVE` / `DAILY` / `REROLL` / `DISCARD` | |
| `pre_suppress_amount` | decimal | when `adjust_flag ∈ {REROLL, DISCARD}` |
| `player_cum_round`, `player_day_round` | int | |
| `balance_before_next_bet` | decimal | optional, enables balance-rescue heads |
| `group_id`, `config_version` | string | experiment arm; configuration alignment |

Login records (`player_id`, `timestamp`, `platform`, `group_id`) and a configuration change history (`config_version` → tables, effective range) complete the input. See `schema/` for the full dictionary.

---

## Quickstart

### 1. Build a corpus

```bash
python scripts/build_corpus.py \
  --ledger data/payout_records_*.csv \
  --logins data/login_records_*.csv \
  --config-history data/config_versions.json \
  --out corpus/banana
```

Bucketing and normalization rules live in `configs/buckets.yaml`. Splits are by player, then by time.

### 2. Synthetic pre-training (no Provider data needed)

```bash
python scripts/pretrain_synthetic.py \
  --tables config/v0602+V4/ \
  --families configs/synth/families.yaml \
  --model configs/model/kairos-mini.yaml \
  --steps 200000
```

`families.yaml` defines prior distributions over behavioral parameters (stop-loss, chase tendency, bet escalation, session budget, per-player loyalty). The engine runs the real control tables; players are sampled from the families.

### 3. Fine-tune on a game

```bash
python scripts/finetune_game.py \
  --init checkpoints/kairos-mini-synth.pt \
  --corpus corpus/banana \
  --lr 4e-5 --epochs 30
```

### 4. Roll out under a candidate configuration

```python
from kairos import KairosModel, Engine, Rollout

model = KairosModel.from_pretrained("checkpoints/kairos-mini-banana.pt")
engine = Engine.from_tables("config/candidate_R2/")

sim = Rollout(model, engine, temperature=1.0, top_p=0.95, sample_count=10)
report = sim.run(cohort="new_players", days=28, players=5000)

print(report.retention.d2)          # mean and interval across sampled paths
print(report.cost.net_giveaway)     # (A + B − C − D) / turnover, upper bound
print(report.fidelity)              # only meaningful against a real reference cohort
```

`sample_count` is the test-time-scaling knob: the mean across paths is the estimate, the spread is the interval a guardrail evaluates against.

### 5. Evaluate

```bash
python scripts/evaluate.py --model checkpoints/kairos-mini-banana.pt --corpus corpus/banana --suite all
```

| Suite | What it measures | Analogue in forecasting benchmarks |
|---|---|---|
| `action` | next-action NLL and calibration; CRPS on session length; D2 hazard AUC | price / return IC, RankIC |
| `volatility` | per-player bet volatility and daily turnover distribution error | volatility MAE, R² |
| `fidelity` | distance of rolled-out vs. real session length, turnover, realized RTP, giveaway accounts; discriminative score | synthetic K-line fidelity, TSTR |
| `replay` | hold out a completed split experiment; DR / FQE off-policy estimate vs. measured lift | backtest AER, IR |

Run `fidelity` before `replay`. Replay is the only suite that demonstrates usefulness rather than fit.

---

## Model zoo (targets)

| Model | Layers | d_model | Heads | Context | Params | Status |
|---|---|---|---|---|---|---|
| Kairos-mini | 6 | 192 | 6 | 2048 | ≈ 4M | in training on synthetic corpus |
| Kairos-small | 8 | 512 | 8 | 2048 | ≈ 20M | planned |

Larger tiers are not planned until the pooled corpus justifies them.

---

## Training stages

| Stage | Script | Data |
|---|---|---|
| A. Synthetic pre-training | `pretrain_synthetic.py` | randomized behavioral families × real tables |
| B. Provider-wide pre-training | `pretrain_provider.py` | all titles, normalized |
| C. Target-game fine-tuning | `finetune_game.py` | ≥ 4 contiguous weeks of the game |
| D. Intervention fine-tuning | `finetune_intervention.py` | split-experiment arms, treatment-balanced |
| E. Task heads | `finetune_game.py --heads hazard,d2,ltv,cate` | same |

Rules that are enforced in code: loss on action tokens only; player-first then time splits; configuration versions aligned to rounds; sessions never pre-segmented; synthetic data excluded from every validation set.

---

## What Kairos does not do

- It never sits in the payout path. The engine draws every outcome.
- It never modifies the math model. Paytables and weights stay with the operator's numerical designers.
- Its outputs always pass through a deterministic guardrail layer: portfolio cost cap, per-player giveaway cap, GGR circuit breaker, and a long-run realized-RTP floor.

---

## Roadmap

- [x] Ledger schema and tokenization rules
- [ ] Parameter-randomized behavioral families and synthetic corpus generator
- [ ] Kairos-mini trained on the synthetic corpus, scored against the hand-written simulator player on the `fidelity` suite
- [ ] Provider-wide pre-training
- [ ] Intervention fine-tuning and CATE head on first split-experiment data
- [ ] Experiment replay on a held-out experiment
- [ ] Learned BSQ tokenizer over the continuous four-dimensional player state (v2)

---

## Citation

```bibtex
@techreport{reslot2026kairos,
  title  = {Kairos: A Foundation Model for the Language of Slot Play},
  author = {reSlot Research},
  year   = {2026},
  url    = {https://reslot.dev/blog/kairos-a-player-world-model-for-slot-play}
}
```

---

## License

Code is released under the MIT License. Model weights, when released, will carry their own license. The paper, the simulation studies and the reslot.dev site content are © 2026 reSlot Inc., all rights reserved.

## Acknowledgements

The control state machine, ledger schema and guardrail layer come from reSlot's dynamic RTP control work; the behavioural generator borrows its mechanisms from the gambling-behaviour literature cited in *The Population Is the Parameter*.

---

<p align="center">
  © 2026 <a href="https://reslot.dev">reSlot Inc.</a> · Kairos is built by <a href="https://reslot.dev">reSlot</a>, slot math that knows the player.<br>
  <a href="https://reslot.dev/kairos">Model page</a> · <a href="https://reslot.dev/blog">Blog</a> · <a href="https://reslot.dev/solutions/certified">For certified studios</a> · <a href="https://reslot.dev/solutions/dynamic-control">For dynamic-control studios</a> · <a href="https://reslot.dev/solutions/llm-studios">For studios designing with an LLM</a> · <a href="mailto:mail@reslot.dev">mail@reslot.dev</a>
</p>
