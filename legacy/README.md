# Legacy (v1)

`spot_factory_inspection.py` is the first version of this project —
*A Context-Aware Anomaly Decision-Making Framework for SPOT Patrol Robots
Based on LLM Reasoning*. It is **not** part of the current pipeline and is not
what the paper's numbers come from; the v2 entry point is
[`src/spot_factory_v2.py`](../src/spot_factory_v2.py).

It is kept here because v2 was built on its skeleton and still inherits several
pieces from it — `SpotPatrolController` (waypoint patrol), `ActionExecutor`
(APPROACH / WAIT_AND_OBSERVE / KEEP_DISTANCE with arc observation), the Isaac
Sim scene setup, the anomaly-zone visualization, and the base LLM call
structure. Reading it is the quickest way to see what changed and why.

What v2 changed: 7 anomaly types (4 in-distribution + 3 held-out OOD) with new
attribute pools, a two-tier decision spec with `tier2_escalate` /
`tier2_action` / `tier2_confidence`, a Fixed policy extended through Tier 2,
escalation fields in the metrics, and simultaneous collection of all three
conditions in a single run.

Full comparison: [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md).

The v1 run outputs (per-model logs, metrics CSV/JSON, decision-time plots) are
generated artifacts and are not tracked here.
