# Lightning pathfinding benchmark

A tool to **reproduce and measure** how well Electrum charts Lightning routes,
without sending any payment. It exists to quantify the failures reported in
[spesmilo/electrum#10443][issue] — payments above ~10k sat failing with
"insufficient fee", and routes that are far longer than necessary — so that a
fix to the router can be benchmarked against the current behaviour.

Scope: **non-trampoline (gossip) pathfinding only.** Trampoline payments do not
use the gossip graph (the trampoline node does the routing), so they are out of
scope for this tool; that failure mode needs a separate measurement.

## What it measures

For a set of payment amounts (default: 100 / 1,000 / 10,000 / 100,000 sat), it
samples well-connected source nodes and a connectivity-stratified set of
destinations, asks `LNPathFinder.find_route()` for a route, and records per
amount bucket:

- **`no_route%`** — attempts where no route was found at all
- **`over_budget%`** — a route was found but its fee exceeds the budget (5% by
  default, matching the bug report)
- **`fail%`** — `no_route + over_budget` (the payment could not proceed)
- **`success%`** — found and within budget
- **mean / median fee rate** — routing fee as a fraction of the amount
- **mean hops** and **mean/median excess hops** — how many more hops the chosen
  route has than the fewest-*feasible*-hops "ideal route" (an independent BFS
  over the same graph). This is the "circuitous routes" symptom from the report.

The methodology is described in detail at the top of
`electrum/pathfinding_benchmark.py`. The "ideal route" baseline is minimum
*hops*, not minimum *fee* (see that file for why); fee quality is captured by the
absolute fee-rate metrics and the over-budget flag.

## Usage

### 1. Capture a gossip snapshot

Pathfinding quality depends on the gossip graph, which Electrum syncs over
several minutes (and keeps filling in afterwards). Run Electrum with gossip
(non-trampoline) enabled and let it sync, then snapshot its `gossip_db`:

```bash
python3 capture_snapshot.py --gossip-db ~/.electrum/gossip_db \
    --out ./snapshots/mainnet.gossip_db
```

The snapshot is a standalone copy so repeated runs use the identical graph.
Snapshots are not committed (see `.gitignore`); regenerate for a fresher graph.

### 2. Run the benchmark

```bash
python3 run_benchmark.py --snapshot ./snapshots/mainnet.gossip_db \
    --out-dir ./results/before
```

Writes `attempts.csv` (one row per attempt), `summary.csv`, and `summary.txt`
(the printed table) into the output dir. With default sample sizes this performs
thousands of Dijkstra searches over the real graph and can take several minutes.

### 3. Compare before/after a router change

Sampling is seeded, so two runs over the same snapshot are directly comparable:

```bash
python3 run_benchmark.py --snapshot ./snapshots/mainnet.gossip_db --out-dir ./results/before
# ... change electrum/lnrouter.py ...
python3 run_benchmark.py --snapshot ./snapshots/mainnet.gossip_db --out-dir ./results/after
# diff results/before/summary.csv results/after/summary.csv
```

### Options

| flag | default | meaning |
|------|---------|---------|
| `--snapshot` | (required) | captured `gossip_db` path |
| `--out-dir` | (required) | results directory |
| `--network` | `mainnet` | `mainnet` or `testnet` |
| `--amounts` | `100 1000 10000 100000` | payment amounts in sat |
| `--num-sources` | `3` | well-connected nodes used as payer |
| `--dests-per-tier` | `50` | destinations sampled per connectivity tier |
| `--max-fee-millionths` | `50000` | fee budget (50000 = 5%) |
| `--seed` | `0` | RNG seed for reproducible sampling |

## Liquidity mode (`--liquidity`)

A static gossip snapshot has **no channel balances**, so plain `find_route()`
always assumes a channel can forward any amount — which is why the static
benchmark reports unrealistically high success. `--liquidity` instead assigns
hidden, directional per-channel balances and runs the real
`find_route → fail-on-shortfall → update_liquidity_hints → retry` loop, the same
loop the wallet drives on a `TEMPORARY_CHANNEL_FAILURE`.

Each channel's capacity is taken from its `htlc_maximum_msat` (fallback
`--default-capacity-sat`), and node1's share of that capacity is drawn from a
`Beta(b, b)` prior set by `--balance-beta`:

- `b = 1.0` — Uniform: both sides usually hold ~half the capacity (over-generous).
- `b < 1.0` — U-shaped/bimodal: channels are *depleted toward one end*, as real
  channels are. Lower `b` is harsher. `b ≈ 0.5` (arcsine) is a sensible start.

### Giving up (time budget, not a fixed attempt count)

The wallet does **not** give up after a fixed number of retries in the default
gossip path. `LNWallet.pay_to_node` keeps re-running `find_route` and re-sending
until **`PAYMENT_TIMEOUT` (120 s) of wall-clock** elapses (a fixed count is only
used for trampoline / unit tests). The simulation reproduces this: each failed
attempt charges a simulated round-trip latency (`~2 × hop_latency × hops-to-failure`)
against `--payment-timeout-sec`, and the payment **times out** once that budget is
spent. The other outcomes match the wallet exactly — `no_route` and `over_budget`
are terminal (the wallet raises `NoPathFound` / `FeeBudgetExceeded` and does not
retry), and only a liquidity shortfall drives a reroute.

This is why a longer route gives up sooner: the HTLC must travel further before
failing, so each retry costs more time and fewer fit in the 120 s budget — which
is the behaviour the older fixed-count model missed (it let every payment retry
until it "succeeded eventually").

### Calibration knobs

There are now **two** knobs to tune against measured mainnet behaviour:

- `--balance-beta` — how depleted channels are (lower = harsher liquidity).
- `--hop-latency-sec` — simulated per-hop HTLC latency; higher means each failed
  attempt costs more time, so fewer retries fit in the timeout (more `timed_out`).

Note that on a well-connected graph the retry loop reroutes around most dry hops,
so a harsher prior mainly raises `mean_try` (retries-to-success) and `timed_out%`
rather than `success%`; the `no_route` failure is topology/feasibility and is
independent of liquidity.

| flag | default | meaning |
|------|---------|---------|
| `--liquidity` | off | enable the balance model + retry loop |
| `--balance-beta` | `0.5` | Beta(b,b) split shape; 1.0=uniform, <1.0=depleted |
| `--balance-seed` | `0` | RNG seed for the hidden balance assignment |
| `--default-capacity-sat` | `5000000` | capacity used when `htlc_maximum_msat` is unset |
| `--payment-timeout-sec` | `120` | wall-clock give-up budget (mirrors `PAYMENT_TIMEOUT`) |
| `--hop-latency-sec` | `1.0` | simulated per-hop HTLC latency; higher = fewer retries per timeout |

## Tests

`tests/test_pathfinding_benchmark.py` covers the metric logic, the BFS oracle,
and a snapshot persist→reload→benchmark round-trip on a synthetic graph
(no network needed): `pytest tests/test_pathfinding_benchmark.py -v`.

[issue]: https://github.com/spesmilo/electrum/issues/10443
