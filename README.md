# rubiks2x2solver

Analyzes 2x2x2 Rubik's Cube (Pocket Cube) solving methods — LBL, Ortega, CLL, and EG — by
brute-forcing move counts from every reachable cube state and comparing efficiency across methods.

## Usage

Run the analysis (builds/caches the global state transition table and optimal-distance table on
first run, then computes per-method move counts):

```
python solver.py --methods all
```

Generate comparison plots and CSV exports from the analysis results:

```
python visualizer.py --export
```

Generate the same plots crediting move cancellations instead — see `methods[].cancel_moves`
below:

```
python visualizer.py --config configs/cn_config_cancellations.json --export
```

Or run every config under `configs/` in one invocation — each writes to its own
`plots/<config-name>/` and `results/exports/<config-name>/` subdirectory instead of
overwriting a shared, flat one:

```
python visualizer.py --config configs --export
```

Compare two configs covering the same methods state-by-state (e.g. how much do cancellations
save, per method) instead of eyeballing their independently-scaled plots side by side:

```
python visualizer.py --compare configs/cn_config.json configs/cn_config_cancellations.json
```

Sample random WCA-legal scrambles and compare method performance on them:

```
python random_solver.py configs/cn_config.json -n 1000 --wca-legal
```

## Command-line arguments

### `solver.py`

Runs the brute-force analysis. Builds (and caches to disk) the global state transition table
and optimal-distance table on first run, then computes per-method move counts for all 6 face
colors.

| Argument | Default | Meaning |
|---|---|---|
| `--methods {ortega,cll,lbl,eg,all}` (one or more) | `all` | Which method(s) to analyze. |
| `--dist-npy PATH` | `pocket2x2_depths_htm_modrot.npy` | Optimal-distance cache file — loaded if it exists, else computed and saved here. |
| `--transition-npy PATH` | `pocket2x2_transitions.npy` | Global state-transition table cache file — loaded if it exists, else built and saved here. |
| `--output-dir DIR` | `results` | Where per-method analysis results are saved: one `<output-dir>/<method>.npz` file per method, holding one array per color. |
| `--seeds-dir DIR` | `seeds` | Where seed-state lookups are cached: one `<seeds-dir>/<criterion>.npz` file per seed criterion (`face.npz`, `layer.npz`), holding one array per color. Ortega and EG share the `face` criterion and CLL/LBL share `layer`; `layer` is always a subset of `face` (a solved layer is a solved face with its sides also aligned), so `layer.npz` is derived from `face.npz` rather than a second full scan. Whichever method runs first populates each file and every other method reuses it — this persists across separate `solver.py` runs, not just within one. |
| `--log-interval N` | `250000` | How often (in states scanned) to print Phase 1 progress. |
| `--force` | off | Recompute a method even if its `<output-dir>/<method>.npz` output already exists with all 6 colors present. |
| `--force-tables` | off | Rebuild `pocket2x2_transitions.npy`, `pocket2x2_depths_htm_modrot.npy`, and the `--seeds-dir` seed-state cache even if they already exist, instead of loading the cached files. Use this after a code change to the move tables, the transition/distance build logic, or the seed criteria (`is_face_solved`/`is_layer_solved`) — a stale cache would otherwise be loaded silently. Combine with `--force` to also recompute method results against the rebuilt tables. |

### `visualizer.py`

Generates comparison plots (and optionally CSV exports) from `solver.py`'s output.

| Argument | Default | Meaning |
|---|---|---|
| `--config PATH` | none — scans `results/` for every method subdirectory | Path to a config JSON (see below) selecting which methods/colors/labels to compare, **or** a directory of them (e.g. `configs/`) to run every config in one invocation — each one's plots/exports then nest under `plots/<config-name>/` and `results/exports/<config-name>/` instead of a shared flat directory, so configs never overwrite each other's output. Mutually exclusive with `--compare`. |
| `--compare CONFIG_A CONFIG_B` | — | Compares two configs covering the same methods (matched by each entry's `method` field, not its `label`) state-by-state instead of generating the usual `--plots` — see "Comparing two configs" below. Mutually exclusive with `--config`. |
| `--plots {matrix,gap,totals,auf,split,perstep,random}` (one or more) | all except `random` | Which plots to generate. `random` requires data from `random_solver.py` (`results/random_<label>_data.npy`); the rest require `solver.py`'s output. Ignored by `--compare`, which always produces its own comparison plot. |
| `--dist-npy PATH` | `pocket2x2_depths_htm_modrot.npy` | Optimal-distance file, used by the `matrix` and `split` plots to overlay the true-optimal distribution. Silently skipped if the file doesn't exist. |
| `--force` | off | Regenerate a plot even if its PNG already exists under `plots/` (or `plots/<config-name>/` / `plots/compare_<a>_vs_<b>/`). |
| `--export` | off | Also write CSVs to `results/exports/` (or its `<config-name>` / `compare_<a>_vs_<b>` subdirectory): the raw per-state data behind each plot, plus every plot's computed data points. |

#### Comparing two configs

`--compare CONFIG_A CONFIG_B` is for configs that analyze the *same* methods against the same
`results/<method>.npz` data but differ in some setting per entry — e.g. `configs/cn_config.json`
vs. `configs/cn_config_cancellations.json`, which are identical except for `cancel_moves`. Rather
than combining both configs' entries into one config and eyeballing 8 independently-scaled
curves, it pairs up each common `method` and computes an exact **per-state** move-count delta
(`total_A - total_B`),
since both sides read the identical, index-aligned state space. Output goes to
`plots/compare_<config_a_stem>_vs_<config_b_stem>/config_comparison.png`: one subplot per method
showing the distribution of that delta plus the % of states that improved, and a grouped bar
chart of mean total moves under each config. A summary (mean moves saved, % improved, per
method) is also printed to the console and, with `--export`, written to CSV. Methods present in
only one config are skipped with a warning rather than failing the whole comparison.

### `random_solver.py`

Samples random states and evaluates all configured methods from a common starting position,
for a head-to-head comparison on the same scrambles.

| Argument | Default | Meaning |
|---|---|---|
| `config` (positional, required) | — | Path to a config JSON (see below) listing the methods to compare. |
| `-n, --num-trials N` | `50` | Number of random states to sample. |
| `--wca-legal` | off | Restrict sampling to states with optimal distance ≥ 4 (WCA-legal scrambles). Without it, any reachable state (including near-solved ones) can be sampled. |
| `--seed N` | none (nondeterministic) | Random seed, for reproducible trials. |
| `--force-tables` | off | Rebuild the transition/distance tables (see `solver.py` above) even if their cache files exist. |

Output is written to `<results_dir>/random_<label>_data.npy` per method (see `results_dir` in
the config below), ready for `visualizer.py --plots random`.

## Configuration file (`--config` / `configs/`)

`solver.py` always analyzes a method against all 6 face colors (color-neutral), saving one
`results/<method>.npz` file per method (one array per color). The config file passed to
`visualizer.py --config` and `random_solver.py` controls how those per-color results are read
back and compared — in particular, which colors are merged together for a given method entry.
Configs live under `configs/` (`configs/cn_config.json`, `configs/cn_config_cancellations.json`)
— point `visualizer.py --config` at the whole directory to run all of them at once (see above).

```json
{
  "name": "Color Neutral 2x2 Method Comparison",
  "results_dir": "results",
  "methods": [
    { "method": "lbl",    "colors": ["W", "Y", "G", "B", "O", "R"], "label": "LBL (CN)" },
    { "method": "ortega", "colors": ["W"],                          "label": "Ortega (White only)", "plot_color": "#9288d1" }
  ]
}
```

| Field                    | Required | Default                          | Meaning |
|--------------------------|----------|-----------------------------------|---------|
| `name`                   | no       | `"Method Comparison"`             | Display name, printed when the config loads. |
| `results_dir`            | no       | `"results"`                       | Where per-method `.npz` analysis data lives. Only read by `random_solver.py`; `visualizer.py` always uses `results/`. |
| `methods`                | yes      | —                                  | List of method entries to compare (see below). |
| `methods[].method`       | yes      | —                                  | One of `lbl`, `ortega`, `cll`, `eg` — must match a `<method>.npz` file under `results_dir` produced by `solver.py`. |
| `methods[].colors`       | no       | `["W", "Y", "G", "B", "R", "O"]`   | Which of the 6 precomputed face colors to consider for this entry. For each state, the color giving the lowest move count among these is picked. Restrict to a single color (e.g. `["W"]`) to compare methods "pinned" to one color instead of color-neutral. |
| `methods[].label`        | no       | `method` uppercased               | Display name used in plot legends and CSV output. Must be unique across entries. |
| `methods[].plot_color`   | no       | built-in per-method color         | Hex color (e.g. `"#9288d1"`) used for this entry's line/bar in plots. |
| `methods[].cancel_moves` | no       | `false`                           | Credit `reduced_savings` — how many fewer physical moves the solve actually takes than the naive sum of every AUF/algorithm cost, once move cancellations are found. This is orthogonal to which *method* a config entry uses (LBL vs. Ortega vs. CLL vs. EG is a difference in technique/skill level); cancellations are something anyone using any of these methods can find given enough time to look for them, so this flag models whether that time was spent, not a different skill tier. Two kinds are credited (see `Solver._solve_cost` / `_best_with_layer_rotation` in `solver.py`): (1) **free quarter-turn reduction** — adjacent same-face turns across an AUF/algorithm boundary compose mod 4 and cascade (e.g. a `pre_auf` of `U2` immediately followed by an algorithm that itself opens with `U2` erase each other, not just the AUF), applied at every seam with literal moves on both sides: `pre_auf`↔first algorithm, that algorithm↔`mid_auf`, `mid_auf`↔second algorithm (plus any rotation Ortega/EG insert in between), and that algorithm↔`post_auf`; and (2) **free D-layer rotation** — `is_face_solved`/`is_layer_solved` don't care which of the 4 D-layer rotations the untracked, intuitive face/layer-building step lands in, so noticing this lets a solver aim for whichever one lets the following algorithm(s) recognize a cheaper case or cancel better, at no extra cost, instead of just going with whichever one they happened to finish in. `configs/cn_config_cancellations.json` sets this on all 4 methods as a ready-made counterpart to `configs/cn_config.json` — keep them as separate config files (they share method colors, so the same method is recognizable across both sets of plots) rather than combining both sets of entries into one config, which would double every plot's series and get hard to read; run `visualizer.py --config configs` to generate both sets of plots in one invocation without combining them. |

Note for `random_solver.py`: the **first** entry in `methods` is used as the baseline for
aligning all methods to a common starting state before sampling (see `find_base_state` in
`random_solver.py`), so method order in the config matters there — it does not matter for
`visualizer.py`.
