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

Sample random WCA-legal scrambles and compare method performance on them:

```
python random_solver.py cn_config.json -n 1000 --wca-legal
```

## Configuration file (`--config` / `cn_config.json`)

`solver.py` always analyzes a method against all 6 face colors (color-neutral), saving one
`.npy` file per color under `results/<method>/`. The config file passed to `visualizer.py
--config` and `random_solver.py` controls how those per-color results are read back and
compared — in particular, which colors are merged together for a given method entry.

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
| `results_dir`            | no       | `"results"`                       | Where per-color `.npy` analysis data lives. Only read by `random_solver.py`; `visualizer.py` always uses `results/`. |
| `methods`                | yes      | —                                  | List of method entries to compare (see below). |
| `methods[].method`       | yes      | —                                  | One of `lbl`, `ortega`, `cll`, `eg` — must match a subdirectory under `results_dir` produced by `solver.py`. |
| `methods[].colors`       | no       | `["W", "Y", "G", "B", "R", "O"]`   | Which of the 6 precomputed face colors to consider for this entry. For each state, the color giving the lowest move count among these is picked. Restrict to a single color (e.g. `["W"]`) to compare methods "pinned" to one color instead of color-neutral. |
| `methods[].label`        | no       | `method` uppercased               | Display name used in plot legends and CSV output. Must be unique across entries. |
| `methods[].plot_color`   | no       | built-in per-method color         | Hex color (e.g. `"#9288d1"`) used for this entry's line/bar in plots. |

Note for `random_solver.py`: the **first** entry in `methods` is used as the baseline for
aligning all methods to a common starting state before sampling (see `find_base_state` in
`random_solver.py`), so method order in the config matters there — it does not matter for
`visualizer.py`.
