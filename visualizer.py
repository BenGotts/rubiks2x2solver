import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List

import plots

class CubeVisualizer:
    def __init__(self, results_dir: str = "results", style: str = "light"):
        self.results_dir = Path(results_dir)
        self.plot_dir = Path("plots")
        self.plot_dir.mkdir(exist_ok=True)
        
        # Color palette matching deep-dive defaults
        self.method_colors = {
            "lbl": "#d65f4d", 
            "ortega": "#9288d1", 
            "cll": "#47c984", 
            "eg": "#f5a631", 
            "optimal": "#555555"
        }
        self._apply_theme(style)

    def _apply_theme(self, style: str):
        if style == "dark":
            plt.style.use('dark_background')
        else:
            plt.style.use('seaborn-v0_8-whitegrid')
            plt.rcParams.update({
                'axes.grid': True,
                'grid.alpha': 0.3,
                'grid.linestyle': '--',
                'axes.spines.top': False,
                'axes.spines.right': False
            })

    def load_optimal_data(self, method: str, colors: List[str]) -> np.ndarray:
        """Chooses the color with smallest total moves per state (Full CN)."""
        method_dir = self.results_dir / method.lower()
        color_data_list = []
        for c in colors:
            p = method_dir / f"{c}_data.npy"
            if p.exists(): color_data_list.append(np.load(p))
            
        if not color_data_list:
            raise FileNotFoundError(f"No data for {method}")

        merged_data = color_data_list[0].copy()
        step_names = [n for n in merged_data.dtype.names if n != 'depth']
        n_colors, N = len(color_data_list), len(merged_data)

        totals = np.full((n_colors, N), np.inf)
        for i, data in enumerate(color_data_list):
            valid_mask = data['depth'] >= 0
            # vectorized sum across all steps and AUFs
            total_moves = np.zeros(N)
            for step in step_names:
                total_moves += data[step]
            totals[i, valid_mask] = total_moves[valid_mask]

        best_idx = np.argmin(totals, axis=0)
        best_total = np.min(totals, axis=0)

        for step in step_names:
            stacked = np.vstack([d[step] for d in color_data_list])
            chosen = stacked[best_idx, np.arange(N)]
            chosen[~np.isfinite(best_total)] = -1
            merged_data[step] = chosen

        return merged_data

    def get_total_moves(self, data: np.ndarray) -> np.ndarray:
        """Returns HTM move counts for valid states."""
        valid = data['depth'] >= 0
        step_names = [n for n in data.dtype.names if n not in ['depth']]
        return np.sum([data[s][valid].astype(np.int32) for s in step_names], axis=0)

    def save_plot(self, fig, filename: str):
        out_path = self.plot_dir / filename
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  [+] Saved {filename}")

def main():
    parser = argparse.ArgumentParser(description='2x2 Method Deep-Dive Visualizer')
    parser.add_argument('--config', help='Path to JSON config')
    parser.add_argument('--dist-npy', default='pocket2x2_depths_htm_modrot.npy', help='Optimal distances file')
    parser.add_argument('--plots', nargs='+', help='Select: matrix, gap, totals, auf, split, perstep')
    parser.add_argument('--force', action='store_true', help='Regenerate existing plots')
    args = parser.parse_args()

    viz = CubeVisualizer()
    data_dict = {}

    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
            methods_list = config.get('methods', [])
            print(f"Loading Configuration: {config.get('name', 'Method Comparison')}")
    else:
        print("No config. Scanning results/ for all methods...")
        methods_list = [{"method": d.name, "label": d.name.upper()} 
                       for d in viz.results_dir.iterdir() if d.is_dir()]

    optimal_dist = np.load(args.dist_npy) if Path(args.dist_npy).exists() else None

    for m in methods_list:
        m_key = m['method'].lower()
        
        label = m.get('label', m_key.upper())
        colors = m.get('colors', ['W', 'Y', 'G', 'B', 'R', 'O'])
        
        base_color = viz.method_colors.get(m_key, '#777777')
        viz.method_colors[label] = m.get('plot_color', base_color)
        
        print(f"  -> Loading {label} ({m_key})")
        data_dict[label] = viz.load_optimal_data(m_key, colors)

    plot_map = {
        'matrix': (plots.plot_main_comparison_matrix, "method_comparison_matrix.png"),
        'gap': (plots.plot_efficiency_gap_faceted, "efficiency_gap_faceted.png"),
        'totals': (plots.plot_per_method_totals, "method_totals_faceted.png"),
        'auf': (plots.plot_auf_grouped_bars, "auf_comparison.png"),
        'split': (plots.plot_first_vs_remainder_lines, "first_vs_remainder.png"),
        'perstep': (plots.plot_per_step_volume, "per_step_distribution.png")
    }

    selected = args.plots if args.plots else plot_map.keys()

    for p_key in selected:
        if p_key in plot_map:
            func, fname = plot_map[p_key]
            if (viz.plot_dir / fname).exists() and not args.force:
                print(f"  [-] Skipping {fname}")
                continue
            
            print(f"  [*] Generating {p_key}...")
            if p_key in ['matrix', 'split']:
                func(viz, data_dict, optimal_dist)
            else:
                func(viz, data_dict)

if __name__ == "__main__":
    main()