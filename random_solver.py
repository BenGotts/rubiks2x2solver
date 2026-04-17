"""
Random Solver: Test solving methods on randomly selected states.

Selects N random states and consistently evaluates all methods from the same 
base position (e.g., a solved face) to ensure fair move-count comparisons.
"""

import argparse
import json
import numpy as np
import random
from pathlib import Path
from collections import deque
from typing import Dict, List, Tuple

import pocket_cube
from solver import get_or_create_global_transitions, get_or_create_optimal_distances

class RandomSolver:
    def __init__(self, config_file: str):
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        self.max_states = pocket_cube.N_STATES
        self.moves = pocket_cube.REDUCED_MOVES
        self.transitions = get_or_create_global_transitions(self.max_states, self.moves)
        self.optimal_dist = get_or_create_optimal_distances(self.transitions, self.max_states)
        
        self.results_dir = Path(self.config.get('results_dir', 'results'))
        self.results_dir.mkdir(exist_ok=True)
        
        self.method_data = {}
        self.first_steps = {}
        
        methods = self.config.get('methods', [])
        if not methods:
            raise ValueError("No methods loaded! Please check your config JSON. It must contain a 'methods' array.")
            
        for m in methods:
            label = m['label']
            m_key = m['method'].lower()
            colors = m.get('colors', ['W', 'Y', 'G', 'B', 'R', 'O'])
            
            print(f"Loading precomputed data for {label}...")
            data = self.load_optimal_data(m_key, colors)
            self.method_data[label] = data
            
            step_names = [n for n in data.dtype.names if n not in ['depth', 'pre_auf', 'mid_auf', 'post_auf']]
            self.first_steps[label] = step_names[0] if step_names else 'unknown'

    def load_optimal_data(self, method: str, colors: List[str]) -> np.ndarray:
        """Loads and merges color neutral data, identical to visualizer.py logic."""
        method_dir = self.results_dir / method.lower()
        color_data_list = []
        for c in colors:
            p = method_dir / f"{c}_data.npy"
            if p.exists(): 
                color_data_list.append(np.load(p))
            
        if not color_data_list:
            raise FileNotFoundError(f"Could not find any .npy files for {method} in {method_dir}. Did you run solver.py?")

        merged_data = color_data_list[0].copy()
        step_names = [n for n in merged_data.dtype.names if n != 'depth']
        n_colors, N = len(color_data_list), len(merged_data)

        totals = np.full((n_colors, N), np.inf)
        for i, data in enumerate(color_data_list):
            valid_mask = data['depth'] >= 0
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

    def find_base_state(self, start_state_id: int) -> Tuple[int, int]:
        """
        BFS to find the nearest base state using the pre-computed arrays.
        Returns: (base_state_id, distance_to_base)
        """
        if self.optimal_dist[start_state_id] <= 1:
            return start_state_id, 0

        baseline_label = self.config['methods'][0]['label']
        baseline_data = self.method_data[baseline_label]
        baseline_step = self.first_steps[baseline_label]

        queue = deque([(start_state_id, 0)])
        visited = {start_state_id}

        while queue:
            current_state, dist = queue.popleft()
            
            if baseline_data[baseline_step][current_state] == 0:
                return current_state, dist
                
            for move_idx in range(len(self.moves)):
                next_state = self.transitions[move_idx, current_state]
                if next_state not in visited:
                    visited.add(next_state)
                    queue.append((next_state, dist + 1))
                    
        return start_state_id, 0

    def run_trials(self, num_trials: int = 50, wca_legal: bool = True, seed: int = None) -> Dict[str, np.ndarray]:
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            print(f"Random seed set to: {seed}")

        min_depth = 4 if wca_legal else 0
        valid_indices = np.where(self.optimal_dist >= min_depth)[0]
        
        if len(valid_indices) < num_trials:
            print(f"Warning: Only {len(valid_indices)} valid states found. Reducing num_trials.")
            num_trials = len(valid_indices)

        selected_states = np.random.choice(valid_indices, num_trials, replace=False)
        print(f"Selected {num_trials} random states (WCA Legal: {wca_legal})")
        
        all_results = {}
        for label, data in self.method_data.items():
            all_results[label] = np.zeros(num_trials, dtype=data.dtype)

        for i, state_id in enumerate(selected_states):
            opt_depth = self.optimal_dist[state_id]
            
            base_state, dist_to_base = self.find_base_state(state_id)
            
            for label, data in self.method_data.items():
                first_step = self.first_steps[label]
                
                all_results[label][i] = data[base_state]
                
                all_results[label][i]['depth'] = opt_depth
                
                if all_results[label][i][first_step] >= 0:
                    all_results[label][i][first_step] += dist_to_base
                    
            if (i + 1) % max(1, num_trials // 10) == 0:
                print(f"Progress: {i + 1}/{num_trials} trials complete.")
                
        return all_results

    def save_results(self, data_dict: Dict[str, np.ndarray]):
        for label, data in data_dict.items():
            out_path = self.results_dir / f"random_{label.lower()}_data.npy"
            np.save(out_path, data)
            print(f"Saved formatted trial data for {label} to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare 2x2 solving methods on random scrambles')
    parser.add_argument('config', help='Configuration JSON file')
    parser.add_argument('-n', '--num-trials', type=int, default=50,
                        help='Number of random trials (default: 50)')
    parser.add_argument('--wca-legal', action='store_true',
                        help='Only test WCA-legal scrambles (depth >= 4)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    rs = RandomSolver(args.config)
    results = rs.run_trials(
        num_trials=args.num_trials, 
        wca_legal=args.wca_legal, 
        seed=args.seed
    )
    rs.save_results(results)