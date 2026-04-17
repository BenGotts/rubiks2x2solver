import argparse
import json
import numpy as np
import signal
import time
from typing import Dict, List, Tuple, Optional
from collections import deque
from pathlib import Path
from abc import ABC, abstractmethod

import pocket_cube

# ==========================================
# DEFAULTS
# ==========================================

DEFAULT_DIST_NPY='pocket2x2_depths_htm_modrot.npy'
DEFAULT_TRANSITION_NPY='pocket2x2_transitions.npy'
DEFAULT_OUTPUT_DIR='results'
DEFAULT_METHODS=['all']
DEFAULT_LOG_INTERVAL=250000

# ==========================================
# JIT COMPILED BFS KERNELS
# ==========================================

try:
    from numba import jit
    NUMBA_AVAILABLE = True
    print("✓ Numba JIT compiler available - using optimized code paths")
except ImportError:
    NUMBA_AVAILABLE = False
    print("⚠ Numba not available - using standard Python (slower)")
    def jit(*args, **kwargs):
        def decorator(func): return func
        return decorator

@jit(nopython=True, cache=True)
def propagate_bfs_kernel(
    data_matrix: np.ndarray,
    current_queue: np.ndarray,
    next_queue: np.ndarray,
    state_transitions: np.ndarray,
    max_states: int,
    first_step_col: int
) -> int:
    """Optimized JIT kernel with double-buffering and no memory allocations."""
    num_moves = state_transitions.shape[0]
    num_fields = data_matrix.shape[1]
    num_expanded = 0
    
    for i in range(len(current_queue)):
        v = current_queue[i]
        parent_dist = data_matrix[v, first_step_col]
        
        for move_idx in range(num_moves):
            w = state_transitions[move_idx, v]
            if w >= 0 and w < max_states and data_matrix[w, first_step_col] == -1:
                data_matrix[w, first_step_col] = parent_dist + 1
                for col in range(1, num_fields):
                    if col != first_step_col:
                        data_matrix[w, col] = data_matrix[v, col]
                next_queue[num_expanded] = w
                num_expanded += 1
    return num_expanded

@jit(nopython=True, cache=True)
def generate_optimal_depths(state_transitions: np.ndarray, max_states: int) -> np.ndarray:
    """BFS to find the optimal move count from the solved state (0)."""
    dist = np.full(max_states, -1, dtype=np.int8)
    dist[0] = 0
    
    queue = np.zeros(max_states, dtype=np.int32)
    queue[0] = 0
    head = 0
    tail = 1
    
    num_moves = state_transitions.shape[0]
    
    while head < tail:
        v = queue[head]
        head += 1
        d = dist[v]
        
        for m in range(num_moves):
            w = state_transitions[m, v]
            if 0 <= w < max_states and dist[w] == -1:
                dist[w] = d + 1
                queue[tail] = w
                tail += 1
                
    return dist

# ==========================================
# SOLVER BASE CLASS
# ==========================================
class Solver(pocket_cube.PocketCube, ABC):
    """Abstract base class for 2x2 solving methods."""
    AUF_MOVES = ["", "U", "U'", "U2"]
    Y_ROTS = ["", "y", "y'", "y2"]

    def __init__(self, algorithm_file: str, solving_colors: List[str] = None, state_id: int = 0):
        super().__init__(state_id)
        
        if solving_colors is None:
            solving_colors = [pocket_cube.WHITE, pocket_cube.YELLOW]
        self.solving_colors = [c.upper() for c in solving_colors]
        
        with open(algorithm_file, 'r') as f:
            self.config = json.load(f)
        
        self.method_name = self.config.get('name', 'Unknown')
        self.steps = self.config.get('steps', [])
        
        self.algorithms = {}
        for step in self.steps:
            self.algorithms[step['name']] = step.get('algorithms', {})

    def is_face_solved(self, perm8: np.ndarray = None, ori8: np.ndarray = None) -> bool:
        if perm8 is None:
            perm8, ori8 = self.perm8, self.ori8
        
        stickers = self.get_stickers8(perm8, ori8)
        for i in range(0, 24, 4):
            face = stickers[i:i+4]
            if len(set(face)) == 1 and face[0] in self.solving_colors:
                return True
        return False
    
    def is_layer_solved(self, perm8: np.ndarray = None, ori8: np.ndarray = None) -> bool:
        if perm8 is None:
            perm8, ori8 = self.perm8, self.ori8
            
        if not self.is_face_solved(perm8, ori8):
            return False
        
        p, o = self.normalize_to_d(perm8, ori8)
        if p is None or o is None:
            return False
        
        stickers = self.get_stickers8(p, o)
        if (stickers[19] == stickers[18] and 
            stickers[11] == stickers[10] and 
            stickers[7] == stickers[6] and 
            stickers[23] == stickers[22]):
            return True
        return False
    
    def normalize_to_d(self, perm8: np.ndarray = None, ori8: np.ndarray = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if perm8 is None:
            perm8, ori8 = self.perm8, self.ori8
        
        for rot in [None, "x", "x'", "x2", "z", "z'"]:
            if rot is None:
                curr_p, curr_o = perm8, ori8
            else:
                curr_p, curr_o = self._apply_rotation(perm8, ori8, rot)
            
            s = self.get_stickers8(curr_p, curr_o)
            d = s[12:16]
            if len(set(d)) == 1 and d[0] in self.solving_colors:
                return curr_p, curr_o
        return None, None
    
    def _apply_rotation(self, perm8: np.ndarray, ori8: np.ndarray, rotation: str) -> Tuple[np.ndarray, np.ndarray]:
        p, t = self.ROTATIONS[rotation]
        return perm8[p], (ori8[p] + t) % 3
    
    def _apply_move(self, perm8: np.ndarray, ori8: np.ndarray, move: str) -> Tuple[np.ndarray, np.ndarray]:
        p, t = self.MOVES[move]
        return perm8[p], (ori8[p] + t) % 3
    
    def _apply_algorithm(self, perm8: np.ndarray, ori8: np.ndarray, alg: str) -> Tuple[np.ndarray, np.ndarray, int]:
        temp_perm, temp_ori = perm8.copy(), ori8.copy()
        count = 0
        if not alg: return temp_perm, temp_ori, 0
            
        for move in alg.split():
            if move in self.ROTATIONS:
                temp_perm, temp_ori = self._apply_rotation(temp_perm, temp_ori, move)
            elif move in self.MOVES:
                temp_perm, temp_ori = self._apply_move(temp_perm, temp_ori, move)
                count += 1
            else:
                raise ValueError(f"Invalid move '{move}'")
        return temp_perm, temp_ori, count

    @abstractmethod
    def solve_from_state(self, perm8: np.ndarray, ori8: np.ndarray) -> Dict:
        pass

    @abstractmethod
    def is_seed_state(self, perm8: np.ndarray, ori8: np.ndarray) -> bool:
        pass

    def run_analysis(self, dist: np.ndarray, state_transitions: np.ndarray, max_states: int, log_interval: int = 100000) -> Tuple[Dict[str, np.ndarray], List[Dict]]:
        structural_steps = [step["name"].lower() for step in self.steps]
        step_names = structural_steps + ['pre_auf', 'mid_auf', 'post_auf']
        dt = np.dtype([('depth', 'i1')] + [(name, 'i1') for name in step_names])
        
        color_data = {c: np.full(max_states, -1, dtype=dt) for c in pocket_cube.COLOR_NEUTRAL}
        for c in color_data: color_data[c]['depth'][:] = dist[:]
        
        color_queues = {c: deque() for c in pocket_cube.COLOR_NEUTRAL}
        seed_counts = {c: 0 for c in pocket_cube.COLOR_NEUTRAL}
        failed_states = []
        interrupted = [False]

        def signal_handler(sig, frame): 
            print("\n\n⚠ Interrupted by user (Ctrl+C)")
            interrupted[0] = True
            
        original_handler = signal.signal(signal.SIGINT, signal_handler)

        try:
            # Phase 1: Find seed states
            print(f"\nPhase 1: Finding seed states for {self.method_name}...")
            p1_start = time.time()
            
            for state_id in range(max_states):
                if interrupted[0]: break
                if state_id % log_interval == 0 and state_id > 0:
                    pct = (state_id / max_states) * 100
                    seed_str = ", ".join([f"{c}: {seed_counts[c]:,}" for c in pocket_cube.COLOR_NEUTRAL if seed_counts[c] > 0])
                    fails = len(failed_states)
                    fail_str = f" | ❌ Fails: {fails:,}" if fails > 0 else ""
                    print(f"  [{pct:>5.1f}%] Scanned {state_id:>9,} | Seeds -> {seed_str if seed_str else 'None yet'}{fail_str}")
                
                if dist[state_id] < 0: continue
                
                p7, q7 = self.unpackcube(state_id)
                perm8, ori8 = self.lift_to_full8(p7, q7)

                for color in pocket_cube.COLOR_NEUTRAL:
                    
                    original_colors = self.solving_colors
                    self.solving_colors = [color]
                    
                    if self.is_seed_state(perm8, ori8):
                        result = self.solve_from_state(perm8, ori8)
                        
                        if result['success']:
                            for step_name, move_count in result['moves'].items():
                                if step_name in color_data[color].dtype.names:
                                    color_data[color][state_id][step_name] = move_count
                            color_queues[color].append(state_id)
                            seed_counts[color] += 1
                        else:
                            failed_states.append({"id": int(state_id), "color": color, "error": result.get('error', 'Unknown Error')})
                            
                    self.solving_colors = original_colors

            if interrupted[0]: return color_data, failed_states

            print(f"\n  ✓ Phase 1 Complete in {time.time() - p1_start:.1f}s")
            print(f"  {'='*30}\n  FINAL SEED COUNTS:")
            for c in sorted(pocket_cube.COLOR_NEUTRAL):
                print(f"    Color {c}: {seed_counts[c]:>9,} states")
            print(f"  {'-'*30}")
            print(f"    Failures: {len(failed_states):>9,} states")
            print(f"  {'='*30}")

            # Phase 2: BFS with Double-Buffering
            print(f"\nPhase 2: Propagating distances using global transition table...")
            first_step = structural_steps[0]
            
            for color in pocket_cube.COLOR_NEUTRAL:
                if interrupted[0]: break
                if seed_counts[color] == 0: continue
                
                print(f"\n  Propagating {color}...")
                p2_start = time.time()
                data_matrix = color_data[color].view(np.int8).reshape(max_states, len(color_data[color].dtype.names))
                buf_a = np.zeros(max_states, dtype=np.int32)
                buf_b = np.zeros(max_states, dtype=np.int32)
                
                q_len = len(color_queues[color])
                if q_len > 0: buf_a[:q_len] = np.array(list(color_queues[color]), dtype=np.int32)
                
                curr_q, next_q, depth = buf_a, buf_b, 0
                total_propagated = 0
                
                while q_len > 0 and not interrupted[0]:
                    total_propagated += q_len
                    print(f"    [Depth {depth:>2}] Propagating {q_len:>9,} states...")
                    q_len = propagate_bfs_kernel(data_matrix, curr_q[:q_len], next_q, state_transitions, max_states, color_data[color].dtype.names.index(first_step))
                    curr_q, next_q = next_q, curr_q
                    depth += 1
                    
                print(f"  ✓ {color} Complete: {total_propagated:,} states mapped in {time.time() - p2_start:.1f}s")
        finally:
            signal.signal(signal.SIGINT, signal.SIG_DFL)
        return color_data, failed_states

# ==========================================
# GLOBAL SOLVER & EXECUTION
# ==========================================

def get_or_create_global_transitions(max_states: int, moves: List[str], log_interval: int = DEFAULT_LOG_INTERVAL, transition_file: str = DEFAULT_TRANSITION_NPY) -> np.ndarray:
    path = Path(transition_file)
    if path.exists():
        print(f"\n[SOLVER] Loading existing global state transitions from {transition_file}...")
        start_time = time.time()
        transitions = np.load(path)
        print(f"✓ Loaded in {time.time() - start_time:.2f}s")
        return transitions

    print(f"\n[SOLVER] Building global state transition table ({max_states:,} states)...")
    start_time = time.time()
    num_moves = len(moves)
    transitions = np.full((num_moves, max_states), -1, dtype=np.int32)
    cube = pocket_cube.PocketCube()
    
    for state_id in range(max_states):
        if state_id > 0 and state_id % log_interval == 0:
            pct = (state_id / max_states) * 100
            print(f"  [{pct:>5.1f}%] Processed {state_id:>9,} / {max_states:,} ...")
            
        p7, q7 = cube.unpackcube(state_id)
        perm8, ori8 = cube.lift_to_full8(p7, q7)
        for move_idx, move_name in enumerate(moves):
            np8, no8 = cube.MOVES[move_name]
            p_res = perm8[np8]
            o_res = (ori8[np8] + no8) % 3
            pp7, qq7 = cube.project_to_7(p_res, o_res)
            w = cube.packcube((pp7, qq7))
            if w < max_states:
                transitions[move_idx, state_id] = w
                
    print(f"✓ Global transition table built in {time.time() - start_time:.1f}s")
    
    print(f"\n[SOLVER] Saving global state transitions to {transition_file}...")
    np.save(path, transitions)
    return transitions

def get_or_create_optimal_distances(transitions: np.ndarray, max_states: int, dist_file: str = DEFAULT_DIST_NPY) -> np.ndarray:
    path = Path(dist_file)
    if path.exists():
        print(f"\n[SOLVER] Loading existing optimal distances from {dist_file}")
        return np.load(path)
        
    print(f"\n[SOLVER] Calculating optimal distances from solved state...")
    start_time = time.time()
    dist = generate_optimal_depths(transitions, max_states)
    np.save(path, dist)
    print(f"✓ Saved to {dist_file} in {time.time() - start_time:.1f}s")
    return dist

def load_solver(method: str) -> Solver:
    algorithm_dir = Path(__file__).parent / "algorithms"
    algorithm_file = algorithm_dir / f"{method}.json"
    
    if not algorithm_file.exists():
        raise FileNotFoundError(f"Algorithm file not found: {algorithm_file}")
    
    if method == 'ortega':
        from method_solvers.solver_ortega import OrtegaSolver
        return OrtegaSolver(str(algorithm_file))
    elif method == 'cll':
        from method_solvers.solver_cll import CLLSolver
        return CLLSolver(str(algorithm_file))
    elif method == 'lbl':
        from method_solvers.solver_lbl import LBLSolver
        return LBLSolver(str(algorithm_file))
    elif method == 'eg':
        from method_solvers.solver_eg import EGSolver
        return EGSolver(str(algorithm_file))
    else:
        raise ValueError(f"Unknown method: {method}")

def main():
    parser = argparse.ArgumentParser(description='2x2x2 Rubik\'s Cube Solver')
    parser.add_argument('--dist-npy', default=DEFAULT_DIST_NPY, help='Optimal distances file')
    parser.add_argument('--transition-npy', default=DEFAULT_TRANSITION_NPY, help='State transitions cache file')
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR, help='Directory to save analysis results')
    parser.add_argument('--methods', nargs='+', choices=['ortega', 'cll', 'lbl', 'eg', 'all'], default=DEFAULT_METHODS)
    parser.add_argument('--log-interval', type=int, default=DEFAULT_LOG_INTERVAL, help='Log interval for long operations')
    parser.add_argument('--force', action='store_true', help='Force recalculation even if method .npy files exist')
    args = parser.parse_args()
    
    total_start_time = time.time()
    max_states = pocket_cube.N_STATES
    moves = pocket_cube.REDUCED_MOVES 
    
    transitions = get_or_create_global_transitions(max_states, moves, args.log_interval, args.transition_npy)
    dist = get_or_create_optimal_distances(transitions, max_states, args.dist_npy)
    
    methods = ['ortega', 'cll', 'lbl', 'eg'] if 'all' in args.methods else args.methods
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    for method in methods:
        method_dir = output_dir / method
        
        if method_dir.exists() and not args.force:
            all_cached = all((method_dir / f"{c}_data.npy").exists() for c in pocket_cube.COLOR_NEUTRAL)
            if all_cached:
                print(f"\n{'='*80}\n⏭️ SKIPPING: {method.upper()} (Already cached in {method_dir}/)\n{'='*80}")
                continue
                
        print(f"\n{'='*80}\nANALYZING: {method.upper()}\n{'='*80}")
        solver = load_solver(method)
        color_data, failed = solver.run_analysis(dist, transitions, max_states, args.log_interval)
        
        method_dir.mkdir(parents=True, exist_ok=True)
        for color, data in color_data.items():
            if np.any(data['depth'] >= 0):
                np.save(method_dir / f"{color}_data.npy", data)
                print(f"  [+] Saved {color}_data.npy")
                
    print(f"\n✅ All analysis complete in {(time.time() - total_start_time)/60:.1f} minutes!")

if __name__ == "__main__":
    main()