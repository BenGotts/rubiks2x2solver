import argparse
import json
import logging
import sys
import numpy as np
import signal
import time
from typing import Dict, List, Tuple, Optional
from collections import deque
from pathlib import Path
from abc import ABC, abstractmethod

import pocket_cube

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

# ==========================================
# DEFAULTS
# ==========================================

DEFAULT_DIST_NPY='pocket2x2_depths_htm_modrot.npy'
DEFAULT_TRANSITION_NPY='pocket2x2_transitions.npy'
DEFAULT_OUTPUT_DIR='results'
DEFAULT_SEEDS_DIR='seeds'
DEFAULT_METHODS=['all']
DEFAULT_LOG_INTERVAL=250000

# ==========================================
# JIT COMPILED BFS KERNELS
# ==========================================

try:
    from numba import jit
    NUMBA_AVAILABLE = True
    logger.info("✓ Numba JIT compiler available - using optimized code paths")
except ImportError:
    NUMBA_AVAILABLE = False
    logger.warning("⚠ Numba not available - using standard Python (slower)")
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

# 0! through 6!, used by the rank/unrank kernels below.
_FACT7 = np.array([1, 1, 2, 6, 24, 120, 720], dtype=np.int64)

@jit(nopython=True, cache=True)
def _rankperm7(p: np.ndarray) -> int:
    """Numeric port of pocket_cube.PocketCube.rankperm, specialized to n=7. Mutates p."""
    q = np.argsort(p)
    r = 0
    for k in range(6, 0, -1):
        s = p[k]
        pk, pqk = p[k], p[q[k]]
        p[k], p[q[k]] = pqk, pk
        qk, qs = q[k], q[s]
        q[k], q[s] = qs, qk
        r += s * _FACT7[k]
    return r

@jit(nopython=True, cache=True)
def _unrankperm7(r: int, out: np.ndarray) -> None:
    """Numeric port of pocket_cube.PocketCube.unrankperm, specialized to n=7. Fills out in place."""
    for i in range(7):
        out[i] = i
    for k in range(6, 0, -1):
        f = _FACT7[k]
        s = r // f
        r = r - s * f
        out[k], out[s] = out[s], out[k]

@jit(nopython=True, cache=True)
def _build_transitions_kernel(moves_perm: np.ndarray, moves_twist: np.ndarray, max_states: int) -> np.ndarray:
    """
    JIT-compiled equivalent of looping PocketCube.unpackcube -> apply move -> project_to_7 ->
    packcube for every (state, move) pair. moves_perm/moves_twist are stacked (num_moves, 8)
    versions of the pocket_cube.MOVES permutation/twist arrays.
    """
    num_moves = moves_perm.shape[0]
    transitions = np.full((num_moves, max_states), -1, dtype=np.int32)

    p7 = np.empty(7, dtype=np.int64)
    q7 = np.empty(7, dtype=np.int64)
    perm8 = np.empty(8, dtype=np.int64)
    ori8 = np.empty(8, dtype=np.int64)
    p_res = np.empty(8, dtype=np.int64)
    o_res = np.empty(8, dtype=np.int64)
    p7b = np.empty(7, dtype=np.int64)
    q7b = np.empty(7, dtype=np.int64)

    for state_id in range(max_states):
        # unpackcube
        p_rank = state_id // 729
        q_int = state_id % 729
        _unrankperm7(p_rank, p7)
        temp = q_int
        q_sum = 0
        for idx in range(5, -1, -1):
            digit = temp % 3
            q7[idx] = digit
            q_sum += digit
            temp //= 3
        q7[6] = (-q_sum) % 3

        # lift_to_full8
        perm8[0:7] = p7
        perm8[7] = 7
        for i in range(7):
            ori8[i] = q7[i] % 3
        ori8[7] = (-(ori8[0] + ori8[1] + ori8[2] + ori8[3] + ori8[4] + ori8[5] + ori8[6])) % 3

        for m in range(num_moves):
            for i in range(8):
                src = moves_perm[m, i]
                p_res[i] = perm8[src]
                o_res[i] = (ori8[src] + moves_twist[m, i]) % 3

            # project_to_7
            pos7 = 0
            for i in range(8):
                if p_res[i] == 7:
                    pos7 = i
                    break
            j = 0
            for i in range(8):
                if i != pos7:
                    p7b[j] = p_res[i]
                    q7b[j] = o_res[i] % 3
                    j += 1

            # packcube
            rank = _rankperm7(p7b)
            total = 0
            for i in range(6):
                total = total * 3 + q7b[i]
            w = rank * 729 + total

            if w < max_states:
                transitions[m, state_id] = w

    return transitions

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

        # Precompiled (permutation, twist, move_count) for every algorithm/AUF, so hot paths
        # (run_analysis is called per-state, per-color) apply them via array indexing instead
        # of re-parsing and replaying the move string each time.
        self.compiled_algorithms = {
            step_name: {name: self._compile_algorithm(alg) for name, alg in algs.items()}
            for step_name, algs in self.algorithms.items()
        }
        self.compiled_auf = [self._compile_algorithm(a) for a in self.AUF_MOVES]

    def is_face_solved(self, perm8: np.ndarray = None, ori8: np.ndarray = None) -> bool:
        if perm8 is None:
            perm8, ori8 = self.perm8, self.ori8

        # Hot path: avoid slicing out `face` and building a set() per face (6x per call);
        # chained == also short-circuits before the `in` check on a mismatch.
        s = self.get_stickers8(perm8, ori8)
        for i in range(0, 24, 4):
            c = s[i]
            if s[i+1] == c and s[i+2] == c and s[i+3] == c and c in self.solving_colors:
                return True
        return False
    
    def is_layer_solved(self, perm8: np.ndarray = None, ori8: np.ndarray = None) -> bool:
        if perm8 is None:
            perm8, ori8 = self.perm8, self.ori8

        if not self.is_face_solved(perm8, ori8):
            return False

        return self.is_layer_solved_given_face(perm8, ori8)

    def is_layer_solved_given_face(self, perm8: np.ndarray = None, ori8: np.ndarray = None) -> bool:
        """
        The rest of is_layer_solved's check, for callers that already know is_face_solved is
        true (e.g. get_or_create_seed_states, which derives 'layer' seeds from the 'face' seed
        list instead of re-testing is_face_solved for every state - every layer-solved state is
        a face-solved state with its side stickers also aligned, so this is strictly additive).
        """
        if perm8 is None:
            perm8, ori8 = self.perm8, self.ori8

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
            c = s[12]
            if s[13] == c and s[14] == c and s[15] == c and c in self.solving_colors:
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

    def _compile_algorithm(self, alg: str) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Precompute an algorithm's net (permutation, twist, move_count) so applying it to any
        state is a single array-index op instead of re-parsing and replaying its moves.

        Move/rotation composition is associative, so applying `alg` starting from the identity
        state yields exactly that net transform: the resulting perm8 equals the composed
        permutation array, and the resulting ori8 equals the composed twist array.
        """
        identity_perm8 = np.arange(8)
        identity_ori8 = np.zeros(8, dtype=int)
        return self._apply_algorithm(identity_perm8, identity_ori8, alg)

    @staticmethod
    def _apply_compiled(perm8: np.ndarray, ori8: np.ndarray, compiled: Tuple[np.ndarray, np.ndarray, int]) -> Tuple[np.ndarray, np.ndarray, int]:
        perm_effect, twist_effect, count = compiled
        return perm8[perm_effect], (ori8[perm_effect] + twist_effect) % 3, count

    @staticmethod
    def _compose_compiled(first: Tuple[np.ndarray, np.ndarray, int], second: Tuple[np.ndarray, np.ndarray, int]) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Combine two compiled (permutation, twist, move_count) transforms into the single
        transform equivalent to applying `first` then `second` -- so a chain of N compiled
        steps can be pre-reduced to one array-index op instead of N.
        """
        p1, t1, c1 = first
        p2, t2, c2 = second
        return p1[p2], (t1[p2] + t2) % 3, c1 + c2

    @staticmethod
    def _invert_compiled(compiled: Tuple[np.ndarray, np.ndarray, int]) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Compute the compiled transform that undoes `compiled`: applying compiled then its
        inverse (in either order) returns the original state.

        Derivation: `compiled` maps (perm, ori) -> (perm[P], (ori[P]+T) % 3). Solving that for
        (perm, ori) given the result gives perm = result_perm[argsort(P)] and
        ori = (result_ori[argsort(P)] - T[argsort(P)]) % 3, i.e. the inverse transform is
        (argsort(P), -T[argsort(P)] % 3).
        """
        p, t, c = compiled
        p_inv = np.argsort(p)
        t_inv = (-t[p_inv]) % 3
        return p_inv, t_inv, c

    def _solved_orbit(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        All (perm8, ori8) pairs that satisfy is_solved_state.

        is_solved_state only checks that each face's 4 stickers match each other, not that
        they hold any particular color -- so every whole-cube reorientation of the solved
        cube (the 24-element rotation group generated by x/y/z) also counts as solved, not
        just the single canonical identity. Anything that treats "reaching a solved state" as
        "reaching the identity permutation" (e.g. a precomputed case table built by inverting
        a transform back from a single target) must target this whole orbit, not just identity.
        """
        identity = (np.arange(8), np.zeros(8, dtype=int))
        orbit = {(tuple(identity[0].tolist()), tuple(identity[1].tolist())): identity}
        frontier = [identity]
        while frontier:
            next_frontier = []
            for p, o in frontier:
                for rot_name in ['x', "x'", 'x2', 'y', "y'", 'y2', 'z', "z'", 'z2']:
                    p2, o2 = self._apply_rotation(p, o, rot_name)
                    key = (tuple(p2.tolist()), tuple(o2.tolist()))
                    if key not in orbit:
                        orbit[key] = (p2, o2)
                        next_frontier.append((p2, o2))
            frontier = next_frontier
        return list(orbit.values())

    @abstractmethod
    def solve_from_state(self, perm8: np.ndarray, ori8: np.ndarray) -> Dict:
        pass

    @abstractmethod
    def is_seed_state(self, perm8: np.ndarray, ori8: np.ndarray) -> bool:
        pass

    @staticmethod
    def _load_npz(path: Path) -> Dict[str, np.ndarray]:
        if not path.exists():
            return {}
        with np.load(path) as npz:
            return {k: npz[k] for k in npz.files}

    def _ensure_face_seeds(self, colors: List[str], dist: np.ndarray, max_states: int, seeds_dir: Path, log_interval: int, force: bool, interrupted: List[bool], rebuilt_this_run: set) -> Optional[Dict[str, np.ndarray]]:
        """
        Loads (or scans and caches to seeds_dir/face.npz, one array per color) the 'face' seed
        states for `colors` - the full state-space scan that both 'face'-criterion methods and,
        via get_or_create_seed_states, 'layer'-criterion methods rely on. Returns None if
        interrupted before an in-progress scan completed.

        rebuilt_this_run: set of criterion names already (re)built during this process
        invocation (see main()). 'face' is a prerequisite of 'layer', so a 'layer' request can
        trigger this same method again; force is only honored the first time 'face' is touched
        in a run - otherwise --force-tables would force a second, redundant full-state rescan
        of 'face' moments after the first one just wrote it.
        """
        seeds_dir.mkdir(parents=True, exist_ok=True)
        path = seeds_dir / "face.npz"
        cached = self._load_npz(path)

        effective_force = force and 'face' not in rebuilt_this_run
        to_scan = list(colors) if effective_force else [c for c in colors if c not in cached]

        if not to_scan:
            logger.info("  Reusing cached 'face' seed states for all colors")
            return {c: cached[c] for c in colors}

        logger.info(f"  Scanning for 'face' seed states ({', '.join(to_scan)})...")
        scan_start = time.time()
        found = {c: [] for c in to_scan}

        for state_id in range(max_states):
            if interrupted[0]: break
            if state_id % log_interval == 0 and state_id > 0:
                pct = (state_id / max_states) * 100
                found_str = ", ".join(f"{c}: {len(found[c]):,}" for c in to_scan if found[c])
                logger.info(f"    [{pct:>5.1f}%] Scanned {state_id:>9,} | Found -> {found_str if found_str else 'None yet'}")

            if dist[state_id] < 0: continue

            p7, q7 = self.unpackcube(state_id)
            perm8, ori8 = self.lift_to_full8(p7, q7)

            for color in to_scan:
                original_colors = self.solving_colors
                self.solving_colors = [color]
                if self.is_face_solved(perm8, ori8):
                    found[color].append(state_id)
                self.solving_colors = original_colors

        if interrupted[0]:
            return None

        for color in to_scan:
            cached[color] = np.array(found[color], dtype=np.int32)
        np.savez(path, **cached)
        rebuilt_this_run.add('face')

        logger.info(f"  ✓ 'face' scan complete in {time.time() - scan_start:.1f}s: " +
                    ", ".join(f"{c}: {len(cached[c]):,}" for c in to_scan))
        return {c: cached[c] for c in colors}

    def get_or_create_seed_states(self, colors: List[str], dist: np.ndarray, max_states: int, seeds_dir: Path, log_interval: int = DEFAULT_LOG_INTERVAL, force: bool = False, interrupted: Optional[List[bool]] = None, rebuilt_this_run: Optional[set] = None) -> Optional[Dict[str, np.ndarray]]:
        """
        Returns {color: seed_state_ids} for this solver's SEED_CRITERION ('face' or 'layer'),
        using one seeds_dir/<criterion>.npz file (one array per color) per criterion - shared
        by any other method with the same criterion (e.g. Ortega and EG both use 'face'), and
        across separate solver.py invocations. One file per criterion rather than per
        (criterion, color) because they're always read/written as a full set of colors anyway.

        Every 'layer' seed state is also a 'face' seed state (is_layer_solved's first check is
        literally is_face_solved - a solved layer is a solved face with its side stickers also
        aligned), so 'layer' seeds are never found by scanning the full state space again: they
        are derived by running just the extra check (is_layer_solved_given_face) over the
        'face' seed list, which is >99% smaller than the full state space. Returns None if
        interrupted (Ctrl+C) before a required scan/derivation completed - partial results must
        never be cached.

        rebuilt_this_run: see _ensure_face_seeds - shared across calls within one process
        invocation (e.g. main()'s discovery pass, which may call this for multiple methods) so
        --force-tables rebuilds each underlying criterion file at most once per run, even though
        'layer' depends on 'face' and both may be requested in the same run.
        """
        if interrupted is None:
            interrupted = [False]
        if rebuilt_this_run is None:
            rebuilt_this_run = set()

        face_by_color = self._ensure_face_seeds(colors, dist, max_states, seeds_dir, log_interval, force, interrupted, rebuilt_this_run)
        if face_by_color is None:
            return None

        if self.SEED_CRITERION == 'face':
            return face_by_color

        seeds_dir.mkdir(parents=True, exist_ok=True)
        path = seeds_dir / "layer.npz"
        cached = self._load_npz(path)

        effective_force = force and 'layer' not in rebuilt_this_run
        to_derive = list(colors) if effective_force else [c for c in colors if c not in cached]

        if not to_derive:
            logger.info("  Reusing cached 'layer' seed states for all colors")
            return {c: cached[c] for c in colors}

        logger.info(f"  Deriving 'layer' seed states from cached 'face' seeds ({', '.join(to_derive)})...")
        derive_start = time.time()

        for color in to_derive:
            if interrupted[0]: break

            original_colors = self.solving_colors
            self.solving_colors = [color]

            layer_ids = []
            for state_id in face_by_color[color]:
                if interrupted[0]: break
                state_id = int(state_id)
                p7, q7 = self.unpackcube(state_id)
                perm8, ori8 = self.lift_to_full8(p7, q7)
                if self.is_layer_solved_given_face(perm8, ori8):
                    layer_ids.append(state_id)

            self.solving_colors = original_colors

            if interrupted[0]:
                return None

            cached[color] = np.array(layer_ids, dtype=np.int32)

        np.savez(path, **cached)
        rebuilt_this_run.add('layer')

        logger.info(f"  ✓ Derived 'layer' seeds in {time.time() - derive_start:.1f}s: " +
                    ", ".join(f"{c}: {len(cached[c]):,}" for c in to_derive))
        return {c: cached[c] for c in colors}

    def run_analysis(self, dist: np.ndarray, state_transitions: np.ndarray, max_states: int, log_interval: int = 100000, seeds_dir: Optional[Path] = None, force_seeds: bool = False) -> Tuple[Dict[str, np.ndarray], List[Dict]]:
        if seeds_dir is None:
            seeds_dir = Path(DEFAULT_SEEDS_DIR)

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
            logger.warning("\n\n⚠ Interrupted by user (Ctrl+C)")
            interrupted[0] = True

        original_handler = signal.signal(signal.SIGINT, signal_handler)

        try:
            # Phase 1: Find seed states
            logger.info(f"\nPhase 1: Finding seed states for {self.method_name}...")
            p1_start = time.time()

            seed_ids_by_color = self.get_or_create_seed_states(
                list(pocket_cube.COLOR_NEUTRAL), dist, max_states, seeds_dir,
                log_interval=log_interval, force=force_seeds, interrupted=interrupted)
            if seed_ids_by_color is None:
                return color_data, failed_states

            scan_elapsed = time.time() - p1_start
            logger.info("  Seed states available: " +
                        ", ".join(f"{c}: {len(seed_ids_by_color[c]):,}" for c in sorted(pocket_cube.COLOR_NEUTRAL)))

            solve_start = time.time()
            for color in pocket_cube.COLOR_NEUTRAL:
                if interrupted[0]: break

                original_colors = self.solving_colors
                self.solving_colors = [color]

                seed_ids = seed_ids_by_color[color]
                n = len(seed_ids)
                color_start = time.time()
                last_log = color_start

                for i, state_id in enumerate(seed_ids):
                    if interrupted[0]: break
                    # Time-throttled rather than count-based: solving is fast enough now
                    # (compiled algorithms) that a handful of states rarely takes over a
                    # second, so a fixed 10-checkpoint log would mostly just print noise.
                    now = time.time()
                    if now - last_log >= 1.0:
                        logger.info(f"    [{color}] Solved {i + 1:>7,} / {n:,} seed states...")
                        last_log = now

                    state_id = int(state_id)
                    p7, q7 = self.unpackcube(state_id)
                    perm8, ori8 = self.lift_to_full8(p7, q7)

                    result = self.solve_from_state(perm8, ori8)
                    if result['success']:
                        for step_name, move_count in result['moves'].items():
                            if step_name in color_data[color].dtype.names:
                                color_data[color][state_id][step_name] = move_count
                        color_queues[color].append(state_id)
                        seed_counts[color] += 1
                    else:
                        failed_states.append({"id": state_id, "color": color, "error": result.get('error', 'Unknown Error')})

                self.solving_colors = original_colors

                if n > 0 and not interrupted[0]:
                    logger.info(f"    [{color}] Solved {n:,} seed states in {time.time() - color_start:.2f}s")

            if interrupted[0]: return color_data, failed_states

            solve_elapsed = time.time() - solve_start
            logger.info(f"\n  ✓ Phase 1 Complete in {time.time() - p1_start:.1f}s (seed lookup: {scan_elapsed:.1f}s, solving: {solve_elapsed:.1f}s)")
            logger.info(f"  {'='*30}\n  SOLVED COUNTS (of the seed states found above):")
            for c in sorted(pocket_cube.COLOR_NEUTRAL):
                logger.info(f"    Color {c}: {seed_counts[c]:>9,} states")
            logger.info(f"  {'-'*30}")
            logger.info(f"    Failures: {len(failed_states):>9,} states")
            logger.info(f"  {'='*30}")

            # Phase 2: BFS with Double-Buffering
            logger.info(f"\nPhase 2: Propagating distances using global transition table...")
            first_step = structural_steps[0]

            for color in pocket_cube.COLOR_NEUTRAL:
                if interrupted[0]: break
                if seed_counts[color] == 0: continue

                logger.info(f"\n  Propagating {color}...")
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
                    logger.info(f"    [Depth {depth:>2}] Propagating {q_len:>9,} states...")
                    q_len = propagate_bfs_kernel(data_matrix, curr_q[:q_len], next_q, state_transitions, max_states, color_data[color].dtype.names.index(first_step))
                    curr_q, next_q = next_q, curr_q
                    depth += 1

                logger.info(f"  ✓ {color} Complete: {total_propagated:,} states mapped in {time.time() - p2_start:.1f}s")
        finally:
            signal.signal(signal.SIGINT, signal.SIG_DFL)
        return color_data, failed_states

# ==========================================
# GLOBAL SOLVER & EXECUTION
# ==========================================

def get_or_create_global_transitions(max_states: int, moves: List[str], log_interval: int = DEFAULT_LOG_INTERVAL, transition_file: str = DEFAULT_TRANSITION_NPY, force: bool = False) -> np.ndarray:
    path = Path(transition_file)
    if path.exists() and not force:
        logger.info(f"\n[SOLVER] Loading existing global state transitions from {transition_file}...")
        start_time = time.time()
        transitions = np.load(path)
        logger.info(f"✓ Loaded in {time.time() - start_time:.2f}s")
        return transitions

    logger.info(f"\n[SOLVER] Building global state transition table ({max_states:,} states)...")
    start_time = time.time()
    moves_perm = np.array([pocket_cube.MOVES[m][0] for m in moves], dtype=np.int64)
    moves_twist = np.array([pocket_cube.MOVES[m][1] for m in moves], dtype=np.int64)
    transitions = _build_transitions_kernel(moves_perm, moves_twist, max_states)

    logger.info(f"✓ Global transition table built in {time.time() - start_time:.1f}s")

    logger.info(f"\n[SOLVER] Saving global state transitions to {transition_file}...")
    np.save(path, transitions)
    return transitions

def get_or_create_optimal_distances(transitions: np.ndarray, max_states: int, dist_file: str = DEFAULT_DIST_NPY, force: bool = False) -> np.ndarray:
    path = Path(dist_file)
    if path.exists() and not force:
        logger.info(f"\n[SOLVER] Loading existing optimal distances from {dist_file}")
        return np.load(path)

    logger.info(f"\n[SOLVER] Calculating optimal distances from solved state...")
    start_time = time.time()
    dist = generate_optimal_depths(transitions, max_states)
    np.save(path, dist)
    logger.info(f"✓ Saved to {dist_file} in {time.time() - start_time:.1f}s")
    return dist

def load_solver(method: str) -> Solver:
    algorithm_dir = Path(__file__).parent / "algorithms"
    algorithm_file = algorithm_dir / f"{method}.json"

    if not algorithm_file.exists():
        raise FileNotFoundError(f"Algorithm file not found: {algorithm_file}")

    from method_solvers.solver_ortega import OrtegaSolver
    from method_solvers.solver_cll import CLLSolver
    from method_solvers.solver_lbl import LBLSolver
    from method_solvers.solver_eg import EGSolver

    solver_classes = {
        'ortega': OrtegaSolver,
        'cll': CLLSolver,
        'lbl': LBLSolver,
        'eg': EGSolver,
    }
    if method not in solver_classes:
        raise ValueError(f"Unknown method: {method}")
    return solver_classes[method](str(algorithm_file))

def main():
    parser = argparse.ArgumentParser(description='2x2x2 Rubik\'s Cube Solver')
    parser.add_argument('--dist-npy', default=DEFAULT_DIST_NPY, help='Optimal distances file')
    parser.add_argument('--transition-npy', default=DEFAULT_TRANSITION_NPY, help='State transitions cache file')
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR, help='Directory to save analysis results')
    parser.add_argument('--seeds-dir', default=DEFAULT_SEEDS_DIR, help='Directory to cache per-color seed-state lookups (shared across methods with the same seed criterion)')
    parser.add_argument('--methods', nargs='+', choices=['ortega', 'cll', 'lbl', 'eg', 'all'], default=DEFAULT_METHODS)
    parser.add_argument('--log-interval', type=int, default=DEFAULT_LOG_INTERVAL, help='Log interval for long operations')
    parser.add_argument('--force', action='store_true', help='Force recalculation even if method .npy files exist')
    parser.add_argument('--force-tables', action='store_true', help='Rebuild the transition/distance/seed-state tables even if their cache files exist')
    args = parser.parse_args()

    total_start_time = time.time()
    max_states = pocket_cube.N_STATES
    moves = pocket_cube.REDUCED_MOVES

    transitions = get_or_create_global_transitions(max_states, moves, args.log_interval, args.transition_npy, force=args.force_tables)
    dist = get_or_create_optimal_distances(transitions, max_states, args.dist_npy, force=args.force_tables)

    methods = ['ortega', 'cll', 'lbl', 'eg'] if 'all' in args.methods else args.methods
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    seeds_dir = Path(args.seeds_dir)

    # Decide up front which methods actually need analysis (skip already-cached ones), then
    # instantiate each once so both the seed-discovery pass and the analysis loop below reuse
    # the same solver objects.
    solvers_to_run: Dict[str, Solver] = {}
    for method in methods:
        method_path = output_dir / f"{method}.npz"
        if method_path.exists() and not args.force:
            with np.load(method_path) as npz:
                all_cached = all(c in npz.files for c in pocket_cube.COLOR_NEUTRAL)
            if all_cached:
                logger.info(f"\n{'='*80}\n⏭️ SKIPPING: {method.upper()} (Already cached in {method_path})\n{'='*80}")
                continue
        solvers_to_run[method] = load_solver(method)

    # Seed discovery: one scan per distinct SEED_CRITERION (e.g. Ortega/EG share 'face',
    # CLL/LBL share 'layer') for all methods about to run, before any of them are analyzed -
    # so no individual method's Phase 1 ever triggers its own scan.
    if solvers_to_run:
        logger.info(f"\n{'='*80}\nDISCOVERING SEED STATES for {', '.join(m.upper() for m in solvers_to_run)}\n{'='*80}")
        seen_criteria = set()
        rebuilt_this_run: set = set()
        for solver in solvers_to_run.values():
            if solver.SEED_CRITERION in seen_criteria:
                continue
            seen_criteria.add(solver.SEED_CRITERION)
            solver.get_or_create_seed_states(list(pocket_cube.COLOR_NEUTRAL), dist, max_states, seeds_dir, log_interval=args.log_interval, force=args.force_tables, rebuilt_this_run=rebuilt_this_run)

    for method, solver in solvers_to_run.items():
        logger.info(f"\n{'='*80}\nANALYZING: {method.upper()}\n{'='*80}")
        # force_seeds=False here: the discovery pass above already (re)built the cache for
        # every needed criterion, including honoring --force-tables, so this is always a
        # cache hit - never a second rescan of the same criterion.
        color_data, failed = solver.run_analysis(dist, transitions, max_states, args.log_interval, seeds_dir=seeds_dir, force_seeds=False)

        valid_colors = {color: data for color, data in color_data.items() if np.any(data['depth'] >= 0)}
        method_path = output_dir / f"{method}.npz"
        np.savez(method_path, **valid_colors)
        logger.info(f"  [+] Saved {method_path} ({', '.join(sorted(valid_colors))})")

    logger.info(f"\n✅ All analysis complete in {(time.time() - total_start_time)/60:.1f} minutes!")

if __name__ == "__main__":
    main()