"""
Base Solver class for 2x2x2 Rubik's Cube solving methods.

Inherits from PocketCube and adds solving-specific functionality.
"""

import json
import numpy as np
import signal
import sys
from typing import Dict, List, Tuple, Optional
from collections import deque
from pathlib import Path
from abc import ABC, abstractmethod

import pocket_cube

# Import numba for JIT compilation
try:
    from numba import jit
    NUMBA_AVAILABLE = True
    print("✓ Numba JIT compiler available - using optimized code paths")
except ImportError:
    NUMBA_AVAILABLE = False
    print("⚠ Numba not available - using standard Python (slower)")
    # Dummy decorator if numba not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


# JIT-compiled helper functions for performance
@jit(nopython=True, cache=True)
def propagate_bfs_kernel(
    first_step_data: np.ndarray,
    step_data_list: List[np.ndarray],
    queue_array: np.ndarray,
    queue_start: int,
    queue_end: int,
    state_transitions: np.ndarray,
    max_states: int
) -> Tuple[np.ndarray, int]:
    """
    JIT-compiled BFS propagation kernel.
    
    Args:
        first_step_data: Array for first step move counts
        step_data_list: List of arrays for other steps
        queue_array: Array of state IDs to process
        queue_start: Start index in queue
        queue_end: End index in queue
        state_transitions: [num_moves, max_states] array of state transitions
        max_states: Maximum state ID
        
    Returns:
        (new_states_array, num_expanded)
    """
    new_states = []
    num_expanded = 0
    num_moves = state_transitions.shape[0]
    num_other_steps = len(step_data_list)
    
    for i in range(queue_start, queue_end):
        v = queue_array[i]
        parent_dist = first_step_data[v]
        
        # Try all moves
        for move_idx in range(num_moves):
            w = state_transitions[move_idx, v]
            
            # Check if valid and unvisited
            if w < max_states and w >= 0 and first_step_data[w] == -1:
                first_step_data[w] = parent_dist + 1
                
                # Copy other step values
                for step_idx in range(num_other_steps):
                    step_data_list[step_idx][w] = step_data_list[step_idx][v]
                
                new_states.append(w)
                num_expanded += 1
    
    return np.array(new_states, dtype=np.int32), num_expanded


class Solver(pocket_cube.PocketCube, ABC):
    """
    Abstract base class for 2x2 solving methods.
    
    Inherits from PocketCube and adds:
    - Algorithm loading from JSON
    - Solving logic
    - Analysis capabilities
    - Color-specific solving
    """
    
    def __init__(
        self,
        algorithm_file: str,
        solving_colors: List[str] = None,
        state_id: int = 0
    ):
        """
        Initialize solver.
        
        Args:
            algorithm_file: Path to JSON file with algorithms
            solving_colors: List of colors considered "solved"
            state_id: Initial cube state
        """
        # Initialize parent PocketCube
        super().__init__(state_id)
        
        # Set solving colors
        if solving_colors is None:
            solving_colors = [pocket_cube.WHITE, pocket_cube.YELLOW]
        self.solving_colors = [c.upper() for c in solving_colors]
        
        # Load algorithm configuration
        with open(algorithm_file, 'r') as f:
            self.config = json.load(f)
        
        self.method_name = self.config.get('name', 'Unknown')
        self.steps = self.config.get('steps', [])
        
        # Build algorithm dictionary
        self.algorithms = {}
        for step in self.steps:
            step_name = step['name']
            self.algorithms[step_name] = step.get('algorithms', {})
        
        # State transitions cache for JIT optimization
        self._state_transitions = None
        self._transitions_max_states = None
    
    def _build_state_transitions(self, max_states: int) -> np.ndarray:
        """
        Build state transition table for fast BFS (full move set).
        
        Precomputes: state_transitions[move_idx, state_id] = new_state_id
        
        This is expensive but makes BFS propagation much faster.
        """
        if self._state_transitions is not None and self._transitions_max_states == max_states:
            return self._state_transitions
        
        print(f"  Building state transition table for {max_states:,} states...")
        
        move_list = list(self.MOVES.keys())
        num_moves = len(move_list)
        
        # Allocate transition table
        state_transitions = np.full((num_moves, max_states), -1, dtype=np.int32)
        
        # Precompute all transitions
        for state_id in range(max_states):
            if state_id % 500000 == 0 and state_id > 0:
                print(f"    {state_id:,} / {max_states:,}...")
            
            p7, q7 = self.unpackcube(state_id)
            perm8, ori8 = self.lift_to_full8(p7, q7)
            
            for move_idx, move_name in enumerate(move_list):
                np8, no8 = self._apply_move(perm8, ori8, move_name)
                pp7, qq7 = self.project_to_7(np8, no8)
                w = self.packcube((pp7, qq7))
                
                if w < max_states:
                    state_transitions[move_idx, state_id] = w
        
        print(f"  ✓ State transition table built")
        
        # Cache for reuse
        self._state_transitions = state_transitions
        self._transitions_max_states = max_states
        
        return state_transitions
    
    def _build_state_transitions_reduced(self, max_states: int, move_list: List[str]) -> np.ndarray:
        """
        Build state transition table for fast BFS (reduced move set).
        
        Uses only specified moves (e.g., R, U, B for 2x2).
        """
        print(f"  Building state transition table for {max_states:,} states ({len(move_list)} moves)...")
        
        num_moves = len(move_list)
        
        # Allocate transition table
        state_transitions = np.full((num_moves, max_states), -1, dtype=np.int32)
        
        # Precompute all transitions
        for state_id in range(max_states):
            if state_id % 500000 == 0 and state_id > 0:
                print(f"    {state_id:,} / {max_states:,}...")
            
            p7, q7 = self.unpackcube(state_id)
            perm8, ori8 = self.lift_to_full8(p7, q7)
            
            for move_idx, move_name in enumerate(move_list):
                np8, no8 = self._apply_move(perm8, ori8, move_name)
                pp7, qq7 = self.project_to_7(np8, no8)
                w = self.packcube((pp7, qq7))
                
                if w < max_states:
                    state_transitions[move_idx, state_id] = w
        
        print(f"  ✓ State transition table built ({len(move_list)} moves)")
        
        return state_transitions
    
    # ==========================================
    # STATE CHECKING WITH SOLVING COLORS
    # ==========================================
    
    def is_face_solved(self, perm8: np.ndarray = None, ori8: np.ndarray = None) -> bool:
        """Check if any face is solved with solving_colors."""
        if perm8 is None:
            perm8, ori8 = self.perm8, self.ori8
        
        stickers = self.get_stickers8(perm8, ori8)
        for i in range(0, 24, 4):
            face = stickers[i:i+4]
            if len(set(face)) == 1 and face[0] in self.solving_colors:
                return True
        return False
    
    def is_layer_solved(self, perm8: np.ndarray = None, ori8: np.ndarray = None) -> bool:
        """Check if first layer is solved."""
        if perm8 is None:
            perm8, ori8 = self.perm8, self.ori8
            
        if not self.is_face_solved(perm8, ori8):
            return False
        
        p, o = self.normalize_to_d(perm8, ori8)
        if p is None or o is None:
            return False
        
        stickers = self.get_stickers8(p, o)
        
        # Check if D-layer adjacent colors match
        if (stickers[19] == stickers[18] and 
            stickers[11] == stickers[10] and 
            stickers[7] == stickers[6] and 
            stickers[23] == stickers[22]):
            return True
        return False
    
    def normalize_to_d(
        self, 
        perm8: np.ndarray = None, 
        ori8: np.ndarray = None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Rotate cube so a solved face is on D (bottom).
        
        Returns:
            (perm8, ori8) tuple if successful, (None, None) if no solved face
        """
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
    
    # ==========================================
    # ALGORITHM APPLICATION (NON-MUTATING)
    # ==========================================
    
    def _apply_rotation(
        self, 
        perm8: np.ndarray, 
        ori8: np.ndarray, 
        rotation: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply cube rotation without modifying input arrays."""
        p, t = self.ROTATIONS[rotation]
        return perm8[p], (ori8[p] + t) % 3
    
    def _apply_move(
        self,
        perm8: np.ndarray,
        ori8: np.ndarray,
        move: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply single move without modifying input arrays."""
        p, t = self.MOVES[move]
        return perm8[p], (ori8[p] + t) % 3
    
    def _apply_algorithm(
        self, 
        perm8: np.ndarray, 
        ori8: np.ndarray, 
        alg: str
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Apply algorithm to state arrays without modification.
        
        Args:
            perm8: Permutation array
            ori8: Orientation array
            alg: Space-separated move sequence
            
        Returns:
            (new_perm8, new_ori8, move_count)
        """
        temp_perm, temp_ori = perm8.copy(), ori8.copy()
        count = 0
        
        if not alg:
            return temp_perm, temp_ori, 0
            
        for move in alg.split():
            if move in self.ROTATIONS:
                temp_perm, temp_ori = self._apply_rotation(temp_perm, temp_ori, move)
            elif move in self.MOVES:
                temp_perm, temp_ori = self._apply_move(temp_perm, temp_ori, move)
                count += 1
            else:
                raise ValueError(f"Invalid move '{move}' in algorithm: {alg}")
                    
        return temp_perm, temp_ori, count
    
    def _get_solved_colors(
        self, 
        perm8: np.ndarray, 
        ori8: np.ndarray
    ) -> List[str]:
        """
        Get list of colors that have a face solved.
        
        Returns:
            List of color letters (e.g., ['W', 'G'])
        """
        stickers = self.get_stickers8(perm8, ori8)
        solved_colors = []
        
        # Check each face
        faces = {
            'U': (0, 4),     # stickers[0:4]
            'R': (4, 8),     # stickers[4:8]
            'F': (8, 12),    # stickers[8:12]
            'D': (12, 16),   # stickers[12:16]
            'L': (16, 20),   # stickers[16:20]
            'B': (20, 24),   # stickers[20:24]
        }
        
        for face_name, (start, end) in faces.items():
            face_stickers = stickers[start:end]
            if len(set(face_stickers)) == 1:
                # This face is solved
                color = face_stickers[0]
                solved_colors.append(color)
        
        return solved_colors
    
    def pretty_print_cube(self, perm8: np.ndarray = None, ori8: np.ndarray = None):
        """Print ASCII representation of cube state."""
        if perm8 is None:
            perm8, ori8 = self.perm8, self.ori8
        
        stickers = self.get_stickers8(perm8, ori8)
        
        # Use parent class static method
        pocket_cube.PocketCube.pretty_print_cube(stickers)
    
    # ==========================================
    # ABSTRACT METHODS
    # ==========================================
    
    @abstractmethod
    def solve_from_state(
        self, 
        perm8: np.ndarray, 
        ori8: np.ndarray
    ) -> Dict:
        """
        Solve cube from given state.
        
        Must return a dictionary with:
        - 'success': bool
        - 'moves': dict mapping step names to move counts
        - Optional: 'error': str if failed
        
        Args:
            perm8: Permutation array
            ori8: Orientation array
            
        Returns:
            Dictionary with solve results
        """
        pass
    
    @abstractmethod
    def is_seed_state(
        self, 
        perm8: np.ndarray, 
        ori8: np.ndarray
    ) -> bool:
        """
        Check if state is a seed (starting point for BFS).
        Must be implemented by subclasses.
        """
        pass
    
    # ==========================================
    # ANALYSIS
    # ==========================================
    
    def run_analysis(
        self,
        dist: np.ndarray,
        max_states: Optional[int] = None,
        log_interval: int = 100000
    ) -> Tuple[Dict[str, np.ndarray], List[Dict]]:
        """
        Run full BFS analysis of method performance.
        
        Creates separate data arrays for each color.
        Handles Ctrl+C gracefully by saving progress and failures.
        
        Args:
            dist: Distance-to-solved array for all states
            max_states: Maximum number of states to analyze
            log_interval: Print progress every N states
            
        Returns:
            (color_data_dict, failed_states_list)
            color_data_dict: Dictionary mapping color -> data array
        """
        if max_states is None:
            max_states = pocket_cube.N_STATES
        else:
            max_states = min(max_states, pocket_cube.N_STATES)
        
        # Create data structure for each color
        step_names = [step["name"] for step in self.steps]
        dtype_list = [('depth', 'i1')] + [(name, 'i1') for name in step_names]
        dt = np.dtype(dtype_list)
        
        # Initialize data for each color
        color_data = {}
        for color in pocket_cube.COLOR_NEUTRAL:
            color_data[color] = np.full(max_states, -1, dtype=dt)
            limit_len = min(len(dist), max_states)
            color_data[color]['depth'][:limit_len] = dist[:limit_len]
        
        # Track which states were successfully seeded per color
        color_queues = {color: deque() for color in pocket_cube.COLOR_NEUTRAL}
        failed_states = []
        seed_counts = {color: 0 for color in pocket_cube.COLOR_NEUTRAL}
        
        # Flag for interruption
        interrupted = [False]  # Use list to allow modification in nested function
        
        def signal_handler(sig, frame):
            """Handle Ctrl+C gracefully."""
            print("\n\n⚠ Interrupted by user (Ctrl+C)")
            print("Saving current progress and failures...")
            interrupted[0] = True
        
        # Register signal handler
        original_handler = signal.signal(signal.SIGINT, signal_handler)
        
        try:
            # Phase 1: Find seed states for each color
            print(f"\nPhase 1: Finding seed states for {self.method_name}...")
            print(f"Testing all colors: {list(pocket_cube.COLOR_NEUTRAL)}")
            print(f"(Press Ctrl+C to stop and save progress)")
            
            for state_id in range(max_states):
                if interrupted[0]:
                    print(f"\nStopping Phase 1 at state {state_id:,}")
                    break
                
                if state_id % log_interval == 0 and state_id > 0:
                    total_seeds = sum(seed_counts.values())
                    print(f"  Scanned {state_id:,}... Total seeds: {total_seeds:,}, Failures: {len(failed_states)}")
                
                if dist[state_id] < 0:
                    continue
                
                p7, q7 = self.unpackcube(state_id)
                perm8, ori8 = self.lift_to_full8(p7, q7)
                
                # Check if this is a seed state (any face solved)
                if self.is_seed_state(perm8, ori8):
                    # Get which colors have faces solved
                    solved_colors = self._get_solved_colors(perm8, ori8)
                    
                    # Try solving for each color that has a face
                    any_success = False
                    for color in solved_colors:
                        # Temporarily set solving colors to this single color
                        original_colors = self.solving_colors
                        self.solving_colors = [color]
                        
                        result = self.solve_from_state(perm8, ori8)
                        
                        self.solving_colors = original_colors
                        
                        if result['success']:
                            # Store data for this specific color
                            for step_name, move_count in result['moves'].items():
                                if step_name in step_names:
                                    color_data[color][state_id][step_name] = move_count
                            color_queues[color].append(state_id)
                            seed_counts[color] += 1
                            any_success = True
                    
                    if not any_success:
                        # Save failure with cube visualization
                        stickers = self.get_stickers8(perm8, ori8)
                        failed_states.append({
                            "id": int(state_id),
                            "error": "All colors failed",
                            "perm8": perm8.tolist(),
                            "ori8": ori8.tolist(),
                            "stickers": stickers,
                            "solved_colors": solved_colors
                        })
            
            if not interrupted[0]:
                print(f"Phase 1 Complete.")
            print(f"  Seed counts per color:")
            for color in sorted(seed_counts.keys()):
                print(f"    {color}: {seed_counts[color]:,}")
            print(f"  Total failures: {len(failed_states)}")
            
            # Don't run Phase 2 if interrupted during Phase 1
            if interrupted[0]:
                print("\nSkipping Phase 2 due to interruption")
                return color_data, failed_states
            
            # Phase 2: BFS propagation for each color independently
            print(f"\nPhase 2: Propagating distances for each color...")
            print(f"  Using reduced move set (R, U, B): 9 moves instead of 18")
            print(f"(Press Ctrl+C to stop and save progress)")
            
            first_step = step_names[0]
            num_steps = len(step_names)
            
            # Use reduced move set for BFS (R, U, B only - equivalent on 2x2)
            move_list = pocket_cube.REDUCED_MOVES
            num_moves = len(move_list)
            
            print(f"  Moves: {', '.join(move_list)}")
            
            # Build state transition table for JIT optimization
            if NUMBA_AVAILABLE:
                state_transitions = self._build_state_transitions_reduced(max_states, move_list)
            else:
                state_transitions = None
            
            for color in pocket_cube.COLOR_NEUTRAL:
                if interrupted[0]:
                    print(f"\n⚠ Stopping at color {color}")
                    break
                
                if seed_counts[color] == 0:
                    print(f"  {color}: No seeds, skipping")
                    continue
                
                print(f"\n  Propagating {color}...")
                queue = color_queues[color]
                data = color_data[color]
                
                states_processed = 0
                states_expanded = 0
                last_log = 0
                
                # Get direct access to data arrays for faster indexing
                first_step_data = data[first_step]
                step_data_arrays = [data[step_name] for step_name in step_names[1:]]
                
                if NUMBA_AVAILABLE and state_transitions is not None:
                    # Use JIT-compiled fast path
                    queue_array = np.array(list(queue), dtype=np.int32)
                    queue.clear()
                    
                    while len(queue_array) > 0 and not interrupted[0]:
                        states_processed += len(queue_array)
                        
                        # Log progress
                        if states_processed - last_log >= log_interval:
                            parent_dist = first_step_data[queue_array[0]] if len(queue_array) > 0 else 0
                            print(f"    Processed: {states_processed:,} | Queue: {len(queue_array):,} | Depth: {parent_dist} | Expanded: {states_expanded:,}")
                            last_log = states_processed
                        
                        # Process batch with JIT kernel
                        new_states, batch_expanded = propagate_bfs_kernel(
                            first_step_data,
                            step_data_arrays,
                            queue_array,
                            0,
                            len(queue_array),
                            state_transitions,
                            max_states
                        )
                        
                        states_expanded += batch_expanded
                        queue_array = new_states
                    
                else:
                    # Standard Python path
                    while queue and not interrupted[0]:
                        v = queue.popleft()
                        states_processed += 1
                        parent_dist = first_step_data[v]
                        
                        # Log progress
                        if states_processed - last_log >= log_interval:
                            print(f"    Processed: {states_processed:,} | Queue: {len(queue):,} | Depth: {parent_dist} | Expanded: {states_expanded:,}")
                            last_log = states_processed
                        
                        # Unpack once per parent
                        p7, q7 = self.unpackcube(v)
                        perm8, ori8 = self.lift_to_full8(p7, q7)
                        
                        # Try all moves
                        for move_name in move_list:
                            np8, no8 = self._apply_move(perm8, ori8, move_name)
                            pp7, qq7 = self.project_to_7(np8, no8)
                            w = self.packcube((pp7, qq7))
                            
                            # Check bounds and if unvisited
                            if w < max_states and first_step_data[w] == -1:
                                first_step_data[w] = parent_dist + 1
                                # Copy other step values
                                for i, step_array in enumerate(step_data_arrays):
                                    step_array[w] = step_data_arrays[i][v]
                                queue.append(w)
                                states_expanded += 1
                
                print(f"    {color} Complete. Processed: {states_processed:,}, Expanded: {states_expanded:,}")
            
            if not interrupted[0]:
                print(f"\nPhase 2 Complete for all colors.")
            else:
                print(f"\n⚠ Phase 2 interrupted - partial results saved")
            
        finally:
            # Restore original signal handler
            signal.signal(signal.SIGINT, original_handler)
        
        return color_data, failed_states


def load_solver(method: str, solving_colors: List[str] = None) -> Solver:
    """
    Factory function to load appropriate solver.
    
    Args:
        method: Method name ('ortega', 'cll', 'lbl', 'eg')
        solving_colors: Colors to solve for
        
    Returns:
        Instantiated solver
    """
    algorithm_dir = Path(__file__).parent / "algorithms"
    algorithm_file = algorithm_dir / f"{method}.json"
    
    if not algorithm_file.exists():
        raise FileNotFoundError(f"Algorithm file not found: {algorithm_file}")
    
    # Import method-specific solver
    if method == 'ortega':
        from solver_ortega import OrtegaSolver
        return OrtegaSolver(str(algorithm_file), solving_colors)
    elif method == 'cll':
        from solver_cll import CLLSolver
        return CLLSolver(str(algorithm_file), solving_colors)
    elif method == 'lbl':
        from solver_lbl import LBLSolver
        return LBLSolver(str(algorithm_file), solving_colors)
    elif method == 'eg':
        from solver_eg import EGSolver
        return EGSolver(str(algorithm_file), solving_colors)
    else:
        raise ValueError(f"Unknown method: {method}")