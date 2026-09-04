"""
Ortega Method Solver for 2x2x2 Rubik's Cube.

Steps:
1. Solve any face (intuitive)
2. Pre-AUF + Orient Last Layer (OLL)
3. Mid-AUF + Permute Both Layers (PBL) + Post-AUF
"""

import numpy as np
from typing import Dict, Tuple
from solver import Solver

class OrtegaSolver(Solver):
    """Solver implementing the Ortega method."""

    # Shared with any other method whose seed states are "one face solved", so a Phase 1
    # seed scan can be reused across methods instead of repeating it (see Solver.run_analysis).
    SEED_CRITERION = 'face'

    def is_seed_state(self, perm8: np.ndarray, ori8: np.ndarray) -> bool:
        """Seed states are those with one face solved."""
        return self.is_face_solved(perm8, ori8)
    
    def solve_from_state(self, perm8: np.ndarray, ori8: np.ndarray) -> Dict:
        """
        Solve cube using Ortega method.
        
        Returns:
            Dictionary with 'success', 'moves', and optional 'error'
        """
        # Step 1: Normalize to D (Put the solved face on the bottom)
        p_norm, o_norm = self.normalize_to_d(perm8, ori8)
        if p_norm is None: return {'success': False, 'error': 'Normalize Fail'}
        
        # Step 2: Solve OLL (Orient Last Layer)
        p_oll, o_oll, pre_auf, oll_moves = self._solve_oll(p_norm, o_norm)
        if oll_moves == -1: return {'success': False, 'error': 'OLL Fail'}
        
        # Step 3: Solve PBL (Permute Both Layers)
        mid_auf, pbl_moves, post_auf = self._solve_pbl(p_oll, o_oll)
        if pbl_moves == -1: return {'success': False, 'error': 'PBL Fail'}
        
        return {
            'success': True,
            'moves': {
                'face': 0,
                'pre_auf': pre_auf,
                'oll': oll_moves,
                'mid_auf': mid_auf,
                'pbl': pbl_moves,
                'post_auf': post_auf
            }
        }
    
    def _solve_oll(self, perm8: np.ndarray, ori8: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """
        Solve OLL and extract the Pre-AUF cost.
        
        Returns:
            (new_perm8, new_ori8, pre_auf_cost, alg_cost)
        """
        stickers = self.get_stickers8(perm8, ori8)

        # Check if already oriented (OLL Skip)
        if stickers[0] == stickers[1] == stickers[2] == stickers[3]:
            return perm8, ori8, 0, 0

        # Try Pre-AUF + algorithm combinations
        for compiled_auf in self.compiled_auf:
            p_auf, o_auf, auf_cost = self._apply_compiled(perm8, ori8, compiled_auf)

            for compiled in self.compiled_algorithms.get('oll', {}).values():
                p_res, o_res, alg_cost = self._apply_compiled(p_auf, o_auf, compiled)
                s = self.get_stickers8(p_res, o_res)

                # Check if the top face is a single solid color
                if s[0] == s[1] == s[2] == s[3]:
                    return p_res, o_res, auf_cost, alg_cost

        return perm8, ori8, -1, -1

    def _solve_pbl(self, perm8: np.ndarray, ori8: np.ndarray) -> Tuple[int, int, int]:
        """
        Solve PBL using Y-rotations for alignment, extracting AUF costs.

        Returns:
            (mid_auf_cost, pbl_cost, post_auf_cost)
        """
        # 1. Check for PBL Skip (Just needs Post-AUF alignment)
        for post_compiled in self.compiled_auf:
            p_test, o_test, post_cost = self._apply_compiled(perm8, ori8, post_compiled)
            if self.is_solved_state(p_test, o_test):
                return 0, 0, post_cost

        # 2. Try Mid-AUF (U) + Rotation (y) + Algorithm + Post-AUF (U)
        for mid_compiled in self.compiled_auf:
            p_mid, o_mid, mid_cost = self._apply_compiled(perm8, ori8, mid_compiled)

            # Apply zero-cost Y rotations to align the bars for the algorithm
            for rot in ["", "y", "y2", "y'"]:
                if rot:
                    p_rot, o_rot = self._apply_rotation(p_mid, o_mid, rot)
                else:
                    p_rot, o_rot = p_mid, o_mid

                for compiled in self.compiled_algorithms.get('pbl', {}).values():
                    p_alg, o_alg, alg_cost = self._apply_compiled(p_rot, o_rot, compiled)

                    # Check if it just needs a final U turn to be solved
                    for post_compiled in self.compiled_auf:
                        p_post, o_post, post_cost = self._apply_compiled(p_alg, o_alg, post_compiled)

                        if self.is_solved_state(p_post, o_post):
                            return mid_cost, alg_cost, post_cost

        return -1, -1, -1