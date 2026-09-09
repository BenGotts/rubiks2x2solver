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
        p_oll, o_oll, pre_auf, oll_moves, pre_auf_str, oll_str = self._solve_oll(p_norm, o_norm)
        if oll_moves == -1: return {'success': False, 'error': 'OLL Fail'}

        # Step 3: Solve PBL (Permute Both Layers)
        mid_auf, pbl_moves, post_auf, mid_auf_str, rot_str, pbl_str, post_auf_str = self._solve_pbl(p_oll, o_oll)
        if pbl_moves == -1: return {'success': False, 'error': 'PBL Fail'}

        naive_total = pre_auf + oll_moves + mid_auf + pbl_moves + post_auf
        reduced_total = self._solve_cost([pre_auf_str, oll_str, mid_auf_str, rot_str, pbl_str, post_auf_str])

        def _reduced_from(p, o):
            p_o, o_o, pre_, oll_, pre_str_, oll_str_ = self._solve_oll(p, o)
            if oll_ == -1: return None
            mid_, pbl_, post_, mid_str_, rot_str_, pbl_str_, post_str_ = self._solve_pbl(p_o, o_o)
            if pbl_ == -1: return None
            return self._solve_cost([pre_str_, oll_str_, mid_str_, rot_str_, pbl_str_, post_str_])

        reduced_total = self._best_with_layer_rotation(p_norm, o_norm, reduced_total, _reduced_from)

        return {
            'success': True,
            'moves': {
                'face': 0,
                'pre_auf': pre_auf,
                'oll': oll_moves,
                'mid_auf': mid_auf,
                'pbl': pbl_moves,
                'post_auf': post_auf,
                'reduced_savings': naive_total - reduced_total,
            }
        }

    def _solve_oll(self, perm8: np.ndarray, ori8: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int, int, str, str]:
        """
        Solve OLL and extract the Pre-AUF cost.

        Returns:
            (new_perm8, new_ori8, pre_auf_cost, alg_cost, pre_auf_str, oll_alg_str)
        """
        stickers = self.get_stickers8(perm8, ori8)

        # Check if already oriented (OLL Skip)
        if stickers[0] == stickers[1] == stickers[2] == stickers[3]:
            return perm8, ori8, 0, 0, "", ""

        # Try Pre-AUF + algorithm combinations
        for auf_str, compiled_auf in zip(self.AUF_MOVES, self.compiled_auf):
            p_auf, o_auf, auf_cost = self._apply_compiled(perm8, ori8, compiled_auf)

            for name, compiled in self.compiled_algorithms.get('oll', {}).items():
                p_res, o_res, alg_cost = self._apply_compiled(p_auf, o_auf, compiled)
                s = self.get_stickers8(p_res, o_res)

                # Check if the top face is a single solid color
                if s[0] == s[1] == s[2] == s[3]:
                    return p_res, o_res, auf_cost, alg_cost, auf_str, self.algorithms['oll'][name]

        return perm8, ori8, -1, -1, "", ""

    def _solve_pbl(self, perm8: np.ndarray, ori8: np.ndarray) -> Tuple[int, int, int, str, str, str, str]:
        """
        Solve PBL using Y-rotations for alignment, extracting AUF costs.

        Returns:
            (mid_auf_cost, pbl_cost, post_auf_cost, mid_auf_str, rot_str, pbl_alg_str, post_auf_str)
        """
        # 1. Check for PBL Skip (Just needs Post-AUF alignment)
        for post_str, post_compiled in zip(self.AUF_MOVES, self.compiled_auf):
            p_test, o_test, post_cost = self._apply_compiled(perm8, ori8, post_compiled)
            if self.is_solved_state(p_test, o_test):
                return 0, 0, post_cost, "", "", "", post_str

        # 2. Try Mid-AUF (U) + Rotation (y) + Algorithm + Post-AUF (U)
        for mid_str, mid_compiled in zip(self.AUF_MOVES, self.compiled_auf):
            p_mid, o_mid, mid_cost = self._apply_compiled(perm8, ori8, mid_compiled)

            # Apply zero-cost Y rotations to align the bars for the algorithm
            for rot in ["", "y", "y2", "y'"]:
                if rot:
                    p_rot, o_rot = self._apply_rotation(p_mid, o_mid, rot)
                else:
                    p_rot, o_rot = p_mid, o_mid

                for name, compiled in self.compiled_algorithms.get('pbl', {}).items():
                    p_alg, o_alg, alg_cost = self._apply_compiled(p_rot, o_rot, compiled)

                    # Check if it just needs a final U turn to be solved
                    for post_str, post_compiled in zip(self.AUF_MOVES, self.compiled_auf):
                        p_post, o_post, post_cost = self._apply_compiled(p_alg, o_alg, post_compiled)

                        if self.is_solved_state(p_post, o_post):
                            return mid_cost, alg_cost, post_cost, mid_str, rot, self.algorithms['pbl'][name], post_str

        return -1, -1, -1, "", "", "", ""