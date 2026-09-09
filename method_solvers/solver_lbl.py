"""
LBL (Layer-By-Layer / Beginner) Method Solver for 2x2x2 Rubik's Cube.

Steps:
1. Solve first layer completely (intuitive)
2. Pre-AUF + Orient Last Layer (OLL)
3. Mid-AUF + Permute Last Layer (PLL) + Post-AUF
"""

import numpy as np
from typing import Dict, Tuple
from solver import Solver


class LBLSolver(Solver):
    """Solver implementing the Layer-By-Layer (Beginner) method."""

    # Shared with any other method whose seed states are "first layer solved" (see CLLSolver).
    SEED_CRITERION = 'layer'

    def is_seed_state(self, perm8: np.ndarray, ori8: np.ndarray) -> bool:
        """Seed states are those with first layer solved."""
        return self.is_layer_solved(perm8, ori8)
    
    def solve_from_state(self, perm8: np.ndarray, ori8: np.ndarray) -> Dict:
        """
        Solve cube using Layer-By-Layer method.
        
        Returns:
            Dictionary with 'success', 'moves', and optional 'error'
        """
        # Step 1: Normalize to D (Put the solved layer on the bottom)
        p_norm, o_norm = self.normalize_to_d(perm8, ori8)
        if p_norm is None: return {'success': False, 'error': 'Normalize Fail'}
        
        # Verify layer is actually solved (adjacent colors match)
        if not self.is_layer_solved(p_norm, o_norm):
            return {'success': False, 'error': 'Layer not solved'}
        
        # Step 2: Solve OLL (Orient Last Layer)
        #
        p_oll, o_oll, pre_auf, oll_moves, pre_auf_str, oll_str = self._solve_oll(p_norm, o_norm)
        if oll_moves == -1: return {'success': False, 'error': 'OLL Fail'}

        # Step 3: Solve PLL (Permute Last Layer)
        #
        mid_auf, pll_moves, post_auf, mid_auf_str, pll_str, post_auf_str = self._solve_pll(p_oll, o_oll)
        if pll_moves == -1: return {'success': False, 'error': 'PLL Fail'}

        naive_total = pre_auf + oll_moves + mid_auf + pll_moves + post_auf
        reduced_total = self._solve_cost([pre_auf_str, oll_str, mid_auf_str, pll_str, post_auf_str])

        def _reduced_from(p, o):
            p_o, o_o, pre_, oll_, pre_str_, oll_str_ = self._solve_oll(p, o)
            if oll_ == -1: return None
            mid_, pll_, post_, mid_str_, pll_str_, post_str_ = self._solve_pll(p_o, o_o)
            if pll_ == -1: return None
            return self._solve_cost([pre_str_, oll_str_, mid_str_, pll_str_, post_str_])

        reduced_total = self._best_with_layer_rotation(p_norm, o_norm, reduced_total, _reduced_from)

        return {
            'success': True,
            'moves': {
                'layer': 0,
                'pre_auf': pre_auf,
                'oll': oll_moves,
                'mid_auf': mid_auf,
                'pll': pll_moves,
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

    def _get_post_auf_cost(self, perm8: np.ndarray, ori8: np.ndarray) -> Tuple[int, str]:
        """
        Helper to find how many U moves it takes to align the top layer with the bottom layer.
        """
        for post_str, compiled_auf in zip(self.AUF_MOVES, self.compiled_auf):
            p1, o1, cost = self._apply_compiled(perm8, ori8, compiled_auf)
            if self.is_solved_state(p1, o1):
                return cost, post_str
        return -1, ""

    def _solve_pll(self, perm8: np.ndarray, ori8: np.ndarray) -> Tuple[int, int, int, str, str, str]:
        """
        Solve PLL and extract Mid-AUF and Post-AUF costs.

        Returns:
            (mid_auf_cost, pll_cost, post_auf_cost, mid_auf_str, pll_alg_str, post_auf_str)
        """
        # 1. Check for PLL Skip (Just needs Post-AUF alignment)
        skip_cost, skip_str = self._get_post_auf_cost(perm8, ori8)
        if skip_cost != -1:
            return 0, 0, skip_cost, "", "", skip_str

        # 2. Try Mid-AUF + algorithm combinations
        for mid_str, mid_compiled in zip(self.AUF_MOVES, self.compiled_auf):
            p_mid, o_mid, mid_cost = self._apply_compiled(perm8, ori8, mid_compiled)

            for name, compiled in self.compiled_algorithms.get('pll', {}).items():
                p_alg, o_alg, alg_cost = self._apply_compiled(p_mid, o_mid, compiled)

                # Check if it just needs a final U turn to be completely solved
                post_cost, post_str = self._get_post_auf_cost(p_alg, o_alg)
                if post_cost != -1:
                    return mid_cost, alg_cost, post_cost, mid_str, self.algorithms['pll'][name], post_str

        return -1, -1, -1, "", "", ""