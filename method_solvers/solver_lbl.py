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
        p_oll, o_oll, pre_auf, oll_moves = self._solve_oll(p_norm, o_norm)
        if oll_moves == -1: return {'success': False, 'error': 'OLL Fail'}
        
        # Step 3: Solve PLL (Permute Last Layer)
        # 
        mid_auf, pll_moves, post_auf = self._solve_pll(p_oll, o_oll)
        if pll_moves == -1: return {'success': False, 'error': 'PLL Fail'}
        
        return {
            'success': True,
            'moves': {
                'layer': 0,
                'pre_auf': pre_auf,
                'oll': oll_moves,
                'mid_auf': mid_auf,
                'pll': pll_moves,
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
        if len(set(stickers[0:4])) == 1:
            return perm8, ori8, 0, 0
        
        # Try Pre-AUF + algorithm combinations
        for auf in self.AUF_MOVES:
            p_auf, o_auf, auf_cost = self._apply_algorithm(perm8, ori8, auf)
            
            for _, alg in self.algorithms.get('oll', {}).items():
                p_res, o_res, alg_cost = self._apply_algorithm(p_auf, o_auf, alg)
                s = self.get_stickers8(p_res, o_res)
                
                # Check if the top face is a single solid color
                if len(set(s[0:4])) == 1:
                    return p_res, o_res, auf_cost, alg_cost
                    
        return perm8, ori8, -1, -1

    def _get_post_auf_cost(self, perm8: np.ndarray, ori8: np.ndarray) -> int:
        """
        Helper to find how many U moves it takes to align the top layer with the bottom layer.
        """
        for u_auf in self.AUF_MOVES:
            p1, o1, cost = self._apply_algorithm(perm8, ori8, u_auf)
            if self.is_solved_state(p1, o1):
                return cost
        return -1

    def _solve_pll(self, perm8: np.ndarray, ori8: np.ndarray) -> Tuple[int, int, int]:
        """
        Solve PLL and extract Mid-AUF and Post-AUF costs.
        
        Returns:
            (mid_auf_cost, pll_cost, post_auf_cost)
        """
        # 1. Check for PLL Skip (Just needs Post-AUF alignment)
        skip_cost = self._get_post_auf_cost(perm8, ori8)
        if skip_cost != -1:
            return 0, 0, skip_cost
            
        # 2. Try Mid-AUF + algorithm combinations
        for mid_auf in self.AUF_MOVES:
            p_mid, o_mid, mid_cost = self._apply_algorithm(perm8, ori8, mid_auf)
            
            for _, alg in self.algorithms.get('pll', {}).items():
                p_alg, o_alg, alg_cost = self._apply_algorithm(p_mid, o_mid, alg)
                
                # Check if it just needs a final U turn to be completely solved
                post_cost = self._get_post_auf_cost(p_alg, o_alg)
                if post_cost != -1:
                    return mid_cost, alg_cost, post_cost
                    
        return -1, -1, -1