"""
LBL (Layer-By-Layer / Beginner) Method Solver for 2x2x2 Rubik's Cube.

Steps:
1. Solve first layer completely (intuitive)
2. Orient Last Layer (OLL)
3. Permute Last Layer (PLL)
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
        # Step 1: Normalize to D
        p_norm, o_norm = self.normalize_to_d(perm8, ori8)
        if p_norm is None:
            return {
                'success': False,
                'error': 'Normalize Fail',
                'moves': {}
            }
        
        # Verify layer is solved
        if not self.is_layer_solved(perm8, ori8):
            return {
                'success': False,
                'error': 'Layer not solved',
                'moves': {}
            }
        
        # Step 2: Solve OLL
        p_oll, o_oll, oll_moves = self._solve_oll(p_norm, o_norm)
        if oll_moves == -1:
            return {
                'success': False,
                'error': 'OLL Fail',
                'moves': {}
            }
        
        # Step 3: Solve PLL
        pll_moves = self._solve_pll(p_oll, o_oll)
        if pll_moves is None:
            return {
                'success': False,
                'error': 'PLL Fail',
                'moves': {}
            }
        
        return {
            'success': True,
            'moves': {
                'layer': 0,  # Already at layer-solved state
                'oll': oll_moves,
                'pll': pll_moves
            }
        }
    
    def _solve_oll(self, perm8: np.ndarray, ori8: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Solve OLL (orient last layer).
        
        Returns:
            (new_perm8, new_ori8, move_count) or (perm8, ori8, -1) on failure
        """
        stickers = self.get_stickers8(perm8, ori8)
        
        # Check if already oriented
        if len(set(stickers[0:4])) == 1:
            return perm8, ori8, 0
        
        # Try AUF + algorithm combinations
        for auf in ["", "U", "U'", "U2"]:
            p_auf, o_auf = perm8.copy(), ori8.copy()
            auf_cost = 0
            
            if auf:
                p_auf, o_auf, auf_cost = self._apply_algorithm(p_auf, o_auf, auf)
            
            for _, alg in self.algorithms['oll'].items():
                p_res, o_res, alg_cost = self._apply_algorithm(p_auf, o_auf, alg)
                s = self.get_stickers8(p_res, o_res)
                
                if len(set(s[0:4])) == 1:
                    return p_res, o_res, alg_cost + auf_cost
        
        return perm8, ori8, -1
    
    def _solve_pll(self, perm8: np.ndarray, ori8: np.ndarray) -> int:
        """
        Solve PLL (permute last layer).
        
        Returns:
            Total move count or None on failure
        """
        # Try pre-AUF + algorithm + post-AUF
        for pre_auf in ["", "U", "U'", "U2"]:
            p_pre, o_pre = perm8.copy(), ori8.copy()
            pre_cost = 0
            
            if pre_auf:
                p_pre, o_pre, pre_cost = self._apply_algorithm(p_pre, o_pre, pre_auf)
            
            # Check if already solved
            if self.is_solved_state(p_pre, o_pre):
                return pre_cost
            
            for _, alg in self.algorithms['pll'].items():
                p_alg, o_alg, alg_cost = self._apply_algorithm(p_pre, o_pre, alg)
                
                for post_auf in ["", "U", "U'", "U2"]:
                    p_post, o_post = p_alg.copy(), o_alg.copy()
                    post_cost = 0
                    
                    if post_auf:
                        p_post, o_post, post_cost = self._apply_algorithm(p_post, o_post, post_auf)

                    if self.is_solved_state(p_post, o_post):
                        return pre_cost + alg_cost + post_cost
        
        return None