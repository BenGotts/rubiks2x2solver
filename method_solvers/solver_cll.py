"""
CLL (Corners Last Layer) Method Solver for 2x2x2 Rubik's Cube.

Steps:
1. Solve first layer completely (intuitive)
2. Pre-AUF + CLL algorithm + Post-AUF
"""

import numpy as np
from typing import Dict, Tuple
from solver import Solver


class CLLSolver(Solver):
    """Solver implementing the CLL method."""
    
    def is_seed_state(self, perm8: np.ndarray, ori8: np.ndarray) -> bool:
        """Seed states are those with the first layer completely solved."""
        return self.is_layer_solved(perm8, ori8)
    
    def solve_from_state(self, perm8: np.ndarray, ori8: np.ndarray) -> Dict:
        # Step 1: Normalize to D (Put the solved layer on the bottom)
        p_norm, o_norm = self.normalize_to_d(perm8, ori8)
        if p_norm is None: return {'success': False, 'error': 'Normalize Fail'}
            
        # Verify the layer is fully solved (adjacent side colors match)
        if not self.is_layer_solved(p_norm, o_norm):
            return {'success': False, 'error': 'Layer not solved'}
        
        # Step 2: Solve CLL (1-Look Last Layer)
        pre_auf, cll_moves, post_auf = self._solve_cll(p_norm, o_norm)
        if cll_moves == -1: return {'success': False, 'error': 'CLL Fail'}
        
        return {
            'success': True,
            'moves': {
                'layer': 0,
                'pre_auf': pre_auf,
                'cll': cll_moves,
                'mid_auf': 0,     # CLL is a 1-look method
                'post_auf': post_auf
            }
        }
    
    def _get_post_auf_cost(self, perm8: np.ndarray, ori8: np.ndarray) -> int:
        """Helper to find how many U moves it takes to align the solved cube."""
        for u_auf in self.AUF_MOVES:
            p1, o1, cost = self._apply_algorithm(perm8, ori8, u_auf)
            if self.is_solved_state(p1, o1):
                return cost
        return -1

    def _solve_cll(self, perm8: np.ndarray, ori8: np.ndarray) -> Tuple[int, int, int]:
        """
        Solve CLL and extract Pre-AUF and Post-AUF costs.
        
        Returns:
            (pre_auf_cost, cll_alg_cost, post_auf_cost)
        """
        # 1. Check for CLL Skip
        skip_cost = self._get_post_auf_cost(perm8, ori8)
        if skip_cost != -1:
            return 0, 0, skip_cost
            
        # 2. Try Pre-AUF (U) + Algorithm + Post-AUF (U)
        for pre_auf in self.AUF_MOVES:
            p_pre, o_pre, pre_cost = self._apply_algorithm(perm8, ori8, pre_auf)
            
            for _, alg in self.algorithms.get('cll', {}).items():
                p_alg, o_alg, alg_cost = self._apply_algorithm(p_pre, o_pre, alg)
                
                # Check if the algorithm solved the cube
                post_cost = self._get_post_auf_cost(p_alg, o_alg)
                if post_cost != -1:
                    return pre_cost, alg_cost, post_cost
                        
        return -1, -1, -1