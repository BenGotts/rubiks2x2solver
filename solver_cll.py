"""
CLL Method Solver for 2x2x2 Rubik's Cube.

Steps:
1. Solve first layer completely (intuitive)
2. Solve Corners of Last Layer (CLL) in one algorithm
"""

import numpy as np
from typing import Dict
from solver import Solver


class CLLSolver(Solver):
    """Solver implementing the CLL (Corners of Last Layer) method."""
    
    def is_seed_state(self, perm8: np.ndarray, ori8: np.ndarray) -> bool:
        """Seed states are those with first layer solved."""
        return self.is_layer_solved(perm8, ori8)
    
    def solve_from_state(self, perm8: np.ndarray, ori8: np.ndarray) -> Dict:
        """
        Solve cube using CLL method.
        
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
        
        # Verify layer is actually solved
        if not self.is_layer_solved(perm8, ori8):
            return {
                'success': False,
                'error': 'Layer not solved',
                'moves': {}
            }
        
        # Step 2: Solve CLL
        cll_moves = self._solve_cll(p_norm, o_norm)
        if cll_moves is None:
            return {
                'success': False,
                'error': 'CLL Fail',
                'moves': {}
            }
        
        return {
            'success': True,
            'moves': {
                'layer': 0,  # Already at layer-solved state
                'cll': cll_moves
            }
        }
    
    def _solve_cll(self, perm8: np.ndarray, ori8: np.ndarray) -> int:
        """
        Solve CLL (corners of last layer).
        
        Returns:
            total_move_count or None on failure
        """
        for pre_auf in ["", "U", "U'", "U2"]:
            p1, o1 = perm8.copy(), ori8.copy()
            pre_cost = 0
            
            if pre_auf:
                p1, o1, pre_cost = self._apply_algorithm(p1, o1, pre_auf)
            
            # Check if already solved
            if self.is_solved_state(p1, o1):
                return pre_cost
            
            # Try each CLL algorithm
            for alg_name, alg in self.algorithms['cll'].items():
                p2, o2, alg_cost = self._apply_algorithm(p1, o1, alg)
                
                # Try post-AUF
                for post_auf in ["", "U", "U'", "U2"]:
                    p3, o3 = p2.copy(), o2.copy()
                    post_cost = 0
                    
                    if post_auf:
                        p3, o3, post_cost = self._apply_algorithm(p3, o3, post_auf)
                    
                    if self.is_solved_state(p3, o3):
                        return pre_cost + alg_cost + post_cost
        
        return None