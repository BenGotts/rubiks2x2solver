"""
EG (Erik Gunnar) Method Solver for 2x2x2 Rubik's Cube.

Steps:
1. Solve first face (intuitive)
2. EG algorithm - finish based on bottom layer state:
   - EG-0: Layer solved (pure CLL)
   - EG-1: Adjacent swap on bottom
   - EG-2: Diagonal swap on bottom
"""

import numpy as np
from typing import Dict, Optional
from solver import Solver


class EGSolver(Solver):
    """Solver implementing the EG (Erik Gunnar) method."""
    
    def is_seed_state(self, perm8: np.ndarray, ori8: np.ndarray) -> bool:
        """Seed states are those with one face solved."""
        return self.is_face_solved(perm8, ori8)
    
    def solve_from_state(self, perm8: np.ndarray, ori8: np.ndarray) -> Dict:
        """
        Solve cube using EG method.
        
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
        
        # Step 2: Determine layer state and solve EG
        eg_moves = self._solve_eg(p_norm, o_norm)
        if eg_moves is None:
            return {
                'success': False,
                'error': 'EG Fail',
                'moves': {}
            }
        
        return {
            'success': True,
            'moves': {
                'face': 0,  # Already at face-solved state
                'eg': eg_moves
            }
        }
    
    def _solve_eg(self, perm8: np.ndarray, ori8: np.ndarray) -> Optional[int]:
        """
        Solve using EG algorithm based on layer state.
        
        Returns:
            total_move_count or None on failure
        """
        # Try pre-AUF + algorithm + post-AUF
        for pre_auf in ["", "U", "U'", "U2"]:
            p1, o1 = perm8.copy(), ori8.copy()
            pre_cost = 0
            
            if pre_auf:
                p1, o1, pre_cost = self._apply_algorithm(p1, o1, pre_auf)
            
            # Check if already solved
            if self.is_solved_state(p1, o1):
                return pre_cost
            
            # Try rotations + EG algorithms
            for rot in ["", "y", "y'", "y2"]:
                p2, o2 = p1.copy(), o1.copy()
                if rot:
                    p2, o2 = self._apply_rotation(p2, o2, rot)
                
                for _, alg in self.algorithms['eg'].items():
                    p3, o3, alg_cost = self._apply_algorithm(p2, o2, alg)
                    
                    for post_auf in ["", "U", "U'", "U2"]:
                        p4, o4 = p3.copy(), o3.copy()
                        post_cost = 0
                        
                        if post_auf:
                            p4, o4, post_cost = self._apply_algorithm(p4, o4, post_auf)

                        if self.is_solved_state(p4, o4):
                            return pre_cost + alg_cost + post_cost
        
        return None