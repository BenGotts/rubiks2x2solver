"""
EG (Erik Gunnar) Method Solver for 2x2x2 Rubik's Cube.

Steps:
1. Solve first face (intuitive)
2. Pre-AUF + EG algorithm + Post-AUF
   - EG-0: Layer solved (pure CLL)
   - EG-1: Adjacent swap on bottom
   - EG-2: Diagonal swap on bottom
"""

import numpy as np
from typing import Dict, Tuple
from solver import Solver


class EGSolver(Solver):
    """Solver implementing the EG (Erik Gunnar) method."""
    
    def is_seed_state(self, perm8: np.ndarray, ori8: np.ndarray) -> bool:
        """Seed states are those with one face solved."""
        return self.is_face_solved(perm8, ori8)
    
    def solve_from_state(self, perm8: np.ndarray, ori8: np.ndarray) -> Dict:
        # Step 1: Normalize to D (Put the solved face on the bottom)
        p_norm, o_norm = self.normalize_to_d(perm8, ori8)
        if p_norm is None: return {'success': False, 'error': 'Normalize Fail'}
        
        # Step 2: Solve EG (1-Look case)
        pre_auf, eg_moves, post_auf = self._solve_eg(p_norm, o_norm)
        if eg_moves == -1: return {'success': False, 'error': 'EG Fail'}
        
        return {
            'success': True,
            'moves': {
                'face': 0,        
                'pre_auf': pre_auf,
                'eg': eg_moves,
                'mid_auf': 0,     # EG is a 1-look method
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

    def _solve_eg(self, perm8: np.ndarray, ori8: np.ndarray) -> Tuple[int, int, int]:
        # 1. Check for EG Skip
        skip_cost = self._get_post_auf_cost(perm8, ori8)
        if skip_cost != -1:
            return 0, 0, skip_cost
            
        # 2. Try Pre-AUF + Rotation (y) + Algorithm + Post-AUF (U)
        for pre_auf in self.AUF_MOVES:
            p_pre, o_pre, pre_cost = self._apply_algorithm(perm8, ori8, pre_auf)
            
            # Apply zero-cost Y rotations to align the algorithm
            for rot in ["", "y", "y'", "y2"]:
                if rot:
                    p_rot, o_rot = self._apply_rotation(p_pre, o_pre, rot)
                else:
                    p_rot, o_rot = p_pre, o_pre
                
                for _, alg in self.algorithms.get('eg', {}).items():
                    p_alg, o_alg, alg_cost = self._apply_algorithm(p_rot, o_rot, alg)
                    
                    # Check if the algorithm solved the cube
                    post_cost = self._get_post_auf_cost(p_alg, o_alg)
                    if post_cost != -1:
                        return pre_cost, alg_cost, post_cost
                        
        return -1, -1, -1