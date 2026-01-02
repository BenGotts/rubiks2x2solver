"""
Ortega Method Solver for 2x2x2 Rubik's Cube.

Steps:
1. Solve any face (intuitive)
2. Orient Last Layer (OLL)
3. Permute Both Layers (PBL)
"""

import numpy as np
from typing import Dict, Tuple
from solver import Solver


class OrtegaSolver(Solver):
    """Solver implementing the Ortega method."""
    
    def is_seed_state(self, perm8: np.ndarray, ori8: np.ndarray) -> bool:
        """Seed states are those with one face solved."""
        return self.is_face_solved(perm8, ori8)
    
    def solve_from_state(self, perm8: np.ndarray, ori8: np.ndarray) -> Dict:
        """
        Solve cube using Ortega method.
        
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
        
        # Step 2: Solve OLL
        p_oll, o_oll, oll_moves = self._solve_oll(p_norm, o_norm)
        if oll_moves == -1:
            return {
                'success': False,
                'error': 'OLL Fail',
                'moves': {}
            }
        
        # Step 3: Solve PBL
        pbl_moves = self._solve_pbl(p_oll, o_oll)
        if pbl_moves is None:
            return {
                'success': False,
                'error': 'PBL Fail',
                'moves': {}
            }
        
        return {
            'success': True,
            'moves': {
                'face': 0,  # Already at face-solved state
                'oll': oll_moves,
                'pbl': pbl_moves
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
    
    def _solve_pbl(self, perm8: np.ndarray, ori8: np.ndarray) -> int:
        """
        Solve PBL (permute both layers).
        
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
            
            # Try rotations + algorithms
            for rot in ["", "y", "y'", "y2"]:
                p2, o2 = p1.copy(), o1.copy()
                if rot:
                    p2, o2 = self._apply_rotation(p2, o2, rot)
                
                for _, alg in self.algorithms['pbl'].items():
                    p3, o3, alg_cost = self._apply_algorithm(p2, o2, alg)
                    
                    for post_auf in ["", "U", "U'", "U2"]:
                        p4, o4 = p3.copy(), o3.copy()
                        post_cost = 0
                        
                        if post_auf:
                            p4, o4, post_cost = self._apply_algorithm(p4, o4, post_auf)

                        if self.is_solved_state(p4, o4):
                            return pre_cost + alg_cost + post_cost
        
        return None