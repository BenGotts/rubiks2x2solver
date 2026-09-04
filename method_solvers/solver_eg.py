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

    # Shared with any other method whose seed states are "one face solved" (see OrtegaSolver).
    SEED_CRITERION = 'face'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Every (pre_auf, rotation, algorithm, post_auf) combination _solve_eg used to search
        # is a fixed transform independent of any state or color, so the exact state each one
        # solves can be computed once (by inverting the composed transform and applying that
        # inverse to the solved state) instead of re-searching all ~8,000 combinations per
        # seed state. Solving becomes a single dict lookup.
        self._eg_case_table = self._build_eg_case_table()

    def _build_eg_case_table(self) -> Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], Tuple[int, int, int]]:
        identity = (np.arange(8), np.zeros(8, dtype=int), 0)

        # is_solved_state accepts any of the 24 whole-cube-rotation images of the identity as
        # "solved" (see Solver._solved_orbit), not just the identity itself, so a combo can
        # land on any of them and still count as a successful post-AUF check -- every one of
        # them must be tried as a target when inverting back to find which state a combo solves.
        solved_states = self._solved_orbit()

        # Tried first in every (pre_auf, rotation) combo, so it reproduces the old dedicated
        # "EG skip" check (no algorithm needed, just AUF alignment) via pre="" + rotation=""
        # + this + post-loop -- the same priority order the old code used before ever trying
        # a real algorithm.
        algs = [identity] + list(self.compiled_algorithms.get('eg', {}).values())

        table: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], Tuple[int, int, int]] = {}
        for pre in self.compiled_auf:
            for rot_name in self.Y_ROTS:
                if rot_name:
                    rot_p, rot_t = self.ROTATIONS[rot_name]
                    rotation = (rot_p, rot_t, 0)
                else:
                    rotation = identity
                pre_rot = self._compose_compiled(pre, rotation)

                for alg in algs:
                    pre_rot_alg = self._compose_compiled(pre_rot, alg)

                    for post in self.compiled_auf:
                        full = self._compose_compiled(pre_rot_alg, post)
                        inv = self._invert_compiled(full)
                        for solved_p, solved_o in solved_states:
                            seed_p, seed_o, _ = self._apply_compiled(solved_p, solved_o, inv)
                            key = (tuple(seed_p.tolist()), tuple(seed_o.tolist()))
                            if key not in table:
                                table[key] = (pre[2], alg[2], post[2])
        return table

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
    
    def _solve_eg(self, perm8: np.ndarray, ori8: np.ndarray) -> Tuple[int, int, int]:
        key = (tuple(perm8.tolist()), tuple(ori8.tolist()))
        result = self._eg_case_table.get(key)
        return result if result is not None else (-1, -1, -1)