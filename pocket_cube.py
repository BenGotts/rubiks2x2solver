"""
PocketCube: Core 2x2x2 Rubik's Cube representation and operations.

This module provides the fundamental cube state representation using a 7-corner encoding scheme.
"""

import math
import numpy as np
from typing import List, Tuple, Optional

WHITE = 'W'
YELLOW = 'Y'
GREEN = 'G'
BLUE = 'B'
ORANGE = 'O'
RED = 'R'

# Color sets for solving
WHITE_YELLOW = {WHITE, YELLOW}
GREEN_BLUE = {GREEN, BLUE}
ORANGE_RED = {ORANGE, RED}
COLOR_NEUTRAL = {WHITE, YELLOW, GREEN, BLUE, ORANGE, RED}

# State space size
N_STATES = math.factorial(7) * (3**6)

# Move set options
# Full move set: All 18 moves (R, R2, R', U, U2, U', F, F2, F', D, D2, D', L, L2, L', B, B2, B')
# Reduced move set: Only R, U, B and their variants (R2, R', U2, U', B2, B')
# On a 2x2, R and L are functionally equivalent (no center to distinguish), same with F/B
# Using RUB is sufficient for generating all positions
# Works with the 7 corner encoding since the 7th corner does not move on R/U/B turns
REDUCED_MOVES = ['R', 'R2', "R'", 'U', 'U2', "U'", 'B', 'B2', "B'"]  # 9 moves instead of 18

# Clockwise moves
U_PERM = np.array([1,2,3,0, 4,5,6,7]); U_TW = np.zeros(8, dtype=int)
R_PERM = np.array([4,0,2,3, 5,1,6,7]); R_TW = np.array([2,1,0,0, 1,2,0,0])
F_PERM = np.array([3,1,2,7, 0,5,6,4]); F_TW = np.array([1,0,0,2, 2,0,0,1])
D_PERM = np.array([0,1,2,3, 7,4,5,6]); D_TW = np.zeros(8, dtype=int)
L_PERM = np.array([0,1,6,2, 4,5,7,3]); L_TW = np.array([0,0,2,1, 0,0,1,2])
B_PERM = np.array([0,5,1,3, 4,6,2,7]); B_TW = np.array([0,2,1,0, 0,1,2,0])

# Counter-clockwise moves
Ui_PERM = np.array([3,0,1,2, 4,5,6,7]); Ui_TW = np.zeros(8, dtype=int)
Ri_PERM = np.array([1,5,2,3, 0,4,6,7]); Ri_TW = np.array([2,1,0,0, 1,2,0,0])
Fi_PERM = np.array([4,1,2,0, 7,5,6,3]); Fi_TW = np.array([1,0,0,2, 2,0,0,1])
Di_PERM = np.array([0,1,2,3, 5,6,7,4]); Di_TW = np.zeros(8, dtype=int)
Li_PERM = np.array([0,1,3,7, 4,5,2,6]); Li_TW = np.array([0,0,2,1, 0,0,1,2])
Bi_PERM = np.array([0,2,6,3, 4,1,5,7]); Bi_TW = np.array([0,2,1,0, 0,1,2,0])

# Double moves
U2_PERM = np.array([2,3,0,1, 4,5,6,7]); U2_TW = np.zeros(8, dtype=int)
R2_PERM = np.array([5,4,2,3, 1,0,6,7]); R2_TW = np.zeros(8, dtype=int)
F2_PERM = np.array([7,1,2,4, 3,5,6,0]); F2_TW = np.zeros(8, dtype=int)
D2_PERM = np.array([0,1,2,3, 6,7,4,5]); D2_TW = np.zeros(8, dtype=int)
L2_PERM = np.array([0,1,7,6, 4,5,3,2]); L2_TW = np.zeros(8, dtype=int)
B2_PERM = np.array([0,6,5,3, 4,2,1,7]); B2_TW = np.zeros(8, dtype=int)

# Rotation moves
X_PERM = np.array([4,0,3,7, 5,1,2,6]); X_TW = np.array([2,1,2,1, 1,2,1,2])
Y_PERM = np.array([1,2,3,0, 5,6,7,4]); Y_TW = np.zeros(8, dtype=int)
Z_PERM = np.array([3,2,6,7, 0,1,5,4]); Z_TW = np.array([1,2,1,2, 2,1,2,1])

Xi_PERM = np.array([1,5,6,2, 0,4,7,3]); Xi_TW = np.array([2,1,2,1, 1,2,1,2])
Yi_PERM = np.array([3,0,1,2, 7,4,5,6]); Yi_TW = np.zeros(8, dtype=int)
Zi_PERM = np.array([4,5,1,0, 7,6,2,3]); Zi_TW = np.array([1,2,1,2, 2,1,2,1])

X2_PERM = np.array([5,4,7,6, 1,0,3,2]); X2_TW = np.zeros(8, dtype=int)
Y2_PERM = np.array([2,3,0,1, 6,7,4,5]); Y2_TW = np.zeros(8, dtype=int)
Z2_PERM = np.array([7,6,5,4, 3,2,1,0]); Z2_TW = np.zeros(8, dtype=int)

# Move dictionaries
MOVES = {
    'R': (R_PERM, R_TW), "R'": (Ri_PERM, Ri_TW), 'R2': (R2_PERM, R2_TW),
    'U': (U_PERM, U_TW), "U'": (Ui_PERM, Ui_TW), 'U2': (U2_PERM, U2_TW),
    'F': (F_PERM, F_TW), "F'": (Fi_PERM, Fi_TW), 'F2': (F2_PERM, F2_TW),
    'L': (L_PERM, L_TW), "L'": (Li_PERM, Li_TW), 'L2': (L2_PERM, L2_TW),
    'D': (D_PERM, D_TW), "D'": (Di_PERM, Di_TW), 'D2': (D2_PERM, D2_TW),
    'B': (B_PERM, B_TW), "B'": (Bi_PERM, Bi_TW), 'B2': (B2_PERM, B2_TW),
}

ROTATIONS = {
    'x': (X_PERM, X_TW), "x'": (Xi_PERM, Xi_TW), 'x2': (X2_PERM, X2_TW),
    'y': (Y_PERM, Y_TW), "y'": (Yi_PERM, Yi_TW), 'y2': (Y2_PERM, Y2_TW),
    'z': (Z_PERM, Z_TW), "z'": (Zi_PERM, Zi_TW), 'z2': (Z2_PERM, Z2_TW),
}

# Face colors in standard orientation
FACE_COLOR = {"U": "W", "R": "R", "F": "G", "D": "Y", "L": "O", "B": "B"}

# Corner piece positions (UFR, URB, UBL, ULF, DFR, DRB, DBL, DLF)
CORNER_TO_FACES = [
    ("U", "F", "R"),  # 0: UFR
    ("U", "R", "B"),  # 1: URB
    ("U", "B", "L"),  # 2: UBL
    ("U", "L", "F"),  # 3: ULF
    ("D", "R", "F"),  # 4: DFR
    ("D", "B", "R"),  # 5: DRB
    ("D", "L", "B"),  # 6: DBL
    ("D", "F", "L"),  # 7: DLF
]

# 2x2 facelet layout (piece_id, facelet_index)
FACELETS_2x2 = {
    "U": [(2, 0), (1, 0), (0, 0), (3, 0)],
    "R": [(0, 2), (1, 1), (5, 2), (4, 1)],
    "F": [(3, 2), (0, 1), (4, 2), (7, 1)],
    "D": [(7, 0), (4, 0), (5, 0), (6, 0)],
    "L": [(2, 2), (3, 1), (7, 2), (6, 1)],
    "B": [(1, 2), (2, 1), (6, 2), (5, 1)],
}

FACE_ORDER = ["U", "R", "F", "D", "L", "B"]


class PocketCube:
    """
    Represents a 2x2x2 Rubik's Cube (Pocket Cube) state.
    
    Uses a 7-corner encoding scheme where:
    - 7 corners are explicitly tracked (8th is implicit)
    - Each corner has a permutation and orientation (0, 1, or 2)
    
    Attributes:
        perm8 (np.ndarray): Current permutation of 8 corners
        ori8 (np.ndarray): Current orientation of 8 corners
    """
    
    def __init__(self, state_id: int = 0):
        """
        Initialize a PocketCube.
        
        Args:
            state_id: Starting state (0 = solved cube)
        """
        self.perm8, self.ori8 = self.lift_to_full8(*self.unpackcube(state_id))
        
        self.MOVES = MOVES
        self.ROTATIONS = ROTATIONS
    
    @staticmethod
    def rankperm(p: List[int]) -> int:
        """Convert permutation to integer rank."""
        p = np.array(p)
        q = np.array(p).argsort()
        r = 0
        for k in range(len(p) - 1, 0, -1):
            s = p[k]
            p[k], p[q[k]] = p[q[k]], p[k]
            q[k], q[s] = q[s], q[k]
            r += int(s) * math.factorial(k)
        return r

    @staticmethod
    def unrankperm(r: int, n: int) -> List[int]:
        """Convert integer rank to permutation."""
        p = list(range(n))
        for k in range(n - 1, 0, -1):
            s, r = divmod(r, math.factorial(k))
            p[k], p[s] = p[s], p[k]
        return p

    def packcube(self, cube: Tuple[np.ndarray, np.ndarray]) -> int:
        """Encode 7-corner state to integer ID."""
        p, q = cube
        return self.rankperm(p) * (3**6) + int(np.sum(q[:6] * (3 ** np.arange(5, -1, -1))))

    def unpackcube(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        """Decode integer ID to 7-corner state."""
        p, q = divmod(int(i), 3**6)
        q = np.array(list(map(ord, np.base_repr(q, 3, 6)[-6:]))) - ord('0')
        q7 = np.append(q, (-int(np.sum(q)) % 3))
        return (np.array(self.unrankperm(p, 7), dtype=int), q7)
    
    @staticmethod
    def lift_to_full8(p7: np.ndarray, q7: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert 7-corner representation to full 8-corner arrays."""
        perm8 = np.empty(8, dtype=int)
        ori8 = np.empty(8, dtype=int)
        perm8[:7] = p7
        perm8[7] = 7
        ori8[:7] = q7[:7] % 3
        ori8[7] = (-int(np.sum(ori8[:7])) % 3)
        return perm8, ori8

    @staticmethod
    def project_to_7(perm8: np.ndarray, ori8: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert 8-corner arrays back to 7-corner representation. Only works if corner 7 is in place."""
        pos7 = int(np.where(perm8 == 7)[0][0])
        keep_idx = [i for i in range(8) if i != pos7]
        p7 = perm8[keep_idx].astype(int)
        q7 = ori8[keep_idx].astype(int) % 3
        return p7, q7
    
    def set_state(self, state_id: int):
        """Set cube to a specific state ID."""
        self.perm8, self.ori8 = self.lift_to_full8(*self.unpackcube(state_id))
    
    def get_state_id(self) -> int:
        """Get current state as integer ID."""
        p7, q7 = self.project_to_7(self.perm8, self.ori8)
        return self.packcube((p7, q7))
        
    def apply_move(self, move: str):
        """Apply a face turn move to the cube."""
        if move not in self.MOVES:
            raise ValueError(f"Invalid move: {move}")
        perm_move, ori_move = self.MOVES[move]
        self.perm8 = self.perm8[perm_move]
        self.ori8 = (self.ori8[perm_move] + ori_move) % 3
        
    def apply_rotation(self, rotation: str):
        """Apply a cube rotation."""
        if rotation not in self.ROTATIONS:
            raise ValueError(f"Invalid rotation: {rotation}")
        perm_rot, ori_rot = self.ROTATIONS[rotation]
        self.perm8 = self.perm8[perm_rot]
        self.ori8 = (self.ori8[perm_rot] + ori_rot) % 3
        
    def apply_algorithm(self, algorithm: str) -> int:
        """
        Apply a sequence of moves/rotations.
        
        Args:
            algorithm: Space-separated move sequence
            
        Returns:
            Number of face turns (excludes rotations)
        """
        count = 0
        moves = algorithm.split()
        for move in moves:
            if move in self.MOVES:
                self.apply_move(move)
                count += 1
            elif move in self.ROTATIONS:
                self.apply_rotation(move)
            else:
                raise ValueError(f"Invalid move or rotation: {move}")
        return count
    
    def get_stickers(self) -> str:
        """Get 24-character sticker string representing visible colors."""
        return self.get_stickers8(self.perm8, self.ori8)
    
    @staticmethod
    def get_stickers8(perm8: np.ndarray, ori8: np.ndarray) -> str:
        """Static method to get stickers from perm/ori arrays."""
        out = []
        for face in FACE_ORDER:
            for (slot, which_face_idx) in FACELETS_2x2[face]:
                piece = perm8[slot]
                ori = ori8[slot] % 3
                face_tuple = CORNER_TO_FACES[piece]
                actual_face_idx = (which_face_idx + ori) % 3
                face_letter = face_tuple[actual_face_idx]
                out.append(FACE_COLOR[face_letter])
        return "".join(out)
    
    @staticmethod
    def pretty_print_cube(s: str):
        """Print ASCII representation of cube state."""
        print(f"     {s[0]} {s[1]}")
        print(f"     {s[3]} {s[2]}")
        print(f"{s[16]} {s[17]}  {s[8]} {s[9]}  {s[4]} {s[5]}  {s[20]} {s[21]}")
        print(f"{s[19]} {s[18]}  {s[11]} {s[10]}  {s[7]} {s[6]}  {s[23]} {s[22]}")
        print(f"     {s[12]} {s[13]}")
        print(f"     {s[15]} {s[14]}")
    
    def is_solved(self) -> bool:
        """Check if entire cube is solved."""
        stickers = self.get_stickers()
        return all(len(set(stickers[i:i+4])) == 1 for i in range(0, 24, 4))
    
    @staticmethod
    def is_solved_state(perm8: np.ndarray, ori8: np.ndarray) -> bool:
        """Static method to check if given perm/ori arrays represent a solved cube."""
        stickers = PocketCube.get_stickers8(perm8, ori8)
        return all(len(set(stickers[i:i+4])) == 1 for i in range(0, 24, 4))