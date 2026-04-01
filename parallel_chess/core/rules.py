import numpy as np
from .board import EMPTY, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING

# Precomputed knight attack offsets
_KNIGHT_DELTAS = np.array([(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)], dtype=np.int8)

# Ray directions for sliding pieces: rook, bishop, queen
_ROOK_DIRS   = [(0,1),(0,-1),(1,0),(-1,0)]
_BISHOP_DIRS = [(1,1),(1,-1),(-1,1),(-1,-1)]
_QUEEN_DIRS  = _ROOK_DIRS + _BISHOP_DIRS


def get_pseudo_legal_moves(board: np.ndarray, color: int) -> np.ndarray:
    """
    Returns boolean mask (8,8,8,8): [from_rank, from_file, to_rank, to_file].
    Pseudo-legal: physically valid per piece movement rules, no check detection.
    color: +1 for white, -1 for black.
    """
    moves = np.zeros((8, 8, 8, 8), dtype=bool)
    own_pieces  = (np.sign(board) == color)
    opp_pieces  = (np.sign(board) == -color)
    occupied    = board != EMPTY

    from_ranks, from_files = np.where(own_pieces)

    for fr, ff in zip(from_ranks, from_files):
        piece = int(board[fr, ff])
        abs_piece = abs(piece)

        if abs_piece == PAWN:
            _pawn_moves(moves, board, fr, ff, color, opp_pieces, occupied)
        elif abs_piece == KNIGHT:
            _knight_moves(moves, fr, ff, own_pieces)
        elif abs_piece == BISHOP:
            _sliding_moves(moves, board, fr, ff, _BISHOP_DIRS, own_pieces, occupied)
        elif abs_piece == ROOK:
            _sliding_moves(moves, board, fr, ff, _ROOK_DIRS, own_pieces, occupied)
        elif abs_piece == QUEEN:
            _sliding_moves(moves, board, fr, ff, _QUEEN_DIRS, own_pieces, occupied)
        elif abs_piece == KING:
            _king_moves(moves, fr, ff, own_pieces)

    return moves


def _pawn_moves(moves, board, fr, ff, color, opp_pieces, occupied):
    direction = -color  # white moves up (rank decreases), black down
    start_rank = 6 if color == 1 else 1

    # Single push
    tr = fr + direction
    if 0 <= tr < 8 and not occupied[tr, ff]:
        moves[fr, ff, tr, ff] = True
        # Double push from starting rank
        tr2 = fr + 2 * direction
        if fr == start_rank and not occupied[tr2, ff]:
            moves[fr, ff, tr2, ff] = True

    # Captures
    for df in (-1, 1):
        tf = ff + df
        if 0 <= tr < 8 and 0 <= tf < 8 and opp_pieces[tr, tf]:
            moves[fr, ff, tr, tf] = True


def _knight_moves(moves, fr, ff, own_pieces):
    targets = _KNIGHT_DELTAS + np.array([fr, ff])
    valid = (targets[:, 0] >= 0) & (targets[:, 0] < 8) & \
            (targets[:, 1] >= 0) & (targets[:, 1] < 8)
    for tr, tf in targets[valid]:
        if not own_pieces[tr, tf]:
            moves[fr, ff, tr, tf] = True


def _sliding_moves(moves, board, fr, ff, directions, own_pieces, occupied):
    for dr, df in directions:
        tr, tf = fr + dr, ff + df
        while 0 <= tr < 8 and 0 <= tf < 8:
            if own_pieces[tr, tf]:
                break
            moves[fr, ff, tr, tf] = True
            if occupied[tr, tf]:  # hit opponent piece, stop ray
                break
            tr += dr
            tf += df


def _king_moves(moves, fr, ff, own_pieces):
    for dr in (-1, 0, 1):
        for df in (-1, 0, 1):
            if dr == 0 and df == 0:
                continue
            tr, tf = fr + dr, ff + df
            if 0 <= tr < 8 and 0 <= tf < 8 and not own_pieces[tr, tf]:
                moves[fr, ff, tr, tf] = True


def moves_mask_to_list(mask: np.ndarray) -> list[tuple]:
    """Convert (8,8,8,8) mask to list of (fr, ff, tr, tf) tuples."""
    indices = np.argwhere(mask)
    return [tuple(idx) for idx in indices]


def encode_move(fr: int, ff: int, tr: int, tf: int) -> tuple[int, int]:
    """Encode move as (from_sq, to_sq) where sq = rank*8 + file."""
    return fr * 8 + ff, tr * 8 + tf


def decode_move(from_sq: int, to_sq: int) -> tuple[int, int, int, int]:
    return from_sq // 8, from_sq % 8, to_sq // 8, to_sq % 8
