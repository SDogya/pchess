import numpy as np

EMPTY  = 0
PAWN   = 1
KNIGHT = 2
BISHOP = 3
ROOK   = 4
QUEEN  = 5
KING   = 6

PIECE_SYMBOLS = {
    EMPTY: ".",
    PAWN: "P", KNIGHT: "N", BISHOP: "B", ROOK: "R", QUEEN: "Q", KING: "K",
    -PAWN: "p", -KNIGHT: "n", -BISHOP: "b", -ROOK: "r", -QUEEN: "q", -KING: "k",
}

FEN_TO_PIECE = {
    "P": PAWN, "N": KNIGHT, "B": BISHOP, "R": ROOK, "Q": QUEEN, "K": KING,
    "p": -PAWN, "n": -KNIGHT, "b": -BISHOP, "r": -ROOK, "q": -QUEEN, "k": -KING,
}

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


class BoardState:
    def __init__(self, fen: str = None):
        self.grid = np.zeros((8, 8), dtype=np.int8)
        self._parse_fen(fen or STARTING_FEN)

    def _parse_fen(self, fen: str):
        rows = fen.split("/")
        for rank_idx, row in enumerate(rows):
            file_idx = 0
            for ch in row:
                if ch.isdigit():
                    file_idx += int(ch)
                else:
                    self.grid[rank_idx, file_idx] = FEN_TO_PIECE[ch]
                    file_idx += 1

    def copy(self) -> "BoardState":
        new = BoardState.__new__(BoardState)
        new.grid = self.grid.copy()
        return new

    def __str__(self) -> str:
        lines = []
        for rank in range(8):
            lines.append(" ".join(PIECE_SYMBOLS.get(int(v), "?") for v in self.grid[rank]))
        return "\n".join(lines)
