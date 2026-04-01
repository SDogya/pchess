import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
from matplotlib.figure import Figure
from matplotlib.axes import Axes

PIECE_UNICODE = {
    1: "♙", 2: "♘", 3: "♗", 4: "♖", 5: "♕", 6: "♔",
    -1: "♟", -2: "♞", -3: "♝", -4: "♜", -5: "♛", -6: "♚",
}

LIGHT_SQUARE = "#F0D9B5"
DARK_SQUARE  = "#B58863"
HIGHLIGHT    = "#FF6B35"
WHITE_PIECE  = "#FAFAFA"
BLACK_PIECE  = "#1A1A2E"


class BoardRenderer:
    _fig: Figure = None
    _ax: Axes = None

    @classmethod
    def render_frame(cls, board: np.ndarray, info: dict, block: bool = False):
        if cls._fig is None:
            cls._fig, cls._ax = plt.subplots(figsize=(7, 7))
            cls._fig.patch.set_facecolor("#16213E")
            plt.ion()

        ax = cls._ax
        ax.clear()
        ax.set_facecolor("#16213E")
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 8)
        ax.set_aspect("equal")
        ax.axis("off")

        collision_squares = set()
        if info.get("mutual_destruction") or info.get("swap_collision"):
            collision_squares = {(r, f) for r in range(8) for f in range(8)}  # placeholder

        for rank in range(8):
            for file in range(8):
                light = (rank + file) % 2 == 0
                color = LIGHT_SQUARE if light else DARK_SQUARE

                display_rank = 7 - rank
                rect = patches.Rectangle(
                    (file, display_rank), 1, 1,
                    linewidth=0, facecolor=color
                )
                ax.add_patch(rect)

                piece = int(board[rank, file])
                if piece != 0:
                    symbol = PIECE_UNICODE.get(piece, "?")
                    piece_color = WHITE_PIECE if piece > 0 else BLACK_PIECE
                    stroke_color = BLACK_PIECE if piece > 0 else WHITE_PIECE
                    ax.text(
                        file + 0.5, display_rank + 0.5, symbol,
                        ha="center", va="center",
                        fontsize=32, color=piece_color,
                        path_effects=[pe.withStroke(linewidth=2, foreground=stroke_color)]
                    )

        # Rank and file labels
        for i in range(8):
            ax.text(-0.3, i + 0.5, str(i + 1), ha="center", va="center",
                    fontsize=10, color="#AAA")
            ax.text(i + 0.5, -0.3, "abcdefgh"[i], ha="center", va="center",
                    fontsize=10, color="#AAA")

        title_parts = []
        if info.get("mutual_destruction"):
            title_parts.append("💥 Mutual Destruction!")
        if info.get("swap_collision"):
            title_parts.append("⚡ Swap Collision!")
        if info.get("white_king_dead"):
            title_parts.append("Black wins! ♔ fallen")
        if info.get("black_king_dead"):
            title_parts.append("White wins! ♚ fallen")

        title = "  |  ".join(title_parts) if title_parts else "Parallel Chess"
        cls._fig.suptitle(title, color="#E8E8E8", fontsize=13, fontweight="bold")

        plt.tight_layout()
        plt.draw()
        plt.pause(0.5)


        if block:
            plt.ioff()
            plt.show()

    @classmethod
    def save_frame(cls, board: np.ndarray, info: dict, path: str):
        cls.render_frame(board, info)
        cls._fig.savefig(path, dpi=100, bbox_inches="tight")
        print(f"Saved frame to {path}")

    @classmethod
    def close(cls):
        if cls._fig is not None:
            plt.close(cls._fig)
            cls._fig = None
            cls._ax = None