"""
caro_environment.py
Core Environment cho trò chơi Caro 11x11.

Quy ước:
    EMPTY = 0: ô trống
    X = 1: quân X
    O = -1: quân O

X đi trước. Người chơi thắng khi có ít nhất 5 quân liên tiếp
theo hàng, cột hoặc một trong hai đường chéo.
Tọa độ (row, col) bắt đầu từ 0.
"""

from copy import deepcopy

BOARD_SIZE = 11
WIN_LENGTH = 5

EMPTY = 0
X = 1
O = -1

DIRECTIONS = (
    (0, 1),    # ngang
    (1, 0),    # dọc
    (1, 1),    # chéo chính
    (1, -1),   # chéo phụ
)


class CaroEnv:
    """Môi trường Caro đơn giản, không phụ thuộc thư viện bên ngoài."""

    def __init__(self, board_size=BOARD_SIZE, win_length=WIN_LENGTH):
        if not isinstance(board_size, int) or isinstance(board_size, bool):
            raise TypeError("board_size phải là số nguyên.")
        if not isinstance(win_length, int) or isinstance(win_length, bool):
            raise TypeError("win_length phải là số nguyên.")
        if win_length < 1 or board_size < win_length:
            raise ValueError("Cần board_size >= win_length >= 1.")

        self.board_size = board_size
        self.win_length = win_length
        self.board = []
        self.current_player = X
        self.winner = EMPTY
        self.done = False
        self.last_move = None
        self.reset()

    def reset(self):
        """Bắt đầu ván mới và trả về bản sao bàn cờ."""
        self.board = [
            [EMPTY for _ in range(self.board_size)]
            for _ in range(self.board_size)
        ]
        self.current_player = X
        self.winner = EMPTY
        self.done = False
        self.last_move = None
        return self.get_observation()

    def get_observation(self):
        """Trả về bản sao bàn cờ để bên ngoài không sửa trực tiếp trạng thái."""
        return deepcopy(self.board)

    def is_valid_move(self, row, col):
        """Kiểm tra tọa độ, ô trống và trạng thái ván."""
        valid_row = isinstance(row, int) and not isinstance(row, bool)
        valid_col = isinstance(col, int) and not isinstance(col, bool)

        return (
            not self.done
            and valid_row
            and valid_col
            and 0 <= row < self.board_size
            and 0 <= col < self.board_size
            and self.board[row][col] == EMPTY
        )

    def get_legal_moves(self):
        """Danh sách các nước đi hợp lệ theo dạng (row, col)."""
        if self.done:
            return []

        return [
            (row, col)
            for row in range(self.board_size)
            for col in range(self.board_size)
            if self.board[row][col] == EMPTY
        ]

    def _count_one_direction(self, row, col, dr, dc, player):
        """Đếm quân liên tiếp cùng màu bắt đầu từ ô kế bên theo một hướng."""
        count = 0
        r, c = row + dr, col + dc

        while (
            0 <= r < self.board_size
            and 0 <= c < self.board_size
            and self.board[r][c] == player
        ):
            count += 1
            r += dr
            c += dc

        return count

    def check_winner(self, row=None, col=None, player=None):
        """
        Trả về X hoặc O nếu có người thắng, ngược lại trả về EMPTY.

        Có thể truyền row, col, player để kiểm tra quanh nước vừa đánh.
        Nếu không truyền đủ tham số, hàm quét toàn bộ bàn cờ.
        """
        if (
            isinstance(row, int)
            and isinstance(col, int)
            and player in (X, O)
        ):
            for dr, dc in DIRECTIONS:
                total = (
                    1
                    + self._count_one_direction(row, col, dr, dc, player)
                    + self._count_one_direction(row, col, -dr, -dc, player)
                )
                if total >= self.win_length:
                    return player
            return EMPTY

        for r in range(self.board_size):
            for c in range(self.board_size):
                player_at_cell = self.board[r][c]
                if player_at_cell == EMPTY:
                    continue

                for dr, dc in DIRECTIONS:
                    end_r = r + (self.win_length - 1) * dr
                    end_c = c + (self.win_length - 1) * dc

                    if not (
                        0 <= end_r < self.board_size
                        and 0 <= end_c < self.board_size
                    ):
                        continue

                    if all(
                        self.board[r + k * dr][c + k * dc] == player_at_cell
                        for k in range(self.win_length)
                    ):
                        return player_at_cell

        return EMPTY

    def is_draw(self):
        """Hòa khi bàn cờ đầy và không có người thắng."""
        board_is_full = all(
            cell != EMPTY
            for row in self.board
            for cell in row
        )
        return board_is_full and self.winner == EMPTY

    def step(self, action):
        """
        Thực hiện nước đi của người đang đến lượt.

        Trả về bộ:
            observation, reward, terminated, truncated, info

        reward:
            +1: người vừa đi thắng
            -1: nước đi không hợp lệ hoặc ván đã kết thúc
             0: nước đi hợp lệ nhưng chưa kết thúc ván

        Nước đi không hợp lệ không làm thay đổi bàn cờ, lượt chơi hoặc
        trạng thái kết thúc. truncated luôn là False.
        """
        if self.done:
            return (
                self.get_observation(),
                -1,
                True,
                False,
                {
                    "error": "Ván đã kết thúc.",
                    "winner": self.winner,
                    "current_player": self.current_player,
                },
            )

        if (
            not isinstance(action, (tuple, list))
            or len(action) != 2
        ):
            return (
                self.get_observation(),
                -1,
                False,
                False,
                {"error": "action phải có dạng (row, col)."},
            )

        row, col = action

        if not self.is_valid_move(row, col):
            return (
                self.get_observation(),
                -1,
                False,
                False,
                {
                    "error": "Nước đi không hợp lệ.",
                    "current_player": self.current_player,
                },
            )

        player = self.current_player
        self.board[row][col] = player
        self.last_move = (row, col)

        found_winner = self.check_winner(row, col, player)
        reward = 0

        if found_winner != EMPTY:
            self.winner = found_winner
            self.done = True
            reward = 1
        elif not self.get_legal_moves():
            self.done = True
        else:
            self.current_player = -player

        info = {
            "player": player,
            "winner": self.winner,
            "current_player": self.current_player,
            "last_move": self.last_move,
            "is_draw": self.is_draw(),
        }

        return self.get_observation(), reward, self.done, False, info

    def render(self):
        """In bàn cờ ra Terminal."""
        symbols = {EMPTY: ".", X: "X", O: "O"}
        header = "    " + " ".join(f"{c:2d}" for c in range(self.board_size))
        print(header)

        for row_index, row in enumerate(self.board):
            cells = " ".join(f"{symbols[cell]:>2}" for cell in row)
            print(f"{row_index:2d}  {cells}")

        if self.winner == X:
            print("Người thắng: X")
        elif self.winner == O:
            print("Người thắng: O")
        elif self.done:
            print("Kết quả: Hòa")
        else:
            print(f"Lượt hiện tại: {'X' if self.current_player == X else 'O'}")


if __name__ == "__main__":
    env = CaroEnv()
    env.render()
