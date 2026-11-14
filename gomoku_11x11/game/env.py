import numpy as np

class GomokuEnv:
    def __init__(self, board_size=11):
        """
        Khởi tạo môi trường game cờ Caro.
        """
        self.board_size = board_size
        self.action_space = self.board_size * self.board_size
        self.observation_space = (self.board_size, self.board_size)
        self.state = None
        self.turn = None

    def reset(self):
        """
        Thiết lập lại môi trường về trạng thái khởi điểm.
        """
        # Khởi tạo ma trận trạng thái bằng Numpy toàn số 0
        self.state = np.zeros((self.board_size, self.board_size), dtype=int)
        self.turn = 1
        return self.state

    def step(self, action):
        """
        Thực hiện một nước đi và trả về trạng thái mới, phần thưởng, 
        và cờ báo hiệu kết thúc ván đấu[cite: 42].
        """
        row = action // self.board_size
        col = action % self.board_size

        # Xử lý nước đi lỗi
        if self.state[row, col] != 0:
            return self.state, -10, True 

        # Đặt quân cờ
        self.state[row, col] = self.turn

        # Kiểm tra thắng
        if self._check_win(row, col):
            return self.state, 1, True 

        # Kiểm tra hòa (sau khi fix hàm get_valid_moves, lệnh len() dưới đây sẽ hoạt động)
        if len(self.get_valid_moves()) == 0:
            return self.state, 0, True

        # Đổi lượt
        self.turn *= -1
        return self.state, 0, False

    def _check_win(self, row, col):
        """
        Quét 4 hướng xung quanh vị trí vừa đánh để kiểm tra chuỗi 5 quân liên tiếp.
        """
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        player = self.state[row, col]

        for dr, dc in directions:
            count = 1 
            r, c = row + dr, col + dc
            while 0 <= r < self.board_size and 0 <= c < self.board_size and self.state[r, c] == player:
                count += 1
                r += dr
                c += dc
                
            r, c = row - dr, col - dc
            while 0 <= r < self.board_size and 0 <= c < self.board_size and self.state[r, c] == player:
                count += 1
                r -= dr
                c -= dc
                
            if count >= 5:
                return True
                
        return False

    def render(self):
        """
        Hiển thị trạng thái hiện tại của bàn cờ[cite: 42].
        """
        pass # Sẽ triển khai ở Bước 5

    def get_valid_moves(self):
        """
        Trả về danh sách các nước đi hợp lệ (các ô còn trống).
        """
        # Trả về danh sách các nước đi hợp lệ
        if self.state is None:
            return []
        valid_positions = np.argwhere(self.state == 0)
        valid_actions = [int(row * self.board_size + col) for row, col in valid_positions]
        return valid_actions