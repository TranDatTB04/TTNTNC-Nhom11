import math
import numpy as np

class Node:
    def __init__(self, state, turn, parent=None, action=None, untried_actions=None):
        """
        Khởi tạo một nút (Node) trong cây tìm kiếm MCTS.
        - state: Ma trận 11x11 biểu diễn trạng thái bàn cờ tại nút này.
        - turn: Lượt của người chơi sẽ thực hiện hành động TIẾP THEO từ trạng thái này (1 hoặc -1).
        - parent: Nút cha dẫn tới nút hiện tại (None nếu là nút gốc).
        - action: Nước đi (0-120) từ nút cha dẫn tới trạng thái này.
        - untried_actions: Danh sách các nước đi hợp lệ chưa được mở rộng từ nút này.
        """
        self.state = state
        self.turn = turn
        self.parent = parent
        self.action = action
        
        self.children = []  # Danh sách các Node con đã được mở rộng
        # Tạo bản sao danh sách nước đi chưa thử để tránh sửa đổi dữ liệu gốc
        self.untried_actions = list(untried_actions) if untried_actions is not None else []
        
        self.visits = 0     # Số lần nút này được duyệt qua (N)
        self.value = 0.0    # Tổng điểm thắng tích lũy (W)

    def is_fully_expanded(self):
        """
        Kiểm tra xem nút này đã mở rộng hết tất cả các nước đi hợp lệ hay chưa.
        """
        return len(self.untried_actions) == 0

    def best_child(self, c_param=1.414):
        """
        Bước 2: Áp dụng công thức UCT (Upper Confidence Bound for Trees) 
        để chọn ra nút con tối ưu nhất trong quá trình Selection.
        c_param: Hằng số khám phá (mặc định là căn bậc 2 của 2 ≈ 1.414).
        """
        best_score = -float('inf')
        best_nodes = []

        for child in self.children:
            # Nếu nút con chưa từng được thăm, ưu tiên tuyệt đối (vô cùng)
            if child.visits == 0:
                uct_score = float('inf')
            else:
                # 1. Thành phần Khai thác (Exploitation): Tỷ lệ thắng trung bình của nút con
                # Lưu ý: child.value lưu điểm theo góc nhìn của người vừa đi tạo ra child (tức là self.turn)
                exploitation = child.value / child.visits
                
                # 2. Thành phần Khai phá (Exploration): Khuyến khích thử các nhánh ít được duyệt
                exploration = c_param * math.sqrt(math.log(self.visits) / child.visits)
                
                uct_score = exploitation + exploration

            # Cập nhật nút con có điểm UCT cao nhất
            if uct_score > best_score:
                best_score = uct_score
                best_nodes = [child]
            elif uct_score == best_score:
                best_nodes.append(child)

        # Nếu có nhiều nút bằng điểm nhau (ví dụ đều là vô cùng), chọn ngẫu nhiên 1 nút
        return best_nodes[np.random.randint(len(best_nodes))]