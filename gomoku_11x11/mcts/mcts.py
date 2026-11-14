import copy
import random
import numpy as np
from mcts.node import Node

class MCTS:
    def __init__(self, env, simulations=1000, c_param=1.414):
        """
        Khởi tạo bộ tìm kiếm MCTS thuần túy.
        - env: Đối tượng môi trường GomokuEnv.
        - simulations: Số lần lặp lại chu trình 4 bước cho mỗi nước đi của AI.
        - c_param: Hằng số khám phá trong công thức UCT.
        """
        self.env = env
        self.simulations = simulations
        self.c_param = c_param

    def search(self, root_state, root_turn):
        """
        Hàm chính thực thi tìm kiếm MCTS và trả về nước đi (action) tối ưu nhất.
        """
        # Tạo bản sao môi trường để lấy danh sách nước đi hợp lệ tại nút gốc
        temp_env = self._clone_env(root_state, root_turn)
        root_node = Node(
            state=root_state.copy(),
            turn=root_turn,
            untried_actions=temp_env.get_valid_moves()
        )

        for _ in range(self.simulations):
            node = root_node
            sim_env = self._clone_env(root_state, root_turn)
            done = False
            reward = 0

            # 1. SELECTION (Lựa chọn)
            # Đi xuống cây theo các nút con có điểm UCT cao nhất cho đến khi gặp nút chưa mở rộng hết hoặc kết thúc
            while node.is_fully_expanded() and len(node.children) > 0:
                node = node.best_child(self.c_param)
                _, reward, done = sim_env.step(node.action)
                if done:
                    break

            # 2. EXPANSION (Mở rộng)
            # Nếu ván đấu chưa kết thúc và nút hiện tại còn nước đi chưa thử, tạo thêm 1 nút con mới
            if not done and not node.is_fully_expanded():
                action = random.choice(node.untried_actions)
                node.untried_actions.remove(action)
                
                next_state, reward, done = sim_env.step(action)
                
                child_node = Node(
                    state=next_state.copy(),
                    turn=sim_env.turn,
                    parent=node,
                    action=action,
                    untried_actions=sim_env.get_valid_moves() if not done else []
                )
                node.children.append(child_node)
                node = child_node

            # 3. SIMULATION / ROLLOUT (Mô phỏng)
            # Đánh ngẫu nhiên luân phiên cho đến khi ván đấu ngã ngũ
            while not done:
                valid_moves = sim_env.get_valid_moves()
                if not valid_moves:
                    reward = 0
                    break
                random_action = random.choice(valid_moves)
                _, reward, done = sim_env.step(random_action)

            # Xác định người chiến thắng thực sự của ván mô phỏng
            # Lưu ý: Trong GomokuEnv, khi thắng (reward == 1), sim_env.turn chính là người vừa đánh nước quyết định
            winner = sim_env.turn if reward == 1 else 0

            # 4. BACKPROPAGATION (Lan truyền ngược)
            # Cập nhật lại lượt thăm và điểm số từ nút lá ngược về tận nút gốc
            while node is not None:
                node.visits += 1
                
                if winner != 0 and node.parent is not None:
                    # Điểm của node lưu theo góc nhìn của người vừa đi tạo ra node đó (tức là node.parent.turn)
                    if node.parent.turn == winner:
                        node.value += 1.0
                    else:
                        node.value -= 1.0
                        
                node = node.parent

        # RA QUYẾT ĐỊNH: Chọn nút con trực tiếp của nút gốc có số lượt ghé thăm (visits) cao nhất
        # (Thực tế chứng minh chọn theo visits ổn định và ít nhiễu hơn chọn theo tỷ lệ thắng thuần túy)
        best_child = max(root_node.children, key=lambda c: c.visits)
        return best_child.action

    def _clone_env(self, state, turn):
        """
        Tạo một bản sao độc lập của môi trường game để chạy mô phỏng mà không làm hỏng bàn cờ thật.
        """
        cloned = copy.copy(self.env)
        cloned.state = state.copy()
        cloned.turn = turn
        return cloned