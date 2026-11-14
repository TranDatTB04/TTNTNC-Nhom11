import copy
import random
import numpy as np
from agents.mcts_node import MCTSNode


class MCTSAgent:
    def __init__(self, simulations=800, c_param=1.414):
        """
        Tác tử MCTS thuần túy tương thích với CaroEnv mới.
        """
        self.simulations = simulations
        self.c_param = c_param

    def get_action(self, env):
        """
        Nhận vào đối tượng CaroEnv hiện tại và trả về action tối ưu dạng tuple (row, col).
        """
        # Sử dụng self.board và self.current_player từ env mới
        root_state = copy.deepcopy(env.board)
        root_turn = env.current_player
        valid_moves = env.get_legal_moves()

        if not valid_moves:
            return None

        # Nếu bàn cờ trống hoàn toàn, đánh ngay tâm (5, 5) để tiết kiệm thời gian
        board_size = env.board_size
        center = (board_size // 2, board_size // 2)
        if len(valid_moves) == board_size * board_size and center in valid_moves:
            return center

        root_node = MCTSNode(
            state=root_state,
            player_turn=root_turn,
            untried_actions=valid_moves
        )

        for _ in range(self.simulations):
            node = root_node
            sim_env = self._clone_env(env, root_state, root_turn)
            done = False
            reward = 0

            # 1. SELECTION: Đi xuống cây theo UCT
            while node.is_fully_expanded() and len(node.children) > 0:
                node = node.best_child(self.c_param)
                _, reward, terminated, truncated, _ = sim_env.step(node.action)
                done = terminated or truncated
                if done:
                    break

            # 2. EXPANSION: Mở rộng 1 nút con mới
            if not done and not node.is_fully_expanded():
                action = random.choice(node.untried_actions)
                node.untried_actions.remove(action)
                
                next_state, reward, terminated, truncated, _ = sim_env.step(action)
                done = terminated or truncated
                
                child_node = MCTSNode(
                    state=copy.deepcopy(next_state),
                    player_turn=sim_env.current_player,
                    parent=node,
                    action=action,
                    untried_actions=sim_env.get_legal_moves() if not done else []
                )
                node.children.append(child_node)
                node = child_node

            # 3. SIMULATION / ROLLOUT: Mô phỏng ngẫu nhiên đến hết ván
            while not done:
                legal_moves = sim_env.get_legal_moves()
                if not legal_moves:
                    break
                random_action = random.choice(legal_moves)
                _, reward, terminated, truncated, _ = sim_env.step(random_action)
                done = terminated or truncated

            # Xác định người chiến thắng dựa vào sim_env.winner
            winner = sim_env.winner

            # 4. BACKPROPAGATION: Cập nhật ngược kết quả lên các nút cha
            curr = node
            while curr is not None:
                curr.visits += 1
                if winner != 0 and curr.parent is not None:
                    if curr.parent.player_turn == winner:
                        curr.value += 1.0
                    else:
                        curr.value -= 1.0
                curr = curr.parent

        # Chọn nút con trực tiếp của nút gốc có số lượt ghé thăm (visits) cao nhất
        best_child = max(root_node.children, key=lambda c: c.visits)
        return best_child.action

    def _clone_env(self, original_env, state, turn):
        """Tạo bản sao môi trường độc lập để mô phỏng không làm ảnh hưởng bàn cờ thật."""
        cloned = copy.copy(original_env)
        cloned.board = copy.deepcopy(state)
        cloned.current_player = turn
        cloned.winner = 0
        cloned.done = False
        return cloned