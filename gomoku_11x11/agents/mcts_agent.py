import copy
import random
from agents.mcts_node import MCTSNode
from environment.caro_environment import EMPTY, X, O


class MCTSAgent:
    def __init__(self, simulations=600, c_param=1.414, max_rollout_depth=16):
        self.simulations = simulations
        self.c_param = c_param
        self.max_rollout_depth = max_rollout_depth

    def get_action(self, env):
        root_board = env.get_observation()
        root_player = env.current_player
        legal_moves = env.get_legal_moves()

        if not legal_moves:
            return None

        # 1. Khai cuộc vào tâm bàn cờ
        center = (env.board_size // 2, env.board_size // 2)
        if len(legal_moves) == env.board_size * env.board_size and center in legal_moves:
            return center

        # 2. PHẢN XẠ CHIẾN THUẬT TỨC THÌ (Thắng ngay hoặc Chặn thua ngay)
        urgent_move = self._find_urgent_move(env, root_player, legal_moves)
        if urgent_move is not None:
            return urgent_move

        candidate_moves = self._get_candidate_moves(env.board, env.board_size, radius=2)
        if not candidate_moves:
            candidate_moves = legal_moves

        root_node = MCTSNode(
            state=root_board,
            player_turn=root_player,
            untried_actions=candidate_moves
        )

        for _ in range(self.simulations):
            node = root_node
            sim_env = self._clone_env(env)
            done = False

            # SELECTION
            while node.is_fully_expanded() and len(node.children) > 0:
                node = node.best_child(self.c_param)
                self._fast_step(sim_env, node.action)
                if sim_env.done:
                    done = True
                    break

            # EXPANSION
            if not done and not node.is_fully_expanded():
                action = random.choice(node.untried_actions)
                node.untried_actions.remove(action)

                self._fast_step(sim_env, action)
                done = sim_env.done
                next_moves = self._get_candidate_moves(sim_env.board, sim_env.board_size, radius=2) if not done else []

                child_node = MCTSNode(
                    state=None,  # Không cần lưu bản sao state ở từng nút con để tiết kiệm RAM & CPU
                    player_turn=sim_env.current_player,
                    parent=node,
                    action=action,
                    untried_actions=next_moves
                )
                node.children.append(child_node)
                node = child_node

            # SIMULATION (Giới hạn độ sâu & ưu tiên vùng giao tranh sát cạnh)
            depth = 0
            while not done and depth < self.max_rollout_depth:
                # Ưu tiên đánh sát cạnh (radius=1) để mô phỏng các pha giao tranh thực tế
                rollout_moves = self._get_candidate_moves(sim_env.board, sim_env.board_size, radius=1)
                if not rollout_moves:
                    break
                random_action = random.choice(rollout_moves)
                self._fast_step(sim_env, random_action)
                done = sim_env.done
                depth += 1

            winner = sim_env.winner

            # BACKPROPAGATION
            curr = node
            while curr is not None:
                curr.visits += 1
                if winner != EMPTY and curr.parent is not None:
                    if curr.parent.player_turn == winner:
                        curr.value += 1.0
                    else:
                        curr.value -= 1.0
                curr = curr.parent

        if not root_node.children:
            return random.choice(candidate_moves)

        best_child = max(root_node.children, key=lambda c: c.visits)
        return best_child.action

    def _find_urgent_move(self, env, player, legal_moves):
        """Kiểm tra nước đi quyết định (tạo 5 quân) hoặc chặn đối thủ tạo 5 quân."""
        opponent = -player
        candidates = self._get_candidate_moves(env.board, env.board_size, radius=1)
        if not candidates:
            candidates = legal_moves

        # Ưu tiên 1: Nếu mình đánh ô này thắng luôn -> đánh ngay
        for r, c in candidates:
            env.board[r][c] = player
            is_win = (env.check_winner(r, c, player) == player)
            env.board[r][c] = EMPTY
            if is_win:
                return (r, c)

        # Ưu tiên 2: Nếu đối thủ đánh ô này sẽ thắng ở lượt sau -> chặn ngay
        for r, c in candidates:
            env.board[r][c] = opponent
            opp_win = (env.check_winner(r, c, opponent) == opponent)
            env.board[r][c] = EMPTY
            if opp_win:
                return (r, c)

        return None

    def _fast_step(self, sim_env, action):
        """Phiên bản step rút gọn không gọi deepcopy để tăng tốc mô phỏng gấp 10 lần."""
        r, c = action
        player = sim_env.current_player
        sim_env.board[r][c] = player
        sim_env.last_move = action

        if sim_env.check_winner(r, c, player) != EMPTY:
            sim_env.winner = player
            sim_env.done = True
        else:
            sim_env.current_player = -player

    def _get_candidate_moves(self, board, size, radius=2):
        candidates = set()
        for r in range(size):
            for c in range(size):
                if board[r][c] != EMPTY:
                    r_min = max(0, r - radius)
                    r_max = min(size, r + radius + 1)
                    c_min = max(0, c - radius)
                    c_max = min(size, c + radius + 1)
                    for nr in range(r_min, r_max):
                        for nc in range(c_min, c_max):
                            if board[nr][nc] == EMPTY:
                                candidates.add((nr, nc))
        return list(candidates)

    def _clone_env(self, original_env):
        cloned = copy.copy(original_env)
        cloned.board = [row[:] for row in original_env.board]
        return cloned