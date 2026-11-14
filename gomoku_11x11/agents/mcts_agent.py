import copy
import random
import numpy as np
from agents.mcts_node import MCTSNode


class MCTSAgent:
    def __init__(self, simulations=800, c_param=1.414):
        self.simulations = simulations
        self.c_param = c_param

    def get_action(self, env):
        root_state = env.state.copy()
        root_turn = env.turn
        valid_moves = env.get_legal_moves()

        board_size = env.state.shape[0]
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

            # 1. SELECTION
            while node.is_fully_expanded() and len(node.children) > 0:
                node = node.best_child(self.c_param)
                _, reward, done, _ = sim_env.step(node.action)
                if done:
                    break

            # 2. EXPANSION
            if not done and not node.is_fully_expanded():
                action = random.choice(node.untried_actions)
                node.untried_actions.remove(action)
                
                next_state, reward, done, _ = sim_env.step(action)
                
                child_node = MCTSNode(
                    state=next_state.copy(),
                    player_turn=sim_env.turn,
                    parent=node,
                    action=action,
                    untried_actions=sim_env.get_legal_moves() if not done else []
                )
                node.children.append(child_node)
                node = child_node

            # 3. SIMULATION
            while not done:
                legal_moves = sim_env.get_legal_moves()
                if not legal_moves:
                    reward = 0
                    break
                random_action = random.choice(legal_moves)
                _, reward, done, _ = sim_env.step(random_action)

            winner = sim_env.turn if reward == 1 else 0

            # 4. BACKPROPAGATION
            curr = node
            while curr is not None:
                curr.visits += 1
                if winner != 0 and curr.parent is not None:
                    if curr.parent.player_turn == winner:
                        curr.value += 1.0
                    else:
                        curr.value -= 1.0
                curr = curr.parent

        best_child = max(root_node.children, key=lambda c: c.visits)
        return best_child.action

    def _clone_env(self, original_env, state, turn):
        cloned = copy.copy(original_env)
        cloned.state = state.copy()
        cloned.turn = turn
        return cloned