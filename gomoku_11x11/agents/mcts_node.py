import math
import numpy as np


class MCTSNode:
    def __init__(self, state, player_turn, parent=None, action=None, untried_actions=None):
        self.state = state
        self.player_turn = player_turn
        self.parent = parent
        self.action = action
        
        self.children = []
        self.untried_actions = list(untried_actions) if untried_actions is not None else []
        
        self.visits = 0     # Số lượt ghé thăm (N)
        self.value = 0.0    # Tổng điểm tích lũy (W)

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c_param=1.414):
        best_score = -float('inf')
        best_nodes = []

        for child in self.children:
            if child.visits == 0:
                uct_score = float('inf')
            else:
                exploitation = child.value / child.visits
                exploration = c_param * math.sqrt(math.log(self.visits) / child.visits)
                uct_score = exploitation + exploration

            if uct_score > best_score:
                best_score = uct_score
                best_nodes = [child]
            elif uct_score == best_score:
                best_nodes.append(child)

        return best_nodes[np.random.randint(len(best_nodes))]