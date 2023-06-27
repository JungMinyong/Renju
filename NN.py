'''
This code is Renju Algorithm using Monte Carlo Tree Search (MCTS) and Self - Play Reignforcement Learning. 
'''

__author__ = "Cha Yuhyun"
__email__ = "caca518@kaist.ac.kr"
__Last_Modify__ = "June 27, 2023"


import numpy as np
import random

class RenjuGame:
    def __init__(self, board_size):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=np.int)
        self.current_player = 1
        self.winner = None

    def is_valid_move(self, row, col):
        return self.board[row][col] == 0

    def make_move(self, row, col):
        if self.is_valid_move(row, col):
            self.board[row][col] = self.current_player
            self.check_winner()
            self.current_player = 3 - self.current_player

    def check_winner(self):
        directions = [(0, 1), (1, 0), (1, 1), (-1, 1)]  # horizontal, vertical, diagonal, anti-diagonal
        for dr, dc in directions:
            for row in range(self.board_size):
                for col in range(self.board_size):
                    if self.board[row][col] != 0:
                        color = self.board[row][col]
                        win = True
                        for i in range(5):
                            if row + i * dr < 0 or row + i * dr >= self.board_size or col + i * dc < 0 or col + i * dc >= self.board_size or self.board[row + i * dr][col + i * dc] != color:
                                win = False
                                break
                        if win:
                            self.winner = color
                            return

    def is_game_over(self):
        return np.count_nonzero(self.board) == self.board_size * self.board_size or self.winner is not None

    def get_state(self):
        return self.board.copy()

    def get_valid_moves(self):
        valid_moves = []
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.is_valid_move(row, col):
                    valid_moves.append((row, col))
        return valid_moves

    def print_board(self):
        for row in self.board:
            print(row)
        print()

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.wins = 0
        self.visits = 0

    def expand(self):
        valid_moves = self.state.get_valid_moves()
        for move in valid_moves:
            new_state = self.state.get_state()
            new_state[move[0]][move[1]] = self.state.current_player
            child = MCTSNode(RenjuGame(self.state.board_size), self)
            child.state.board = new_state
            child.state.current_player = 3 - self.state.current_player
            self.children.append(child)

    def select_best_child(self, c_param):
        best_child = None
        best_score = float('-inf')
        for child in self.children:
            score = child.wins / child.visits + c_param * np.sqrt(2 * np.log(self.visits) / child.visits)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def rollout(self):
        state = self.state
        while not state.is_game_over():
            valid_moves = state.get_valid_moves()
            move = random.choice(valid_moves)
            state.make_move(move[0], move[1])
        return state.winner

    def backpropagate(self, result):
        node = self
        while node is not None:
            node.visits += 1
            if node.state.current_player == result:
                node.wins += 1
            node = node.parent

    def get_best_move(self):
        best_child = None
        best_score = float('-inf')
        for child in self.children:
            score = child.wins / child.visits
            if score > best_score:
                best_score = score
                best_child = child
        return best_child.state.get_valid_moves()[0]

class MCTSAgent:
    def __init__(self, iterations):
        self.iterations = iterations

    def get_action(self, state):
        root = MCTSNode(state)
        for _ in range(self.iterations):
            node = self.select_node(root)
            if node.state.is_game_over():
                result = node.state.winner
            else:
                node.expand()
                child = random.choice(node.children)
                result = child.rollout()
            node.backpropagate(result)
        return root.get_best_move()

    def select_node(self, node):
        while len(node.children) > 0:
            if not all(child.visits > 0 for child in node.children):
                return node
            node = node.select_best_child(c_param=1.0 / np.sqrt(2))
        return node

def self_play(agent):
    game = RenjuGame(board_size=15)
    while not game.is_game_over():
        move = agent.get_action(game)
        game.make_move(move[0], move[1])
    return game.winner

# Training
iterations = 1000
agent = MCTSAgent(iterations)

num_games = 100
wins = 0
losses = 0
draws = 0

for _ in range(num_games):
    winner = self_play(agent)
    if winner == 1:
        wins += 1
    elif winner == 2:
        losses += 1
    else:
        draws += 1

print(f"Wins: {wins}, Losses: {losses}, Draws: {draws}")
