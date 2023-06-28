'''
This code is Renju Algorithm using Monte Carlo Tree Search (MCTS) and Self - Play Reignforcement Learning. 
'''

__author__ = "Cha Yuhyun"
__email__ = "caca518@kaist.ac.kr"
__Last_Modify__ = "June 27, 2023"

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

class RenjuGame:
    def __init__(self, board_size):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=np.int)
        self.current_player = 1                                             # current_player = 1 or 2
        self.winner = None
        self.last_move_col = 0
        self.last_move_row = 0

    def is_valid_move(self, row, col):
        return self.board[row][col] == 0

    def make_move(self, row, col):
        if self.is_valid_move(row, col):
            self.board[row][col] = self.current_player
            self.check_winner()
            self.current_player = 3 - self.current_player
            self.last_move_row = row
            self.last_move_col = col

 def check_winner(self):
     directions = [(0, 1), (1, 0), (1, 1), (-1, 1)]  # horizontal, vertical, diagonal, anti-diagonal
     
     last_row, last_col = self.last_move_row, self.last_move_col
     last_color = self.board[last_row][last_col]
 
     for dr, dc in directions:
         win = True
         for i in range(-4, 5):
             row = last_row + i * dr
             col = last_col + i * dc
             if (
                 row < 0
                 or row >= self.board_size
                 or col < 0
                 or col >= self.board_size
                 or self.board[row][col] != last_color
             ):
                 win = False
                 break
 
         if win:
             self.winner = last_color
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

class PolicyNetwork(nn.Module):
    def __init__(self, board_size):
        super(PolicyNetwork, self).__init__()
        self.board_size = board_size
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)    # First convolutional layer with input channels=1 and output channels=32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)   # Second convolutional layer with input channels=32 and output channels=64
        self.fc1 = nn.Linear(64 * (board_size ** 2 // 4), 256)               # Fully connected layer with input features=64*(board_size^2/4) and output features=256
        self.fc2 = nn.Linear(256, board_size ** 2)                           # Fully connected layer with input features=256 and output features=board_size^2

    def forward(self, x):
        x = torch.relu(self.conv1(x))        # Apply ReLU activation to the output of the first convolutional layer
        x = torch.relu(self.conv2(x))        # Apply ReLU activation to the output of the second convolutional layer
        x = x.view(x.size(0), -1)            # Reshape x into a 2D matrix with size (batch_size, -1)
        x = torch.relu(self.fc1(x))          # Apply ReLU activation to the output of the first fully connected layer
        x = self.fc2(x)                      # Output the final logits from the second fully connected layer
        return x

class MCTSAgent:
    def __init__(self, iterations, board_size):
        self.iterations = iterations
        self.board_size = board_size
        self.policy_network = PolicyNetwork(board_size)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=0.001)

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

    def self_play(self):
        game = RenjuGame(board_size=self.board_size)
        states = []
        while not game.is_game_over():
            state_tensor = torch.tensor(game.get_state().reshape(1, 1, self.board_size, self.board_size), dtype=torch.float32)
            states.append(state_tensor)
            move = self.get_action(game)
            game.make_move(move[0], move[1])
        winner = game.winner
        labels = torch.zeros(len(states), self.board_size ** 2)
        if winner != 0:
            for i in range(len(states)):
                labels[i][move[0] * self.board_size + move[1]] = 1 # label : i th move is what, move represented by 1 dimension
        return states, labels

    def train(self, num_games):
        for _ in range(num_games):
            states, labels = self.self_play()
            inputs = torch.cat(states, dim=0)
            targets = torch.cat(labels, dim=0)
            self.optimizer.zero_grad()
            outputs = self.policy_network(inputs)
            loss = nn.BCEWithLogitsLoss()(outputs, targets)
            loss.backward()
            self.optimizer.step()

# Training
iterations = 10
agent = MCTSAgent(iterations, board_size=15)

num_games = 100                                 
agent.train(num_games)

# Testing
game = RenjuGame(board_size=15)
while not game.is_game_over():
    state_tensor = torch.tensor(game.get_state().reshape(1, 1, 15, 15), dtype=torch.float32)
    output = agent.policy_network(state_tensor)
    probabilities = torch.sigmoid(output)
    valid_moves = game.get_valid_moves()
    probabilities = probabilities.squeeze().detach().numpy()
    valid_probabilities = probabilities.reshape(15, 15)[np.array(valid_moves)[:, 0], np.array(valid_moves)[:, 1]]
    best_move = valid_moves[np.argmax(valid_probabilities)]
    game.make_move(best_move[0], best_move[1])

print(f"Winner: {game.winner}")
