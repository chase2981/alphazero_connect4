import torch
from torch import nn
import torch.nn.functional as F
import math
import random
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from base64 import b64encode

class AlphaZeroAgent:
    def __init__(self, alphazero):
        self.alphazero = alphazero
        self.alphazero.network.eval()

        # Remove noise from move calculations
        self.alphazero.config.dirichlet_eps = 0

    def select_action(self, state, search_iterations=200):
        state_tensor = torch.tensor(self.alphazero.game.encode_state(state), dtype=torch.float).to(self.alphazero.config.device)

        # Get action without using search
        if search_iterations == 0:
            with torch.no_grad():
                _, logits = self.alphazero.network(state_tensor.unsqueeze(0))

            # Get action probs and mask for valid actions
            action_probs = F.softmax(logits.view(-1), dim=0).cpu().numpy()
            valid_actions = self.alphazero.game.get_valid_actions(state)
            valid_action_probs = action_probs[valid_actions]
            best_action = valid_actions[np.argmax(valid_action_probs)]
            return best_action
        # Else use MCTS
        else:
            action, _ = self.alphazero.mcts.search(state, search_iterations)
            return action

    def get_win_confidence(self, state, search_iterations):
        """
        Compute AI's win confidence rate based on the value head.
        """
        # Perform MCTS (to keep it consistent with gameplay behavior)
        _, mcts_statistics = self.alphazero.mcts.search(state, search_iterations)

        # Encode the state for the neural network
        state_tensor = torch.tensor(
            self.alphazero.game.encode_state(state), dtype=torch.float
        ).unsqueeze(0).to(self.alphazero.config.device)

        # Pass the state through the network
        policy_logits, value = self.alphazero.network(state_tensor)

        # If the value tensor has multiple elements, average them to get a single confidence score
        if value.numel() > 1:
            confidence_value = value.mean().item()  # Take the mean if multiple values are present
        else:
            confidence_value = value.item()  # Use the scalar directly if only one value

        # Convert the value head output to a confidence percentage
        confidence = (confidence_value + 1) / 2 * 100  # Convert range [-1, 1] to [0%, 100%]

        return confidence

    def predict_human_move(self, state, search_iterations=200):
        """
        Predict the most likely move the human will make next.
        """
        # Get valid actions
        valid_actions = self.alphazero.game.get_valid_actions(state)

        # Store potential moves and their evaluations
        human_move_scores = {}

        for action in valid_actions:
            # Simulate the human's move
            simulated_state = self.alphazero.game.get_next_state(state, action, to_play=1)
            
            # Perform MCTS to evaluate the state after the human's move
            _, mcts_statistics = self.alphazero.mcts.search(simulated_state, search_iterations)
            
            # Evaluate the state from the human's perspective
            human_move_scores[action] = -mcts_statistics.total_score  # Flip the sign for the human's perspective

        # Sort moves by their score (higher score = better for the human)
        sorted_moves = sorted(human_move_scores.items(), key=lambda x: x[1], reverse=True)

        # Return the best predicted move
        best_move = sorted_moves[0][0]
        best_score = sorted_moves[0][1]

        return best_move, best_score
