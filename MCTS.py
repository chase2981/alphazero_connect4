import torch
from torch import nn
import torch.nn.functional as F
import math
import random
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from IPython.display import HTML
from base64 import b64encode

class MCTS:
    def __init__(self, network, game, config):
        """
        Initialize Monte Carlo Tree Search with a given neural network, game instance, and configuration.
        """
        self.network = network
        self.game = game
        self.config = config

    def search(self, state, total_iterations, temperature=None):
        """
        Performs a search for the desired number of iterations, returns an action and the tree root.
        """
        # Create the root
        root = Node(None, state, 1, self.game, self.config)

        # Expand the root, adding noise to each action
        # Get valid actions
        valid_actions = np.array(self.game.get_valid_actions(state), dtype=int)  # Ensure valid_actions is integer array
        state_tensor = torch.tensor(self.game.encode_state(state), dtype=torch.float).unsqueeze(0).to(self.config.device)
        with torch.no_grad():
            self.network.eval()
            value, logits = self.network(state_tensor)

        # Get action probabilities
        action_probs = F.softmax(logits.view(self.game.cols), dim=0).cpu().numpy()

        # Calculate and add Dirichlet noise
        noise = np.random.dirichlet([self.config.dirichlet_alpha] * self.game.cols)
        action_probs = ((1 - self.config.dirichlet_eps) * action_probs) + self.config.dirichlet_eps * noise

        # Mask unavailable actions
        mask = np.full(self.game.cols, False, dtype=bool)
        mask[valid_actions] = True
        action_probs = action_probs[mask]

        # Normalize probabilities
        action_probs /= np.sum(action_probs)

        # Create a child for each possible action
        for action, prob in zip(valid_actions, action_probs):
            child_state = -self.game.get_next_state(state, action)
            root.children[action] = Node(root, child_state, -1, self.game, self.config)
            root.children[action].prob = prob

        # Initialize root statistics
        root.n_visits = 1
        root.total_score = value.item()  # Use neural network value prediction

        # Begin search
        for _ in range(total_iterations):
            current_node = root

            # Phase 1: Selection
            while not current_node.is_leaf():
                current_node = current_node.select_child()

            # Phase 2: Expansion
            if not current_node.is_terminal():
                current_node.expand()
                state_tensor = torch.tensor(self.game.encode_state(current_node.state), dtype=torch.float).unsqueeze(0).to(self.config.device)
                with torch.no_grad():
                    self.network.eval()
                    value, logits = self.network(state_tensor)
                    value = value.item()

                # Mask invalid actions and calculate action probabilities
                valid_actions = np.array(self.game.get_valid_actions(current_node.state), dtype=int)
                mask = np.full(self.game.cols, False, dtype=bool)
                mask[valid_actions] = True
                action_probs = F.softmax(logits.view(self.game.cols)[mask], dim=0).cpu().numpy()
                for child, prob in zip(current_node.children.values(), action_probs):
                    child.prob = prob
            else:
                value = self.game.evaluate(current_node.state)

            # Phase 3: Backpropagation
            current_node.backpropagate(value)

        # Select action with specified temperature
        if temperature is None:
            temperature = self.config.temperature
        return self.select_action(root, temperature), root

    def select_action(self, root, temperature=None):
        """
        Select an action from the root based on visit counts, adjusted by temperature, 0 temp for greedy.
        """
        if temperature is None:
            temperature = self.config.temperature

        action_counts = {key: val.n_visits for key, val in root.children.items()}
        if temperature == 0:
            return max(action_counts, key=action_counts.get)
        elif temperature == np.inf:
            return np.random.choice(list(action_counts.keys()))
        else:
            distribution = np.array([*action_counts.values()]) ** (1 / temperature)
            return np.random.choice([*action_counts.keys()], p=distribution / sum(distribution))


class Node:
    def __init__(self, parent, state, to_play, game, config):
        """
        Represents a node in the MCTS, holding the game state and statistics for MCTS to operate.
        """
        self.parent = parent
        self.state = state
        self.to_play = to_play
        self.config = config
        self.game = game

        self.prob = 0
        self.children = {}
        self.n_visits = 0
        self.total_score = 0

    def expand(self):
        """
        Create child nodes for all valid actions. If state is terminal, evaluate and set the node's value.
        """
        valid_actions = np.array(self.game.get_valid_actions(self.state), dtype=int)  # Ensure valid_actions is integer array

        # If there are no valid actions, state is terminal, so get value using game instance
        if len(valid_actions) == 0:
            self.total_score = self.game.evaluate(self.state)
            return

        # Create a child for each possible action
        for action in valid_actions:
            # Make move, then flip board to perspective of next player
            child_state = -self.game.get_next_state(self.state, action)
            self.children[action] = Node(self, child_state, -self.to_play, self.game, self.config)

    def select_child(self):
        """
        Select the child node with the highest PUCT score.
        """
        best_puct = -np.inf
        best_child = None
        for child in self.children.values():
            puct = self.calculate_puct(child)
            if puct > best_puct:
                best_puct = puct
                best_child = child
        return best_child

    def calculate_puct(self, child):
        """
        Calculate the PUCT score for a given child node.
        """
        exploitation_term = 1 - (child.get_value() + 1) / 2  # Scale Q(s,a) to [0, 1]
        exploration_term = child.prob * math.sqrt(self.n_visits) / (child.n_visits + 1)
        return exploitation_term + self.config.exploration_constant * exploration_term

    def backpropagate(self, value):
        """
        Update the current node and its ancestors with the given value.
        """
        self.total_score += value
        self.n_visits += 1
        if self.parent is not None:
            self.parent.backpropagate(-value)  # Backpropagate the negative value to switch perspectives

    def is_leaf(self):
        """
        Check if the node is a leaf (no children).
        """
        return len(self.children) == 0

    def is_terminal(self):
        """
        Check if the node represents a terminal state.
        """
        return (self.n_visits != 0) and (len(self.children) == 0)

    def get_value(self):
        """
        Calculate the average value of this node.
        """
        if self.n_visits == 0:
            return 0
        return self.total_score / self.n_visits

    def __str__(self):
        """
        Return a string containing the node's relevant information for debugging purposes.
        """
        return (f"State:\n{self.state}\nProb: {self.prob}\nTo play: {self.to_play}" +
                f"\nNumber of children: {len(self.children)}\nNumber of visits: {self.n_visits}" +
                f"\nTotal score: {self.total_score}")
