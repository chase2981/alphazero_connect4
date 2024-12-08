import streamlit as st
import time
import torch
from datetime import datetime
import pytz
import numpy as np

# Replace these imports with your actual implementations
from config import config
from AlphaZero import AlphaZero
from Connect4 import Connect4
from AlphaZeroAgent import AlphaZeroAgent


# Add custom CSS styles for buttons
st.markdown("""
    <style>

        .stButton button {
            margin: 2px; /* Adjust margin for tighter spacing */
            padding: 8px 12px; /* Adjust padding for button size */
            font-size: 18px; /* Adjust font size */
            width: 100%;
        }
        .stHorizontalBlock {
            // margin: auto;
        }
        .stColumn {
            // flex: 1 1 50px; /* Adjust this value for column width */
            // max-width: 50px; /* Same as above to limit the max width */
            min-width: 40px;
            // width: 50px !important;
            // margin: 2px;
            width: 100%;
            padding: 0px;
        }
        .customTd {
            padding: 5px;
        }
        .dynamic-size {
            font-size: 2rem; /* Default size for smaller screens */
        }

        @media (min-width: 768px) { /* Tablet size and above */
            .dynamic-size {
            font-size: 3rem;
            }
        }
    </style>
""", unsafe_allow_html=True)

# Initialize the game
class Connect4Game:
    def __init__(self):
        self.game = Connect4()
        self.alphazero = AlphaZero(self.game, config)
        self.agent = AlphaZeroAgent(self.alphazero)
        self.state = self.game.reset()
        self.turn = 0  # 0 = Human, 1 = AI
        self.done = False
        self.winning_slots = None
        self.difficulty_map = {"Easy": 200, "Medium": 500, "Hard": 1000}
        self.difficulty = "Medium"
        self.ai_confidence = "N/A"  # AI confidence as a percentage

    def check_winner(self):
        rows, cols = self.game.rows, self.game.cols
        for r in range(rows):
            for c in range(cols - 3):  # Horizontal check
                if abs(sum(self.state[r, c:c + 4])) == 4:
                    return [(r, c + i) for i in range(4)]
        for r in range(rows - 3):
            for c in range(cols):  # Vertical check
                if abs(sum(self.state[r:r + 4, c])) == 4:
                    return [(r + i, c) for i in range(4)]
        for r in range(rows - 3):
            for c in range(cols - 3):  # Main diagonal check
                if abs(sum(self.state[r + i, c + i] for i in range(4))) == 4:
                    return [(r + i, c + i) for i in range(4)]
        for r in range(3, rows):
            for c in range(cols - 3):  # Anti-diagonal check
                if abs(sum(self.state[r - i, c + i] for i in range(4))) == 4:
                    return [(r - i, c + i) for i in range(4)]
        return None

    def draw_board(self):
        html = '<table style="border-collapse: collapse; margin: 0 auto; text-align: center; width: 100%;">'
        for r, row in enumerate(self.state):
            html += "<tr>"
            for c, cell in enumerate(row):
                if self.winning_slots and (r, c) in self.winning_slots:
                    style = "background-color: gold;"  # Highlight winning slots
                else:
                    style = ""
                if cell == 1:
                    color = "🔴"
                elif cell == -1:
                    color = "🟡"
                else:
                    color = "⚪"
                html += f'<td class="customTd dynamic-size" style="{style}">{color}</td>'
            html += "</tr>"
        html += "</table>"
        return html

    def ai_thinking_html(self):
        thinking_html = '''
        <div style="text-align: center; margin: 10px;">
            <svg width="50" height="50" viewBox="0 0 50 50" xmlns="http://www.w3.org/2000/svg" fill="#007BFF">
                <circle cx="25" cy="25" r="20" stroke-width="4" stroke="#007BFF" stroke-dasharray="31.4 31.4" fill="none">
                    <animateTransform attributeName="transform" type="rotate" from="0 25 25" to="360 25 25" dur="1s" repeatCount="indefinite" />
                </circle>
            </svg>
            <p>AI is thinking...</p>
        </div>
        '''
        return thinking_html

    def calculate_ai_confidence(self):
        """Calculate the AI's confidence of winning."""
        with torch.no_grad():
            state_tensor = torch.tensor(
                self.game.encode_state(self.state), dtype=torch.float
            ).unsqueeze(0).to(config.device)
            value, _ = self.alphazero.network(state_tensor)
            confidence = (value.item() + 1) / 2 * 100  # Scale to 0-100%
        self.ai_confidence = f"{confidence:.2f}%"

    def handle_move(self, column):
        if self.done:
            return "Game is over! Reset to play again."
        if self.turn == 0:  # Human move
            if column not in self.game.get_valid_actions(self.state):
                return f"Column {column} is full. Choose another column."
            self.state, reward, self.done = self.game.step(self.state, column, to_play=1)
            self.turn = 1  # Switch to AI
        else:  # AI move
            column = self.agent.select_action(
                self.state, search_iterations=self.difficulty_map[self.difficulty]
            )
            self.state, reward, self.done = self.game.step(self.state, column, to_play=-1)
            self.calculate_ai_confidence()  # Update AI confidence
            self.turn = 0  # Switch back to Human

        self.winning_slots = self.check_winner()
        if self.done:
            if reward == 1:
                return "🎉 You win!"
            elif reward == -1:
                return "😞 AI wins!"
            else:
                return "🤝 It's a draw!"
        return None

    def reset_game(self):
        self.state = self.game.reset()
        self.turn = 0
        self.done = False
        self.winning_slots = None
        self.ai_confidence = "N/A"  # Reset AI confidence


# Initialize the game
if "connect4_game" not in st.session_state:
    st.session_state["connect4_game"] = Connect4Game()

game = st.session_state["connect4_game"]

# Sidebar
st.sidebar.header("Game Settings")

# Model Loading
model_selection = st.sidebar.selectbox(
    "Select Model", ["Untrained Model", "50-epochs", "80-epochs"], index=0
)

difficulty = st.sidebar.selectbox("Select Difficulty", ["Easy", "Medium", "Hard"], index=1)
game.difficulty = difficulty


if st.sidebar.button("Load Model"):
    try:
        file_path = f"./models/{model_selection}.pth"
        pre_trained_weights = torch.load(file_path, map_location=config.device)
        game.alphazero.network.load_state_dict(pre_trained_weights)
        st.sidebar.success(f"Loaded {model_selection} successfully!")
    except FileNotFoundError:
        st.sidebar.error("Model file not found.")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
# else:
#     try:
#         file_path = f"./models/50-epochs.pth"
#         pre_trained_weights = torch.load(file_path, map_location=config.device)
#         game.alphazero.network.load_state_dict(pre_trained_weights)
#         st.sidebar.success(f"Loaded 50-epochs model successfully!")
#     except FileNotFoundError:
#         st.sidebar.error("Model file not found.")
#     except Exception as e:
#         st.sidebar.error(f"Error loading model: {e}")

# Reset button
if st.sidebar.button("Reset Game"):
    game.reset_game()
    st.rerun()

# Main Interface
st.markdown("<h1 style='text-align: center;'>Connect 4</h1>", unsafe_allow_html=True)
st.markdown(game.draw_board(), unsafe_allow_html=True)


# Column Buttons
cols = st.columns(game.game.cols)
for i, col in enumerate(cols):
    if col.button(f"{i}", disabled=game.turn == 1):
        message = game.handle_move(i)
        if message:
            st.success(message)
            st.sidebar.success(message)
        st.rerun()

# AI Thinking Indicator
if game.turn == 1 and not game.done:
    st.markdown(game.ai_thinking_html(), unsafe_allow_html=True)
    time.sleep(1)  # Simulate AI thinking
    message = game.handle_move(None)  # AI makes its move
    st.rerun()

# Display AI Confidence
if game.turn == 0:
    st.markdown(f"<p style='text-align: center;'>AI Win Confidence: {game.ai_confidence}</p>", unsafe_allow_html=True)
