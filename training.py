from capture import GameState
from myutils import PacmanQAgent
import util

def training_agent():
    num_training = 10
    pacman = PacmanQAgent()


    for train in range(num_training):
        state = GameState()
        pacman.register_initial_state(state)

        while not state.is_over():
            action = pacman.getAction(state)
            state = state.generate_successor(pacman.index, action)
        pacman.final(state)
