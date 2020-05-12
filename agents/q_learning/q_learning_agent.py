import ludopy
import numpy as np

from copy import deepcopy

from ..agent import Agent
from .q_table import QTable
from .q_state import QState

class QLearningAgent(Agent):
    def __init__(
            self,
            game,
            discount_factor=1,
            learning_rate=0.1
    ):
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate

        self.states = []

        Agent.__init__(
                self,
                game
        )

    def determine_piece_to_move(self): #todo
        (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner, there_is_a_winner), player_i = self.game.get_observation()

        self.dice = deepcopy(dice)
        self.move_pieces = deepcopy(move_pieces)
        self.player_pieces = deepcopy(player_pieces)
        self.enemy_pieces = deepcopy(enemy_pieces)

        if len(move_pieces) > 0:
            return move_pieces[np.random.randint(0, len(move_pieces))]
        else:
            return -1

    def on_finished_move(self): #todo
        q_state = QState(
                self.dice,
                self.player_pieces,
                self.enemy_pieces
        )

        key = q_state.get_key()

        if key in self.states:
            print("ups")

        self.states.append(key)

