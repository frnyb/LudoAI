import ludopy
import numpy as np

from copy import deepcopy

from ..agent import Agent
from .q_table import QTable

class QLearningAgent(Agent):
    def __init__(
            self,
            game,
            player_number,
            discount_factor=1,
            learning_rate=0.1
    ):
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate

        self.q_table = QTable(
                self.discount_factor,
                self.learning_rate
        )

        Agent.__init__(
                self,
                game,
                player_number
        )

    # Todo: Something with epsilon
    def determine_piece_to_move(self): 
        (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner, there_is_a_winner), player_i = self.game.get_observation()

        self.dice = deepcopy(dice)
        self.move_pieces = deepcopy(move_pieces)
        self.player_pieces = deepcopy(player_pieces)
        self.enemy_pieces = deepcopy(enemy_pieces)

        self.q_table.on_new_turn(
                self.dice,
                self.player_pieces,
                self.enemy_pieces,
                self.move_pieces
        )

        return self.q_table.get_action(epsilon=0.5)

    def on_finished_move(self):
        if self.game.first_winner_was == self.player_number:
            self.q_table.update_Q_value(reward=1)
        else:
            self.q_table.update_Q_value()

