import ludopy
import numpy as np

from copy import deepcopy

from .agent import Agent


class RandomAgent(Agent):
    def determine_piece_to_move(self):
        (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner, there_is_a_winner), player_i = self.game.get_observation()

        self.dice = deepcopy(dice)
        self.move_pieces = deepcopy(move_pieces)
        self.player_pieces = deepcopy(player_pieces)
        self.enemy_pieces = deepcopy(enemy_pieces)

        if len(move_pieces) > 0:
            return move_pieces[np.random.randint(0, len(move_pieces))]
        else:
            return -1
