import ludopy
import numpy as np

from .agent import Agent

class RandomAgent(Agent):
    def determine_piece_to_move(self):
        (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner, there_is_a_winner), player_i = self.game.get_observation()

        if len(move_pieces) > 0:
            return move_pieces[np.random.randint(0, len(move_pieces))]
        else:
            return -1
