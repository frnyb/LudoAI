import ludopy
import numpy as np

from copy import deepcopy
import json

from .q_learning_agent import QLearningAgent
from ..agent import Agent
from .q_table import QTable
from .q_state import QState

class QAgent(Agent):
    def __init__(
            self,
            game,
            q_table_dict
    ):
        self.game = game
        self.q_table_dict = q_table_dict

        self.first = True

    def new_game(
            self,
            game
    ):
        Agent.new_game(
                self,
                game
        )

    def determine_piece_to_move(self):
        (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner, there_is_a_winner), player_i = self.game.get_observation() 

        state = QState(
                self.game,
                player_i
        )

        state_key = state.get_key()

        piece = None
        
        if len(move_pieces) == 0:
            piece = -1
        elif state_key in self.q_table_dict.keys():
            actions = [player_pieces[m] for m in move_pieces]
            action = int(
                    max(
                        actions,
                        key=lambda k: self.q_table_dict[state_key][str(k)]
                    )
            )
            piece = list(player_pieces).index(action)
        else:
            piece = move_pieces[np.random.randint(
                0,
                high=len(move_pieces)
            )]

        return piece

    def on_finished_move(self):
        pass

