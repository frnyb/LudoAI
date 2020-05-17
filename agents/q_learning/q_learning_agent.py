import ludopy
import numpy as np

from copy import deepcopy
import json

from ..agent import Agent
from .q_table import QTable

class QLearningAgent(Agent):
    def __init__(
            self,
            game,
            player_number,
            discount_factor=1.0,
            learning_rate=0.1,
            epsilon=0.1,
            win_reward=10.0,
            lost_reward=-10.0,
            piece_in_reward=5.0,
            land_on_globe_reward=1.0,
            land_on_star_reward=2.0,
            knock_enemy_home_reward=0.9,
            got_knocked_home_reward=-1.1,
            no_move_reward=-0.5,
            piece_number_scale_reward=0.001,
            piece_number_init_func_value=5,
            q_table_filename=None,
            q_table=None
    ):
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.win_reward = win_reward
        self.lost_reward = lost_reward
        self.piece_in_reward = piece_in_reward
        self.land_on_globe_reward = land_on_globe_reward
        self.land_on_star_reward = land_on_star_reward
        self.knock_enemy_home_reward = knock_enemy_home_reward
        self.got_knocked_home_reward = got_knocked_home_reward
        self.no_move_reward = no_move_reward
        self.piece_number_scale_reward = piece_number_scale_reward
        self.piece_number_init_func_value = piece_number_init_func_value

        q_table = None

        if q_table != None:
            self.q_table = q_table

        else:
            if q_table_filename != None:
                with open(q_table_filename, "r") as f:
                    q_table = json.load(f)

            self.q_table = QTable(
                    game,
                    player_number,
                    discount_factor,
                    learning_rate,
                    epsilon,
                    win_reward,
                    lost_reward,
                    piece_in_reward,
                    land_on_globe_reward,
                    land_on_star_reward,
                    knock_enemy_home_reward,
                    got_knocked_home_reward,
                    no_move_reward,
                    piece_number_scale_reward,
                    piece_number_init_func_value,
                    q_table
            )

        Agent.__init__(
                self,
                game,
                player_number
        )

    def new_game(
            self,
            game,
            discount_factor=None,
            learning_rate=None,
            epsilon=None,
            win_reward=None,
            piece_in_reward=None,
            land_on_globe_reward=None,
            land_on_star_reward=None,
            knock_enemy_home_reward=None,
            got_knocked_home_reward=None,
            no_move_reward=None,
            piece_number_scale_reward=None
    ):
        if discount_factor != None:
            self.discount_factor = discount_factor
        if learning_rate != None:
            self.learning_rate = learning_rate
        if epsilon != None:
            self.epsilon = epsilon
        if win_reward != None:
            self.win_reward = win_reward
        if piece_in_reward != None:
            self.piece_in_reward = piece_in_reward
        if land_on_globe_reward != None:
            self.piece_in_reward = piece_in_reward
        if land_on_star_reward != None:
            self.land_on_star_reward = land_on_star_reward
        if knock_enemy_home_reward != None:
            self.knock_enemy_home_reward = knock_enemy_home_reward
        if got_knocked_home_reward != None:
            self.got_knocked_home_reward = got_knocked_home_reward
        if no_move_reward != None:
            self.no_move_reward = no_move_reward
        if piece_number_scale_reward != None:
            self.piece_number_scale_reward = piece_number_scale_reward

        self.q_table.new_episode(game)

        Agent.new_game(
                self,
                game
        )

    def determine_piece_to_move(self):
        self.q_table.on_new_turn()

        return self.q_table.get_move()

    def on_finished_move(self):
        self.q_table.update_Q_value()

    def dump_q_table(
            self,
            filename
    ):
        with open(filename, "+w") as f:
            json.dump(
                    self.q_table.q_table, 
                    f,
                    indent=2
            )

