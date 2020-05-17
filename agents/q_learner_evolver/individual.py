import ludopy
import numpy as np

import os
import json

from ..q_learning.q_learning_agent import QLearningAgent
from ..q_learning.q_agent import QAgent
from ..random_agent import RandomAgent

discount_factor_bounds = [0.0, 1.0]
learning_rate_bounds = [0.01, 1.0]            
epsilon_bounds = [0.1, 1.0]
win_reward_bounds = [1.0, 10.0]
lost_reward_bounds = [-10.0, 0.0]
piece_in_reward_bounds = [0.0, 10.0]
land_on_globe_reward_bounds = [0.0, 5.0]
land_on_star_reward_bounds = [0.0, 5.0]
knock_enemy_home_reward_bounds = [0.0, 5.0]
got_knocked_home_reward_bounds = [-5.0, 0.0]
no_move_reward_bounds = [-5.0, 0.0]
piece_number_scale_reward_bounds = [0.0, 5.0]
piece_number_init_func_value_bounds = [0.0, 100.0]

class Individual():
    def __init__(
            self,
            individual_id,
            discount_factor=None,
            learning_rate=None,            
            epsilon=None,
            win_reward=None,
            lost_reward=None,
            piece_in_reward=None,
            land_on_globe_reward=None,
            land_on_star_reward=None,
            knock_enemy_home_reward=None,
            got_knocked_home_reward=None,
            no_move_reward=None,
            piece_number_scale_reward=None,
            piece_number_init_func_value=None,
            mutation_rate=0
    ):
        self.id = individual_id

        random_int = 100
        if mutation_rate > 0:
            random_int = np.random.randint(
                    0,
                    high=100
            )
        if discount_factor == None or random_int < 100 * mutation_rate:
            self.discount_factor = np.random.uniform(discount_factor_bounds[0],discount_factor_bounds[1])
        else:
            self.discount_factor = discount_factor

        random_int = 100
        if mutation_rate > 0:
            random_int = np.random.randint(
                    0,
                    high=100
            )
        if learning_rate == None or random_int < 100 * mutation_rate:
            self.learning_rate = np.random.uniform(learning_rate_bounds[0],learning_rate_bounds[1])
        else:
            self.learning_rate = learning_rate

        random_int = 100
        if mutation_rate > 0:
            random_int = np.random.randint(
                    0,
                    high=100
            )
        if epsilon == None or random_int < 100 * mutation_rate:
            self.epsilon = np.random.uniform(epsilon_bounds[0],epsilon_bounds[1])
        else:
            self.epsilon = epsilon

        random_int = 100
        if mutation_rate > 0:
            random_int = np.random.randint(
                    0,
                    high=100
            )
        if win_reward == None or random_int < 100 * mutation_rate:
            self.win_reward = np.random.uniform(win_reward_bounds[0],win_reward_bounds[1])
        else:
            self.win_reward = win_reward

        random_int = 100
        if mutation_rate > 0:
            random_int = np.random.randint(
                    0,
                    high=100
            )
        if lost_reward == None or random_int < 100 * mutation_rate:
            self.lost_reward = np.random.uniform(lost_reward_bounds[0],lost_reward_bounds[1])
        else:
            self.lost_reward = lost_reward

        random_int = 100
        if mutation_rate > 0:
            random_int = np.random.randint(
                    0,
                    high=100
            )
        if piece_in_reward == None or random_int < 100 * mutation_rate:
            self.piece_in_reward = np.random.uniform(piece_in_reward_bounds[0],piece_in_reward_bounds[1])
        else:
            self.piece_in_reward = piece_in_reward

        random_int = 100
        if mutation_rate > 0:
            random_int = np.random.randint(
                    0,
                    high=100
            )
        if land_on_globe_reward == None or random_int < 100 * mutation_rate:
            self.land_on_globe_reward = np.random.uniform(land_on_globe_reward_bounds[0],land_on_globe_reward_bounds[1])
        else:
            self.land_on_globe_reward = land_on_globe_reward

        random_int = 100
        if mutation_rate > 0:
            random_int = np.random.randint(
                    0,
                    high=100
            )
        if land_on_star_reward == None or random_int < 100 * mutation_rate:
            self.land_on_star_reward = np.random.uniform(land_on_star_reward_bounds[0],land_on_star_reward_bounds[1])
        else:
            self.land_on_star_reward = land_on_star_reward

        random_int = 100
        if mutation_rate > 0:
            random_int = np.random.randint(
                    0,
                    high=100
            )
        if knock_enemy_home_reward == None or random_int < 100 * mutation_rate:
            self.knock_enemy_home_reward = np.random.uniform(knock_enemy_home_reward_bounds[0],knock_enemy_home_reward_bounds[1])
        else:
            self.knock_enemy_home_reward = knock_enemy_home_reward

        random_int = 100
        if mutation_rate > 0:
            random_int = np.random.randint(
                    0,
                    high=100
            )
        if got_knocked_home_reward == None or random_int < 100 * mutation_rate:
            self.got_knocked_home_reward = np.random.uniform(got_knocked_home_reward_bounds[0],got_knocked_home_reward_bounds[1])
        else:
            self.got_knocked_home_reward = got_knocked_home_reward

        random_int = 100
        if mutation_rate > 0:
            random_int = np.random.randint(
                    0,
                    high=100
            )
        if no_move_reward == None or random_int < 100 * mutation_rate:
            self.no_move_reward = np.random.uniform(no_move_reward_bounds[0],no_move_reward_bounds[1])
        else:
            self.no_move_reward = no_move_reward

        random_int = 100
        if mutation_rate > 0:
            random_int = np.random.randint(
                    0,
                    high=100
            )
        if piece_number_scale_reward == None or random_int < 100 * mutation_rate:
            self.piece_number_scale_reward = np.random.uniform(piece_number_scale_reward_bounds[0],piece_number_scale_reward_bounds[1])
        else:
            self.piece_number_scale_reward = piece_number_scale_reward

        random_int = 100
        if mutation_rate > 0:
            random_int = np.random.randint(
                    0,
                    high=100
            )
        if piece_number_init_func_value == None or random_int < 100 * mutation_rate:
            self.piece_number_init_func_value = np.random.uniform(piece_number_init_func_value_bounds[0],piece_number_init_func_value_bounds[1])
        else:
            self.piece_number_init_func_value = piece_number_init_func_value

        self.game = ludopy.Game()

        self.q_learner = QLearningAgent(
                self.game,
                0,
                self.discount_factor,
                self.learning_rate,
                self.epsilon,
                self.win_reward,
                self.lost_reward,
                self.piece_in_reward,
                self.land_on_globe_reward,
                self.land_on_star_reward,
                self.knock_enemy_home_reward,
                self.got_knocked_home_reward,
                self.no_move_reward,
                self.piece_number_scale_reward,
                self.piece_number_init_func_value
        )


        self.agents = [self.q_learner]

        for i in range(1,4):
            self.agents.append(
                    RandomAgent(
                        self.game,
                        i
                    )
            )

        self.games_won = []
        self.n_games = 0
        self.winning_rates = []
        self.known_state_rates = []

        self.fitness = None

    def train(
            self,
            training_iterations=10000,
            win_rate_iterations=100
    ):
        for i in range(training_iterations):
            there_is_a_winner = False

            while not there_is_a_winner:
                self.agents[self.game.current_player].move()

                there_is_a_winner = len(self.game.game_winners) > 0

            self.n_games += 1

            if self.game.first_winner_was == 0:
                self.games_won.append(1)
            else:
                self.games_won.append(0)

            if len(self.games_won) > win_rate_iterations:
                self.games_won = self.games_won[-win_rate_iterations:]

            self.winning_rates.append(sum(self.games_won) / len(self.games_won))
            self.known_state_rates.append(sum(self.q_learner.q_table.known_state_encountered) / len(self.q_learner.q_table.known_state_encountered))

            if self.n_games % 1000 == 0:
                self.q_learner.dump_q_table(str(self.id) + "/qtable.json")

            self.game = ludopy.Game()

            if i == training_iterations - 1:
                self.q_learner.new_game(
                        self.game,
                        epsilon=0
                )
            else:
                self.q_learner.new_game(self.game)

            for i in range(1,4):
                self.agents[i].new_game(self.game)

    def evaluate_fitness(
            self,
            evaluation_iterations=500
    ):
        self.q_agent = QAgent(
                self.game,
                self.q_learner.q_table.q_table
        )

        #self.q_agent.q_table.evaluating = True

        self.agents[0] = self.q_agent

        self.eval_games_won = []

        for i in range(evaluation_iterations):
            there_is_a_winner = False

            while not there_is_a_winner:
                self.agents[self.game.current_player].move()

                there_is_a_winner = len(self.game.game_winners) > 0

            if self.game.first_winner_was == 0:
                self.eval_games_won.append(1)
            else:
                self.eval_games_won.append(0)

            self.game = ludopy.Game()

            self.q_agent.new_game(self.game)

            for i in range(1,4):
                self.agents[i].new_game(self.game)

        self.fitness = sum(self.eval_games_won) / len(self.eval_games_won)

        return self.fitness

    def save(
            self,
            directory=None
    ):
        info = {
                "id": self.id,
                "fitness": self.fitness,
                "eval_games_won": self.eval_games_won,
                "winning_rate": self.winning_rates,
                "known_states_rate": self.known_state_rates,
                "discount_factor": self.discount_factor,
                "learning_rate": self.learning_rate,
                "epsilon": self.epsilon,
                "win_reward": self.win_reward,
                "lost_reward": self.lost_reward,
                "piece_in_reward": self.piece_in_reward,
                "land_on_globe_reward": self.land_on_globe_reward,
                "land_on_star_reward": self.land_on_star_reward,
                "knock_enemy_home_reward": self.knock_enemy_home_reward,
                "got_knocked_home_reward": self.got_knocked_home_reward,
                "no_move_reward": self.no_move_reward,
                "piece_number_scale_reward": self.piece_number_scale_reward,
                "piece_number_init_func_value": self.piece_number_init_func_value
        }

        _directory = ""

        if directory != None:
            _directory = directory + "/"

        os.mkdir(_directory + str(self.id))

        with open(_directory + str(self.id) + "/info.json", "+w") as f:
            json.dump(
                    info,
                    f,
                    indent=2
            )

        self.q_learner.dump_q_table(_directory + str(self.id) + "/qtable.json")


