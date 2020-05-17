#!/usr/bin/python3.6

import ludopy
import numpy as np

from agents.random_agent import RandomAgent
from agents.q_learning.q_learning_agent import QLearningAgent

q_table_filename = None

game = ludopy.Game()

q_learner = QLearningAgent(
        game,
        0,
        learning_rate=0.5,
        discount_factor=0.9,
        epsilon=0.5,
        win_reward=10.0,
        lost_reward=-10.0,
        piece_in_reward=5.0,
        land_on_globe_reward=1.0,
        land_on_star_reward=2.0,
        knock_enemy_home_reward=0.5,
        got_knocked_home_reward=-1.1,
        no_move_reward=-0.5,
        piece_number_scale_reward=0.001,
        piece_number_init_func_value=5,
        q_table_filename=q_table_filename
)
print(len(q_learner.q_table.q_table))
random_agent_1 = RandomAgent(
        game,
        1
)
random_agent_2 = RandomAgent(
        game,
        2
)
random_agent_3 = RandomAgent(
        game,
        3
)

agents = [q_learner,random_agent_1,random_agent_2,random_agent_3]

games_won = []

n_games = 0

policy = "explore"

try:
    while True:
        there_is_a_winner = False

        while not there_is_a_winner:
            agents[game.current_player].move()

            there_is_a_winner = len(game.game_winners) > 0

        n_games += 1

        if game.first_winner_was == 0:
            games_won.append(1)
        else:
            games_won.append(0)

        if len(games_won) > 100:
            games_won = games_won[-100:]

        winning_rate = sum(games_won) / len(games_won)

        known_states_rate = sum(q_learner.q_table.known_state_encountered) / len(q_learner.q_table.known_state_encountered)

        print(str(n_games) + ":\tWinning rate last 100: " + str(int(winning_rate * 100)) + "%\tKnown states in round: " + str(int(known_states_rate * 100)) + "%\tVisited states in round: " + str(len(q_learner.q_table.known_state_encountered)))

        game = ludopy.Game()

        q_learner.new_game(game)

        if n_games % 100 == 0:
            q_learner.dump_q_table("qtable_new.json")

        #if n_games % 1000 == 0 and policy == "explore":
        #    policy = "exploit"
        #elif n_games % 100 == 0 and policy == "exploit":
        #    policy = "explore"


        #if policy == "explore":
        #    q_learner.new_game(
        #            game,
        #            epsilon=0.9
        #    )
        #elif policy == "exploit":
        #    q_learner.new_game(
        #            game,
        #            epsilon=0
        #    )

        random_agent_1.new_game(game)
        random_agent_2.new_game(game)
        random_agent_3.new_game(game)
except KeyboardInterrupt:
    print("wtf")
    q_learner.dump_q_table("qtable_new.json")
    exit()

#print("Saving history to numpy file")
#game.save_hist(f"game_history.npy")
#print("Saving game video")
#game.save_hist_video(f"game_video.mp4")
