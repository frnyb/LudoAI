#!/usr/bin/python3.6

import ludopy
import numpy as np

from agents.random_agent import RandomAgent
from agents.q_learning.q_learning_agent import QLearningAgent

game = ludopy.Game()
there_is_a_winner = False

q_learner = QLearningAgent(
        game,
        0
)
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

counter = 0

while not there_is_a_winner:
    agents[game.current_player].move()

    there_is_a_winner = len(game.game_winners) > 0

    counter += 1

    if counter == 100:
        for i in range(4):
            print(game.get_pieces(seen_from=i))

        game.save_hist_video("hej.mp4")
        exit()


#print("Saving history to numpy file")
#game.save_hist(f"game_history.npy")
#print("Saving game video")
#game.save_hist_video(f"game_video.mp4")
