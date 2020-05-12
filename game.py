#!/usr/bin/python3.6

import ludopy
import numpy as np

from agents.random_agent import RandomAgent

game = ludopy.Game()
there_is_a_winner = False

agents = [RandomAgent(game),RandomAgent(game),RandomAgent(game),RandomAgent(game)]

while not there_is_a_winner:
    agents[game.current_player].move()

    there_is_a_winner = len(game.game_winners) > 0

print("Saving history to numpy file")
game.save_hist(f"game_history.npy")
print("Saving game video")
game.save_hist_video(f"game_video.mp4")
