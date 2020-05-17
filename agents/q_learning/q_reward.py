import ludopy
import numpy as np

class QReward():
    def __init__(
            self,
            new_player_pieces,
            last_enemy_pieces,
            player_number,
            last_move,
            first_winner_was,
            last_player_pieces=None,
            win_reward=10.0,
            lost_reward=-10.0,
            piece_in_reward=5.0,
            land_on_globe_reward=1.0,
            land_on_star_reward=2.0,
            knock_enemy_home_reward=0.9,
            got_knocked_home_reward=-1.1,
            no_move_reward=-0.5,
            piece_number_init_func_value=5
    ):
        self.new_player_pieces = new_player_pieces
        self.last_enemy_pieces = last_enemy_pieces
        self.player_number = player_number
        self.last_move = last_move
        self.first_winner_was = first_winner_was

        self.last_player_pieces = last_player_pieces

        self.win_reward = win_reward
        self.lost_reward = lost_reward
        self.piece_in_reward = piece_in_reward
        self.land_on_globe_reward = land_on_globe_reward
        self.land_on_star_reward = land_on_star_reward
        self.knock_enemy_home_reward = knock_enemy_home_reward
        self.got_knocked_home_reward = got_knocked_home_reward
        self.no_move_reward = no_move_reward
        self.piece_number_init_func_value = piece_number_init_func_value

    def _determine_events(self):
        win = False
        lost = False
        piece_in = False
        land_on_globe = False
        land_on_star = False
        knock_enemy_home = False
        got_knocked_home = False
        no_move = False

        for i in range(3):
            for j, p in enumerate(self.last_enemy_pieces[i]):
                if p == 0 or p >= 54:
                    self.last_enemy_pieces[i][j] = p
                else:
                    self.last_enemy_pieces[i][j] = (p + 13 * (i + 1)) % 53
                    
                if self.last_enemy_pieces[i][j] < p:
                    self.last_enemy_pieces[i][j] += 1

        self.new_piece = -1

        if self.last_move != -1:
            self.new_piece = self.new_player_pieces[self.last_move]

        new_piece = self.new_piece

        if new_piece != -1:
            if self.first_winner_was == self.player_number:
                win = True
            elif self.first_winner_was != -1:
                lost = True

            if new_piece == 53:
                new_piece = 1

            if new_piece == 59:
                piece_in = True

            elif new_piece == 0:
                got_knocked_home = True

            elif new_piece >= 54:
                pass

            elif np.any(
                    np.isin(
                        new_piece,
                        self.last_enemy_pieces[0]
                    )
            ):
                if (
                        new_piece == 9 or
                        new_piece == 14 or
                        new_piece == 22 or
                        new_piece == 35 or
                        new_piece == 48
                ):
                    got_knocked_home = True
                
                else:
                    knock_enemy_home = True

            elif np.any(
                    np.isin(
                        new_piece,
                        self.last_enemy_pieces[1]
                    )
            ):
                if (
                        new_piece == 9 or 
                        new_piece == 22 or
                        new_piece == 27 or
                        new_piece == 35 or
                        new_piece == 48
                ):
                    got_knocked_home = True
                
                else:
                    knock_enemy_home = True

            elif np.any(
                    np.isin(
                        new_piece,
                        self.last_enemy_pieces[2]
                    )
            ):
                if (
                        new_piece == 9 or
                        new_piece == 22 or
                        new_piece == 35 or
                        new_piece == 40 or
                        new_piece == 48
                ):
                    got_knocked_home = True
                
                else:
                    knock_enemy_home = True

            elif (
                    new_piece == 1 or
                    new_piece == 9 or
                    new_piece == 22 or
                    new_piece == 35 or
                    new_piece == 48 or
                    new_piece == 53
            ):
                land_on_globe = True

            if (
                    new_piece == 5 or
                    new_piece == 12 or
                    new_piece == 18 or
                    new_piece == 25 or
                    new_piece == 31 or
                    new_piece == 38 or
                    new_piece == 44 or
                    new_piece == 51
            ):
                land_on_star = True

        else:
            no_move = True

        return win, lost, piece_in, land_on_globe, land_on_star, knock_enemy_home, got_knocked_home, no_move

    def get_reward(self):
        win, lost, piece_in, land_on_globe, land_on_star, knock_enemy_home, got_knocked_home, no_move = self._determine_events()

        reward = 0

        if win:
            reward += self.win_reward

        if lost:
            reward += self.lost_reward

        if piece_in:
            reward += self.piece_in_reward

        if land_on_globe:
            reward += self.land_on_globe_reward

        if land_on_star:
            reward += self.land_on_star_reward

        if knock_enemy_home:
            reward += self.knock_enemy_home_reward

        if got_knocked_home:
            reward += self.got_knocked_home_reward

        if no_move:
            reward += self.no_move_reward

        #for p in self.player_pieces:
        #    reward += self.piece_number_scale_reward * p

        if self.last_player_pieces is not None:
            for i in range(4):
                if got_knocked_home and i == self.last_move:
                    continue

                if (
                        self.last_player_pieces[i] != 0 and 
                        self.new_player_pieces[i] == 0
                ):
                    reward += self.got_knocked_home_reward

        if self.new_piece != -1:
            reward *= self.piece_number_init_func_value ** (self.new_piece / 59)

        return reward



