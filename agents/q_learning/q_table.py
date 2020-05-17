import ludopy
import numpy as np

from copy import deepcopy
from math import inf

from .q_state import QState
from .q_reward import QReward

# action = place number of piece, e.g. player_pieces
# piece = index of piece, e.g. move_pieces

class QTable():
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
            q_table=None,
            evaluating=False
    ):
        self.game = game
        self.player_number = player_number

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

        self.evaluating = evaluating

        self.q_table = {}

        self.known_state_encountered = []

        if q_table != None:
            self.q_table = q_table

        self.action, self.piece, self.Q = None, None, None

        self.first = True

    def new_episode(
            self,
            game,
            evaluating=False
    ):
        self._update_last_state_Q(
                self.game.get_pieces(seen_from=self.player_number)[0],
                self.last_player_pieces,
                self.last_enemy_pieces,
                self.piece,
                self.action,
                self.q_state.get_key()
        )

        self.game = game
        self.evaluating = evaluating
        self.known_state_encountered = []
        self.action, self.piece, self.Q = None, None, None

    def on_new_turn(self):
        # Get new observation
        (self.dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner, there_is_a_winner), player_i = self.game.get_observation() 
        dice = self.dice

        if player_i != self.player_number:
            raise Exception("Player numbers doesn't match!")

        # Update last state/action if not first iteration:
        if self.action != None:
            self._update_last_state_Q(
                    player_pieces,
                    self.last_player_pieces,
                    self.last_enemy_pieces,
                    self.piece,
                    self.action,
                    self.q_state.get_key()
            )

        self.q_state = QState(
                self.game,
                self.player_number
        )
        state_key = self.q_state.get_key()

        # Initialize table entry if not already present
        if state_key not in self.q_table.keys():
            self._init_table_entry(
                    player_pieces, 
                    state_key
            )
            self.known_state_encountered.append(0)
        else:
            self.known_state_encountered.append(1)

        # Determine action
        self.action, self.piece, self.Q = self._get_action(
                state_key,
                self.epsilon,
                player_pieces,
                move_pieces=move_pieces
        )

        # Calculate expected maximum next state Q value and expected reward
        self.expected_Q_max, self.expected_rewards = self._get_expected_rewards(
                player_pieces,
                enemy_pieces,
                state_key
        )

        self.game_snap = deepcopy(self.game)

        # Set variables
        self.last_player_pieces = player_pieces
        self.last_enemy_pieces = enemy_pieces

    def _update_last_state_Q(
            self,
            new_player_pieces,
            last_player_pieces,
            last_enemy_pieces,
            piece,
            action,
            last_state_key
    ):
        reward = QReward(
                new_player_pieces,
                last_enemy_pieces,
                self.player_number,
                piece,
                self.game.first_winner_was,
                last_player_pieces,
                win_reward=self.win_reward,
                lost_reward=self.lost_reward,
                piece_in_reward=self.piece_in_reward,
                land_on_globe_reward=self.land_on_globe_reward,
                land_on_star_reward=self.land_on_star_reward,
                knock_enemy_home_reward=self.knock_enemy_home_reward,
                got_knocked_home_reward=self.got_knocked_home_reward,
                no_move_reward=self.no_move_reward,
                piece_number_init_func_value=self.piece_number_init_func_value
        )
        
        action_key = str(action)

        delta_Q = self.learning_rate * (reward.get_reward() + self.discount_factor * self.expected_Q_max[action_key] - self.q_table[last_state_key][action_key])
        self.q_table[last_state_key][action_key] += delta_Q

    def get_move(self):
        return self.piece

    def update_Q_value(self):
        #q_reward = QReward(
        #        self.game.get_pieces(seen_from=self.player_number)[0],
        #        self.game_snap.get_pieces(seen_from=self.player_number)[1],
        #        self.player_number,
        #        self.piece,
        #        self.game.first_winner_was,
        #        win_reward=self.win_reward,
        #        piece_in_reward=self.piece_in_reward,
        #        land_on_globe_reward=self.land_on_globe_reward,
        #        land_on_star_reward=self.land_on_star_reward,
        #        knock_enemy_home_reward=self.knock_enemy_home_reward,
        #        got_knocked_home_reward=self.got_knocked_home_reward,
        #        no_move_reward=self.no_move_reward
        #)

        last_state_key = self.q_state.get_key()
        
        # Update simulated state/action pairs, not the action taken
        for action_key in self.expected_Q_max.keys():
            if action_key == str(self.action):
                continue

            delta_Q = self.learning_rate * (self.expected_rewards[action_key] + self.discount_factor * self.expected_Q_max[action_key] - self.q_table[last_state_key][action_key])
            self.q_table[last_state_key][action_key] += delta_Q

        #delta_Q = self.learning_rate * (q_reward.get_reward() + self.discount_factor * self.expected_Q_max[str(self.action)] - self.Q)

        #self.q_table[last_state_key][str(self.action)] += delta_Q

    def _init_table_entry(
            self,
            player_pieces,
            state_key
    ):
            init_val = self._get_init_Q_val(player_pieces)

            table_entry = {"-1": init_val}

            for p in player_pieces:
                table_entry[str(p)] = init_val

            self.q_table[state_key] = table_entry
    
    def _get_init_Q_val(
            self,
            player_pieces
    ):
        val = 0.0
        for p in player_pieces:
            val += p * self.piece_number_scale_reward
            #val += self.piece_number_init_func_value ** (p / 59)

        return val

    def _get_expected_rewards(
            self,
            player_pieces,
            enemy_pieces,
            state_key
    ):
        dices = [1,2,3,4,5,6]
        if self.evaluating:
            dices = [self.dice]
        expected_Q_max = {"-1": []}
        expected_rewards = {"-1": []}
        for p in player_pieces:
            if p != 59:
                expected_Q_max[str(p)] = []
                expected_rewards[str(p)] = []
                #expected_Q_max[str(p)] = -inf

        for _dice in dices:
            temp_game = deepcopy(self.game)

            dice = _dice
            temp_game.current_dice = dice
            move_pieces = temp_game.players[self.player_number].get_pieces_that_can_move(dice)
            temp_game.current_move_pieces = move_pieces

            if temp_game.current_player != self.player_number:
                raise Exception("Player numbers doesn't match!")

            # Simulate all possible actions:
            pieces_list = [-1]
            if len(move_pieces) > 0:
                pieces_list = move_pieces

            if self.evaluating:
                pieces_list = [self.piece]

            for piece in pieces_list:
                action = -1

                if piece != -1:
                    action = player_pieces[piece]

                temp_temp_game = deepcopy(temp_game)
                temp_temp_game.answer_observation(piece)

                temp_q_state = QState(
                        temp_temp_game,
                        self.player_number
                )
                temp_state_key = temp_q_state.get_key()

                pieces = temp_temp_game.get_pieces(seen_from=self.player_number)
                temp_player_pieces = pieces[0]

                if temp_state_key not in self.q_table.keys():
                    self._init_table_entry(
                            temp_player_pieces,
                            temp_state_key,
                    )
                    self.known_state_encountered.append(0)
                else:
                    self.known_state_encountered.append(1)

                expected_Q_max_temp, _ = self._get_Q_max(
                        temp_state_key,
                        temp_player_pieces
                )

                expected_Q_max[str(action)].append(expected_Q_max_temp)

                reward = QReward(
                        temp_player_pieces,
                        enemy_pieces,
                        self.player_number,
                        piece,
                        temp_temp_game.first_winner_was,
                        win_reward=self.win_reward,
                        lost_reward=self.lost_reward,
                        piece_in_reward=self.piece_in_reward,
                        land_on_globe_reward=self.land_on_globe_reward,
                        land_on_star_reward=self.land_on_star_reward,
                        knock_enemy_home_reward=self.knock_enemy_home_reward,
                        got_knocked_home_reward=self.got_knocked_home_reward,
                        no_move_reward=self.no_move_reward,
                        piece_number_init_func_value=self.piece_number_init_func_value
                )

                expected_rewards[str(action)].append(reward.get_reward())

                #if expected_Q_max_temp > expected_Q_max[str(action)]:
                #    expected_Q_max[str(action)] = expected_Q_max_temp

        expected_Q_max_final = {}
        expected_rewards_final = {}

        for action_key in expected_Q_max.keys():
            Q_max_list = expected_Q_max[action_key]
            if len(Q_max_list) == 0:
                continue

            expected_Q_max_final[action_key] = sum(Q_max_list) / len(Q_max_list)

            rewards_list = expected_rewards[action_key]

            expected_rewards_final[action_key] = sum(rewards_list) / len(rewards_list)




            ## Greedy action:
            #action, piece, _ = self._get_action(
            #        state_key,
            #        0,
            #        player_pieces,
            #        move_pieces=move_pieces
            #)

        return expected_Q_max_final, expected_rewards_final

    def _get_action(
            self,
            state_key,
            epsilon,
            player_pieces,
            move_pieces=[0,1,2,3]
    ):
        # Determine action
        action, piece, Q = None, None, None
        random_int = np.random.randint(
                0,
                high=100
        )
        if random_int >= 100 * epsilon: # Greedy policy
            Q, action = self._get_Q_max(
                    state_key,
                    player_pieces,
                    move_pieces=move_pieces
            )
            if action != -1:
                #piece = list(player_pieces).index(action)
                piece = np.min(np.nonzero(player_pieces == action)[0]) # Equivalent to list.index()
            else:
                piece = -1

        elif len(move_pieces) > 0: # Random policy
            piece = move_pieces[ 
                    np.random.randint( 
                        0, 
                        high=len(move_pieces)
                    ) 
            ]
            action = player_pieces[piece]
            Q = self.q_table[state_key][str(action)]

        else:
            piece, action = -1, -1
            Q = self.q_table[state_key][str(action)]

        return action, piece, Q

    def _get_Q_max(
            self,
            state_key,
            player_pieces,
            move_pieces=[0,1,2,3]
    ):
        Q_max, action = None, None

        if len(move_pieces) == 0:
            action = -1
        else:
            possible_actions = {}
            possible_player_pieces = [player_pieces[p] for p in move_pieces]
            for action in possible_player_pieces:
                action_key = str(action)
                if len(possible_actions) == 0:
                    Q_max = self.q_table[state_key][action_key]
                    possible_actions[action_key] = Q_max
                    continue

                temp_Q = self.q_table[state_key][action_key]

                if temp_Q == Q_max:
                    possible_actions[action_key] = Q_max

                elif temp_Q > Q_max:
                    Q_max = temp_Q
                    possible_actions = {action_key: Q_max}

            #for action_key in self.q_table[state_key].keys():
            #    if action_key == "-1":
            #        continue
            #    if int(action_key) in possible_player_pieces:
            #        if len(possible_actions) == 0:
            #            possible_actions[action_key] = self.q_table[state_key][action_key]
            #        elif self.q_table[state_key][action_key] == possible_actions[list(possible_actions.keys())[0]]:
            #            possible_actions[action_key] = self.q_table[state_key][action_key]
            #        elif self.q_table[state_key][action_key] > possible_actions[list(possible_actions.keys())[0]]:
            #            possible_actions = {action_key: self.q_table[state_key][action_key]}

            if len(possible_actions) > 1:
                action = int(
                        list(possible_actions)[
                            np.random.randint(
                                0,
                                high=len(possible_actions)
                            )
                        ]
                )
            else:
                action = int(list(possible_actions)[0])

        if Q_max == None:
            Q_max = self.q_table[state_key][str(action)]

        #if action == -1:
        #    Q_max = self.q_table[state_key][str(action)]
        #else:
        #    Q_max = possible_actions[str(action)]

        return Q_max, action
