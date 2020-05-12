import ludopy
import numpy as np

from copy import deepcopy

from .q_state import QState

class QTable():
    def __init__(
            self,
            discount_factor,
            learning_rate
    ):
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate

        self.last_state = None
        self.last_action = None

        self.current_table_entry = None

        self.state_dict = {}

    def on_new_turn(
            self,
            dice,
            player_pieces,
            enemy_pieces,
            move_pieces
    ):
        # Get state information
        self.q_state = QState(
                dice,
                player_pieces,
                enemy_pieces
        )
        key = self.q_state.get_key()

        # Initialize table entry if not already present
        if key not in self.state_dict.keys():
            self.current_table_entry = {}

            if len(move_pieces) == 0:
                self.current_table_entry[-1] = 0
            else:
                for i in move_pieces:
                    self.current_table_entry[i] = 0

            self.state_dict[key] = deepcopy(self.current_table_entry)
        else:
            self.current_table_entry = deepcopy(self.state_dict[key])

    def get_action(
            self,
            epsilon=0.1
    ):
        # Determine action
        random_int = np.random.randint(
                0,
                high=100
        )
        if random_int > 100 * epsilon:
            self.action = int(max(
                self.current_table_entry.keys(),
                key=lambda k: self.current_table_entry[k]
            ))
        else:
            entry_list = list(self.current_table_entry)
            self.action = self.current_table_entry.keys()[
                    np.random.randint(
                        0, 
                        high=len(entry_list)
                    )
            ]

        return self.action

    def update_Q_value(
            self,
            reward=0
    ):
        # Update last state's Q value if not the first state
        if self.last_state != None:
            last_key = self.last_state.get_key()
            last_Q = self.state_dict[last_key][self.last_action]

            delta_Q = self.learning_rate * (reward + self.discount_factor * max(list(self.current_table_entry)) - last_Q)
            last_Q += delta_Q

            self.state_dict[last_key][self.last_action] = last_Q

        # Update variables
        self.last_state = self.q_state
        self.last_action = self.action
        self.current_table_entry = None
        self.q_state = None
        self.action = None


                

