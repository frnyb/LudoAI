import ludopy
import numpy as np

class QState():
    def __init__(
            self,
            dice,
            player_pieces,
            enemy_pieces
    ):
        player_pieces = np.sort(player_pieces)
        enemy_pieces = np.concatenate((
            np.sort(enemy_pieces[0]),
            np.sort(enemy_pieces[1]),
            np.sort(enemy_pieces[2])
        ))

        self.state = tuple(
                np.concatenate((
                    np.array([dice]), 
                    player_pieces, 
                    enemy_pieces
                ))
        )

    def get_key(self):
        return hash(self.state)


