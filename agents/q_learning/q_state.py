import ludopy
import numpy as np

class QState():
    def __init__(
            self,
            game,
            player_number
    ):
        pieces = game.get_pieces(seen_from=player_number)

        player_pieces = np.sort(pieces[0])
        enemy_pieces = np.sort(
                np.concatenate((
                    pieces[1][0],
                    pieces[1][1],
                    pieces[1][2]
                ))
        )

        self.state = self._determine_state(
                player_pieces,
                enemy_pieces
        )

        #self.key = self.get_key(self.state)
        self.key = str(self.state)

    def get_key(
            self,
            state=None
    ):
        #if state != None:
        #    return str(hash(self.state))
        #else:
        return self.key

    def _determine_state(
            self,
            player_pieces,
            enemy_pieces
    ):
        temp_enemy_pieces = []
        for i, p in enumerate(enemy_pieces):
            if p == 0 or p >= 54:
                pass
            else:
                temp_enemy_pieces.append((p + 13 * (i + 1)) % 53)
                
                if temp_enemy_pieces[-1] < p:
                    temp_enemy_pieces[-1] += 1

        intervals = []

        for p in player_pieces:
            if p == 0 or p >= 54:
                pass

            bounds = []

            if (
                    p == 5 or
                    p == 12 or
                    p == 18 or
                    p == 25 or
                    p == 31 or
                    p == 38 or
                    p == 44 or
                    p == 51
            ):
                bounds = [p-7,p+7]

            else:
                bounds = [p-6,p+6]

            if bounds[0] < 1:
                bounds[0] = 53 + bounds[0]
            if bounds[1] > 53:
                bounds[1] = 53

            intervals.append(bounds)

        enemy_pieces = []

        for p in temp_enemy_pieces:
            within = False
            for bounds in intervals:
                if bounds[0] <= p and p <= bounds[1]:
                    enemy_pieces.append(p)
                    within = True
                    break

            if within:
                continue

        state = list(player_pieces)
        if len(enemy_pieces) > 0:
            state = state + list(enemy_pieces)

        return tuple(state)

