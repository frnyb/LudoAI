import ludopy

class Agent():
    def __init__(
            self,
            game,
            player_number
    ):
        self.game = game
        self.player_number = player_number

    def move(self):
        piece_to_move = self.determine_piece_to_move()
        self.game.answer_observation(piece_to_move)
        self.on_finished_move()

    def on_finished_move(self):
        pass
