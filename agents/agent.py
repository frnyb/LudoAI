import ludopy

class Agent():
    def __init__(
            self,
            game
    ):
        self.game = game

    def move(self):
        piece_to_move = self.determine_piece_to_move()
        self.game.answer_observation(piece_to_move)
        self.on_finished_move()

    def on_finished_move(self):
        pass
