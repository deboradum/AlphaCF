import importlib

__all__ = [
    'Encoder',
    'get_encoder_by_name',
]

class Encoder:
    def name(self):
        raise NotImplementedError()

    def encode(self, game_state):
        raise NotImplementedError()

    def encode_point(self, point):
        raise NotImplementedError()

    def decode_point_index(self, index):
        raise NotImplementedError()

    def num_points(self):
        raise NotImplementedError()

    def shape(self):
        raise NotImplementedError()

def get_encoder_by_name(name, board_size) -> Encoder:
    if isinstance(board_size, int):
        if isinstance(board_size, int):
            raise ValueError("Board size should be a tuple (h, w)")

    module = importlib.import_module('DLCF.encoders.' + name)
    constructor = getattr(module, 'create')

    return constructor(board_size)
