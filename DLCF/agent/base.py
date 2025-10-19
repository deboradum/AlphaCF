from typing import List
from DLCF.DLCFtypes import Move
from DLCF.getGameState import GameStateTemplate

class Agent:
    def __init__(self):
        pass

    def select_moves(self, game_states: List[GameStateTemplate]) -> List[Move]:
        raise NotImplementedError()

    def diagnostics(self):
        return {}
