from DLCF.DLCFtypes import GameStateTemplate
from DLCF.cfBoard import GameState as connectFourGameState
from DLCF.gomokuBoard import GameState as gomokuGameState


def getGameState(game_name: str) -> GameStateTemplate:
    if game_name == "ConnectFour":
        return connectFourGameState
    elif game_name == "Gomoku":
        return gomokuGameState
    else:
        raise Exception(f"Game '{game_name}' not implemented yet.")
