# AlphaCF
AlphaGo like Deep Reinforcement learning for connet four and gomoku. While this project uses the general idea of AlphaGo, it implements some extra features and additions to help training and convergence. Some of these features are:
- Wandb logging;
- Multithreaded selfplay;
- Improved actor-critic model architecture;
- Early stopping when actor-critic model starts overfitting;
- A decaying entropy loss term to prevent the actor-critic model from becoming overconfident too early on;
- GAE advantage computation;
- PPO policy loss;
- Among others.

Run `python improve.py --game <game_name> --device <device> --lr <learning_rate> --num-generations 50 --num-games-per-iteration 20000 --num-workers 4` to initialize a new agent and automatically train it for 50 generations. You can also manually do this by using `initAgent.py`, `selfPlay.py`, `trainAgent.py` and `evalAgent.py`.
