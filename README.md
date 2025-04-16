# ikt-460-RL-project

### Contents
* [ppo.py](./ppo.py): Train a landing agent using proximal policy otimization
* [ddpg.py](./ddpg.py): Train landing agent using deep deterministic policy gradient
* [sac.py](./sac.py): Train landing agent using soft-actor-critic
* [spacecraft.py](./spacecraft.py): The spacecraft simulator engine
* [gyms.py](./gyms.py): gyms that are used to train the various RL algorithms for landing. It is interacting with the spaceflight simulator. It has the step function, handling of discrete/continous action space, normalization of observation space, etc.. It is compatible with stable baselines 3
* [test_agent.py](./test_agent.py): Test agents in environment (visualization)
* [environment_renderer.py](./environment_renderer.py): Used by [test_agent.py](./test_agent.py) to render things
* [common.py](./common.py): Settings, and things that are used in various locations

### Dependencies
- numpy
- pygame
- gymnasium
- pytorch
- optuna