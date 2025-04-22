# IKT-460-RL-project: Spacecraft landing using reinforcement learning

### Contents
* [ppo.py](./ppo.py), [sac.py](./sac.py) and [ddpg.py](./ddpg.py): Train landing agent
* [spacecraft.py](./spacecraft.py): The physics, and stuff, of the spacecraft
* [gyms.py](./gyms.py): gyms that are used to train the various RL algorithms for landing. It is interacting with the spaceflight simulator. It has the step function, handling of discrete/continous action space, normalization of observation space, etc..
* [test_agent.py](./test_agent.py): Test landing agent (visualization)
* [environment_renderer.py](./environment_renderer.py): Used by [test_agent.py](./test_agent.py) to render things
* [common.py](./common.py): Settings, and common things™

### Dependencies
- numpy
- pygame
- gymnasium
- pytorch
- optuna