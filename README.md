# IKT-460-RL-project: Autonomous Spacecraft landing using reinforcement learning

### Contents
* [ppo.py](./ppo.py), [sac.py](./sac.py) and [ddpg.py](./ddpg.py): Train landing agent++
* [spacecraft.py](./spacecraft.py): The physics, and stuff, of the spacecraft
* [gyms.py](./gyms.py): Contains the landing gym. I initially planned to also include a hovering gym, but decided to remove it to lessen the scope of the project.
* [demo.py](./demo.py): Test landing agents (visualization)
* [environment_renderer.py](./environment_renderer.py): Used in [demo.py](./demo.py) to render things
* [common.py](./common.py): Settings, and common things™

### You probably need to install these
- numpy
- pygame
- gymnasium
- pytorch
- optuna

### Acknowledgements
I woud like to thank Seungeun Rho for his minimalistic implementation of SAC, DDPG and PPO (https://github.com/seungeunrho/minimalRL). I used these implementations as a starting point, and found them to be easy to understand and adapt to my project. Awesome stuff!