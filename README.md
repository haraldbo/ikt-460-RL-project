# ikt-460-RL-project

### Files
* [dqn.py](./dqn.py): 
* [ppo.py](./ppo.py): 
* [ddpg.py](./ddpg.py):
* [spacecraft.py](./spacecraft.py): The spaceflight simulator
* [gyms.py](./gyms.py): gyms that are used to train the various RL algorithms for hovering/landing. It is interacting with the spaceflight simulator. It has the step function, handling of discrete/continous action space, normalization of observation space, etc..
* [test_agent.py](./test_agent.py): Test agent in environment (visualization)
* [environment_renderer.py](./environment_renderer.py): Used by [test_agent.py](./test_agent.py) to render the environment
* [common.py](./common.py): Settings, interfaces that are used in various locations

### Useful things
* To monitor training: `tensorboard --logdir tensorboard`
* Run tensorboard on remote/local: https://stackoverflow.com/a/40413202/6210364


## Testing