![TETRIS RL](https://github.com/jaybutera/tetris-environment/blob/master/tetrisRL_logo.png)

## Installation
```bash
$ git clone https://github.com/jaybutera/tetrisRL
```

## Usage

The interface is similar to an [OpenAI Gym](https://gym.openai.com/docs) environment. 

Initialize the Tetris RL environment

```python
from engine import TetrisEngine

width, height = 10, 20
env = TetrisEngine(width, height)
```

Simulation loop
```python
# Reset the environment
obs = env.clear()

while True:
    # Get an action from a theoretical AI agent
    action = agent(obs)

    # Sim step takes action and returns results
    obs, reward, done = env.step(action)

    # Done when game is lost
    if done:
        break
```
