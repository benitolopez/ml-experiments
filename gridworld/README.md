# Q-learning in GridWorld

This project demonstrates Q-learning, a fundamental reinforcement learning algorithm, implemented from scratch in C. The agent learns the optimal path through a GridWorld environment by balancing exploration and exploitation.

The program runs through a specified number of episodes, learns the optimal Q-values, and outputs the optimal path visually in the console after training.

## Example Output

```
Episode 1000 completed.
Episode 2000 completed.
...
Episode 10000 completed.
Goal reached!

Agent's path:
S * * * G
. X X . .
. . . . .
. . X . .
. . . . .
```

- `S`: Start position
- `G`: Goal
- `X`: Obstacle
- `*`: Path taken by the agent
- `.`: Empty cell
