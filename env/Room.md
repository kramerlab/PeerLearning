# The Room Environment
<p align="center">
  <img width="200" height="200" src="room.png?raw=true">
</p>

This package includes our [room](room.py) grid-world environment displayed in the figure. In this environment, the agents starts in the middle of a squared grid and its task is to reach a specific goal position which is chosen randomly among the border position. The agent observes the position of the goal next to its own position.
The exploration in this task is more complex than in the usual grid-world tasks and satisfies all requirements for our baseline.
In the figure, we display the environment with a small grid size of 5x5 for visualization purposes. In the [experiments](../Experiments.md) for our paper, we use a bigger grid sizes of 
27x27 and 21x21 creating the right amount of challenge for the agent.
