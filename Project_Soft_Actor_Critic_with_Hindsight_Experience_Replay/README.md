# Soft Actor-Crtic (SAC) with Hindsight Experience Replay (HER)

This folder contains our final project for the course "Deep Reinforcement Learning" in the summer term 2022 at the University of Osnabrück.

**Authors**: Arne Klages, Erik Nickel and Jan-Luca Schröder.


In this folder you can find our code, the evaluation and our [final paper](Deep-Reinforcement-Learning/Project_Soft_Actor_Critic_with_Hindsight_Experience_Replay/Soft_Actor_Critic_with_Hindsight_Experience_Replay.pdf).

## Project description
In this project report, we show how HER can be implemented in combination with an Soft Actor-Critic (SAC) agent to solve sparse reward settings. 
Furthermore, we compared two of the most important goal sampling strategies to a new strategy called k final,
Here we wanted to see how final would perform, if it had a parameter k like the strategy future.
This parameter compares the number of addittional relabled samples, which are  added to the replay buffer.
In our experiments, we were able to show that SAC with HER can solve sparse reward problems that can not be solved with SAC alone. 
Moreover, SAC+HER performed better than SAC with reward shaping.
The results of comparing the different goal sampling strategies show that the k final significantly outperforms the simple final
strategy, but still lags behind the future goal sampling strategy.
