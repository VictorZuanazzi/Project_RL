# Project_RL

This is a group project for the Reinforcement learning course offered in M.Sc Artificial Intelligence 
at the University of Amsterdam.

### Contributors
* [David Speck](https://github.com/Saduras)
* [Masoumeh Bakhtiariziabari](https://github.com/mbakhtiariz)
* [Ruth Wijma](https://github.com/rwq)
* [Victor Zuanazzi](https://github.com/VictorZuanazzi)

## Dependencies

This project uses python 3. To install dependencies run:
```
pip install -r requirements.txt
```

## Problem Statement

There are different types of experience replay, e.g. prioritized experience replay and hindsight experience replay. Compare two or more types of experience replay. Does the ‘winner’ depend on the type of environment?

## Experience Replays
We mainly experimented with three experience replays techniques which are:
```
- Without Experience Replay (TODO) 
- Naive Experience Replay 
- Prioritized Experience Replay
    + rank base
    + proportion base
- Combined Experience Replay
- Adaptive Experience Replay 
  + Adaptive ER
  + Adaptive CER
  + Adaptive rank PER

```
# TODOs
```
1) add without memory exps as baseline
2) experiment on effect of various BATCH SIZEs on each method
3) experiment on effect of various BUFFER SIZEs on each method
```

* [Code](code/)
* [Poster presentation](Poster.pdf)

