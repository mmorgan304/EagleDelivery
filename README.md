# Eagle Delivery: Project 1 starter

This is our project 1 that focuses on search and reinforcement learning. Edit this readme file with your own project introduction and run commands for your GitHub commits.

## Requirements

Use Python 3.9+.

Install the plotting dependency with:

```bash
python3 -m pip install -r requirements.txt
```

## What already works

```bash
python3 run_project.py show
python3 run_project.py search --algorithm bfs --start S --goal W
python3 run_project.py search --algorithm bfs --map maps/emu_detour.txt --start S --goal W
python3 run_project.py random --start S --goal W --seed 0
python3 examples/interaction_demo.py
python3 -m unittest tests.test_environment -v
```

On `emu_campus.txt`, BFS from S to W should return path length **18** and **67**
expanded nodes, The path visualization uses `o`.

## Your implementation tasks

Edit only these four functions in `agents.py` for the required algorithms:

1. `manhattan(state, goal)`
2. `astar(campus, start, goal)`
3. `epsilon_greedy(q_values, state, actions, epsilon, rng)`
4. `q_learning_update(q_values, state, action, reward, next_state, actions, alpha, gamma, terminal)`


After TODOs 1-2:

```bash
python3 -m unittest tests.test_student.SearchTests -v
python3 run_project.py search --algorithm astar --start S --goal W
```

After TODOs 3-4:

```bash
python3 -m unittest tests.test_student.LearningTests -v
python3 -m unittest discover -v
python3 run_project.py train --start S --goal W --output results/baseline_SW
```

Training saves a `training_steps.png` plot in the output directory, with episode
number on the x-axis and steps per episode on the y-axis. Each training seed is
shown as a separate line, along with a 10-episode rolling average across all
seeds. The plot title includes the start and goal building names.
