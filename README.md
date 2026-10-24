# Eagle Delivery

Eagle Delivery is a small campus-delivery simulator for exploring search and
reinforcement learning. A delivery agent navigates a grid map from a starting
building to a goal. The project includes breadth-first search (BFS), A* search,
a random baseline, and Q-learning experiments.

## Setup

Use Python 3.9 or newer. Install the plotting dependency with:

```bash
python -m pip install -r requirements.txt
```

## Run the project

```bash
python run_project.py show
python run_project.py search --algorithm bfs --start S --goal W
python run_project.py search --algorithm astar --map "maps/my campus.txt" --start S --goal W
python run_project.py random --start S --goal W --seed 0
python run_project.py train --start S --goal W --output results/baseline_SW
```

Map cells use `#` for walls, `.` for walkable paths, and `S`, `L`, `H`, `D`,
`P`, and `W` for building entrances. The default map is
`maps/emu_campus.txt`.

## Settings

All commands accept `--map`, `--start`, and `--goal`. The default start and
goal are `S` (Student Center) and `W` (Walton Hall).

| Setting | Applies to | Default | Description |
| --- | --- | --- | --- |
| `--algorithm` | `search` | `bfs` | Search method: `bfs` or `astar`. |
| `--slip` | `random`, `train` | `0.0` | Probability that a movement does not follow the chosen action; from 0 to 1. |
| `--episodes` | `random`, `train` | `100` / `2000` | Number of episodes. |
| `--max-steps` | `random`, `train` | `200` | Step limit per episode. |
| `--seed` | `random` | `0` | Random baseline seed. |
| `--seeds` | `train` | `0 1 2` | Independent training seeds. |
| `--alpha` | `train` | `0.3` | Q-learning learning rate. |
| `--gamma` | `train` | `0.95` | Discount factor. |
| `--epsilon` | `train` | `0.2` | Probability of choosing a random action during training. |
| `--eval-episodes` | `train` | `100` | Greedy evaluation episodes per training seed. |
| `--output` | `train` | Required | Directory for experiment results. |

## Training experiments

Training uses Q-learning with epsilon-greedy exploration. Each seed starts with
a fresh Q-table. After training, the agent is evaluated with exploration turned
off; evaluation uses separate random streams. Results include per-seed training
and evaluation CSV files, a summary CSV, and `training_steps.png`. The plot
shows steps for each seed, the 10-episode average steps across seeds, and the
10-episode average reward across seeds (purple, on an inverted secondary axis).
Its title includes the selected start and goal.

Change experiment parameters on the command line, for example:

```bash
python run_project.py train --map "maps/my campus.txt" --start S --goal W --episodes 2000 --max-steps 200 --seeds 0 1 2 --alpha 0.3 --gamma 0.95 --epsilon 0.2 --slip 0.1 --eval-episodes 100 --output results/my_campus_SW
```

Run the test suite with:

```bash
python -m unittest discover -v
```
