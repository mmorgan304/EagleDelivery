import argparse
import csv
from pathlib import Path
import random
import statistics

import agents
import matplotlib.pyplot as plt
from campus import ACTIONS, BUILDING_NAMES, CampusMap, DeliveryEnvironment


def write_csv(filename, rows):
    with open(filename, "w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_training_steps(training_runs, filename, start_name, goal_name):
    figure, axis = plt.subplots()
    reward_axis = axis.twinx()
    for seed, records in training_runs:
        episodes = [row["episode"] for row in records]
        axis.plot(episodes, [row["steps"] for row in records],
                  label="Seed " + str(seed))
    episode_numbers = [row["episode"] for row in training_runs[0][1]]
    mean_steps = [
        statistics.mean(records[index]["steps"] for _, records in training_runs)
        for index in range(len(episode_numbers))
    ]
    mean_rewards = [
        statistics.mean(records[index]["return"] for _, records in training_runs)
        for index in range(len(episode_numbers))
    ]
    window_size = 10
    if len(mean_steps) >= window_size:
        smoothed_steps = [
            statistics.mean(mean_steps[index - window_size + 1:index + 1])
            for index in range(window_size - 1, len(mean_steps))
        ]
        axis.plot(episode_numbers[window_size - 1:], smoothed_steps,
                  linewidth=2.5, label="10-episode average (all seeds)")
        smoothed_rewards = [
            statistics.mean(mean_rewards[index - window_size + 1:index + 1])
            for index in range(window_size - 1, len(mean_rewards))
        ]
        reward_line, = reward_axis.plot(
            episode_numbers[window_size - 1:], smoothed_rewards,
            color="purple", linewidth=2.5,
            label="10-episode average reward (all seeds)")
    axis.set_xlabel("Episode #")
    axis.set_ylabel("Number of steps")
    reward_axis.set_ylabel("Episode reward")
    axis.set_title("Training Steps and Reward per Episode: " + start_name + " to " +
                   goal_name)
    axis.set_ylim(bottom=0)
    reward_axis.invert_yaxis()
    axis.grid(True, alpha=0.3)
    lines, labels = axis.get_legend_handles_labels()
    if len(mean_steps) >= window_size:
        lines.append(reward_line)
        labels.append(reward_line.get_label())
    axis.legend(lines, labels)
    figure.tight_layout()
    figure.savefig(filename)
    plt.close(figure)


# Run complete evaluation trials
def evaluate(environment, choose_action, episodes, max_steps):
    records = []
    for episode in range(1, episodes + 1):
        state = environment.reset()
        total_reward = 0
        terminal = False
        for step in range(1, max_steps + 1):
            action = choose_action(state)
            state, reward, terminal = environment.step(action)
            total_reward += reward
            if terminal:
                break
        records.append({"episode": episode, "steps": step,
                        "return": total_reward, "success": int(terminal),
                        "truncated": int(not terminal)})
    return records


def summarize(records):
    successful_steps = [row["steps"] for row in records if row["success"]]
    return {
        "success_rate": statistics.mean(row["success"] for row in records),
        "mean_return": statistics.mean(row["return"] for row in records),
        "mean_steps_success": (statistics.mean(successful_steps)
                               if successful_steps else "NA"),
    }


def train_once(campus, start, goal, args, seed):
    environment = DeliveryEnvironment(campus, start, goal, args.slip, seed)
    action_rng = random.Random(seed + 1000)
    q_values = {}  # A fresh table for every seed, task, and condition.
    training_records = []
    for episode in range(1, args.episodes + 1):
        state = environment.reset()
        total_reward = 0
        terminal = False
        for step in range(1, args.max_steps + 1):
            action = agents.epsilon_greedy(q_values, state, ACTIONS,
                                          args.epsilon, action_rng)
            next_state, reward, terminal = environment.step(action)
            agents.q_learning_update(q_values, state, action, reward,
                                     next_state, ACTIONS, args.alpha,
                                     args.gamma, terminal)
            total_reward += reward
            state = next_state
            if terminal:
                break

        training_records.append({"episode": episode, "steps": step,
                                 "return": total_reward,
                                 "success": int(terminal),
                                 "truncated": int(not terminal)})

    # evaluation uses epsilon=0 and frozen Q values with new random streams.
    evaluation_environment = DeliveryEnvironment(campus, start, goal,
                                                 args.slip, seed + 10000)
    evaluation_rng = random.Random(seed + 11000)

    def choose_greedy(state):
        return agents.epsilon_greedy(q_values, state, ACTIONS, 0.0,
                                    evaluation_rng)

    evaluation_records = evaluate(evaluation_environment, choose_greedy,
                                  args.eval_episodes, args.max_steps)
    return training_records, evaluation_records


def main():
    parser = argparse.ArgumentParser(description="Eagle Delivery teaching simulator")
    subparsers = parser.add_subparsers(dest="command", required=True)
    default_map = str(Path(__file__).parent / "maps" / "emu_campus.txt")
    for name in ("show", "search", "random", "train"):
        command = subparsers.add_parser(name)
        command.add_argument("--map", default=default_map)
        command.add_argument("--start", choices=list(BUILDING_NAMES), default="S")
        command.add_argument("--goal", choices=list(BUILDING_NAMES), default="W")
        if name == "search":
            command.add_argument("--algorithm", choices=("bfs", "astar"), default="bfs")
        if name in ("random", "train"):
            command.add_argument("--slip", type=float, default=0.0)
            command.add_argument("--max-steps", type=int, default=200)
            command.add_argument("--episodes", type=int,
                                 default=2000 if name == "train" else 100)
        if name == "random":
            command.add_argument("--seed", type=int, default=0)
        if name == "train":
            command.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
            command.add_argument("--alpha", type=float, default=0.3)
            command.add_argument("--gamma", type=float, default=0.95)
            command.add_argument("--epsilon", type=float, default=0.2)
            command.add_argument("--eval-episodes", type=int, default=100)
            command.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.command in ("random", "train"):
        if not 0 <= args.slip <= 1 or args.max_steps < 1 or args.episodes < 1:
            parser.error("Need slip in [0,1] and positive episode/step counts.")
    if args.command == "train":
        if not 0 < args.alpha <= 1 or not 0 <= args.gamma < 1:
            parser.error("Need alpha in (0,1] and gamma in [0,1).")
        if not 0 <= args.epsilon <= 1 or args.eval_episodes < 1:
            parser.error("Need epsilon in [0,1] and positive evaluation count.")
        if len(set(args.seeds)) != len(args.seeds):
            parser.error("Use distinct seeds.")

    campus = CampusMap(args.map)
    if args.command == "show":
        print(campus.render())
        print("\nSchematic map only. # = blocked, . = walkway; letters are entrances.")
        for letter, state in campus.locations.items():
            print(letter + " = " + BUILDING_NAMES[letter] + " at " + str(state))
        return
    start, goal = campus.location(args.start), campus.location(args.goal)
    print(BUILDING_NAMES[args.start] + " -> " + BUILDING_NAMES[args.goal])

    if args.command == "search":
        algorithm = agents.bfs if args.algorithm == "bfs" else agents.astar
        result = algorithm(campus, start, goal)
        print(campus.render(path=result.path))
        print("Expanded:", result.expanded)
        if result.path is None:
            print("No route exists.")
        else:
            print("Path length:", len(result.path) - 1)
            print("Path:", result.path)
        return

    if args.command == "random":
        environment = DeliveryEnvironment(campus, start, goal, args.slip, args.seed)
        rng = random.Random(args.seed + 1000)
        records = evaluate(environment, lambda state: rng.choice(ACTIONS),
                           args.episodes, args.max_steps)
        print("Random-agent evaluation:", summarize(records))
        return

    output = Path(args.output)
    if output.exists() and any(output.glob("*.csv")):
        raise ValueError("Output directory already has CSV files. Choose a new name.")
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    training_runs = []
    for seed in args.seeds:
        training, evaluation = train_once(campus, start, goal, args, seed)
        write_csv(output / ("training_seed" + str(seed) + ".csv"), training)
        write_csv(output / ("evaluation_seed" + str(seed) + ".csv"), evaluation)
        training_runs.append((seed, training))
        row = {"map": campus.filename, "start": args.start, "goal": args.goal,
               "seed": seed, "episodes": args.episodes,
               "eval_episodes": args.eval_episodes, "max_steps": args.max_steps,
               "alpha": args.alpha, "gamma": args.gamma,
               "epsilon": args.epsilon, "slip": args.slip}
        row.update(summarize(evaluation))
        summaries.append(row)
        print("Seed", seed, "evaluation:", summarize(evaluation))
    write_csv(output / "summary.csv", summaries)
    plot_path = output / "training_steps.png"
    plot_training_steps(training_runs, plot_path, BUILDING_NAMES[args.start],
                        BUILDING_NAMES[args.goal])
    print("Saved CSV files to", output)
    print("Saved training steps plot to", plot_path)
    print("Mean success rate across seeds:",
          statistics.mean(row["success_rate"] for row in summaries))
    print("Mean evaluation return across seeds:",
          statistics.mean(row["mean_return"] for row in summaries))


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as error:
        print("\nStarter TODO:", error)
        print("BFS, map display, random baseline, and environment tests already work.")
        raise SystemExit(1)
