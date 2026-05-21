import argparse
import csv
import logging
import re
import time
from pathlib import Path

from ai_agent import Agent, BATTLE_SCREEN_TYPES
from game import Game
from main import CHECKPOINT_DIR, should_skip_agent


DEFAULT_EPISODES = 3
DEFAULT_MAX_STEPS = 5000
DEFAULT_SLEEP_SECONDS = 0.1


def checkpoint_sort_key(path: Path) -> tuple[int, int, str]:
    match = re.search(r"_step_(\d+)", path.stem)
    if match:
        return (0, int(match.group(1)), path.name)
    if path.stem.endswith("_latest"):
        return (1, 0, path.name)
    return (2, 0, path.name)


def find_checkpoints(checkpoint_dir: Path) -> list[Path]:
    return sorted(checkpoint_dir.glob("*.pt"), key=checkpoint_sort_key)


def choose_eval_action(agent: Agent, raw_state: dict) -> dict:
    forced_action = agent._forced_transition_action(raw_state)
    if forced_action is not None:
        return forced_action

    state_type = raw_state.get("state_type")
    if state_type in BATTLE_SCREEN_TYPES:
        return agent.battle_agent.choose_action(raw_state, training=False)

    policy_state = {
        "screen_type": state_type,
        "raw_state": raw_state,
    }
    return agent.choose_action(policy_state)


def evaluate_episode(
    game: Game,
    agent: Agent,
    max_steps: int,
    sleep_seconds: float,
    checkpoint_name: str,
    episode_index: int,
) -> dict:
    raw_state = game.reset()
    episode_reward = 0.0
    battle_reward = 0.0
    steps = 0
    battle_steps = 0
    battle_wins = 0
    battle_losses = 0

    while raw_state.get("state_type") != "game_over" and steps < max_steps:
        if should_skip_agent(raw_state):
            action = {"type": "proceed"}
        else:
            action = choose_eval_action(agent, raw_state)

        _, reward, done, info = game.step(action, raw_state)
        reward_details = info.get("reward_details", {})
        episode_reward += reward
        steps += 1

        if reward_details.get("type") == "battle":
            battle_reward += reward_details.get("total", reward)
            battle_steps += 1
            if reward_details.get("result") == "won":
                battle_wins += 1
            if reward_details.get("result") == "lost":
                battle_losses += 1

        next_raw_state = info.get("raw_state")
        print(
            f"{checkpoint_name} episode={episode_index} step={steps} "
            f"{raw_state.get('state_type')} -> {next_raw_state.get('state_type')} "
            f"action={action} reward={reward:.2f} done={done}"
        )
        raw_state = next_raw_state
        if done:
            break

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    run = raw_state.get("run", {})
    return {
        "reward": episode_reward,
        "battle_reward": battle_reward,
        "steps": steps,
        "battle_steps": battle_steps,
        "battle_wins": battle_wins,
        "battle_losses": battle_losses,
        "floor": run.get("floor", 0),
        "act": run.get("act", 0),
        "timed_out": raw_state.get("state_type") != "game_over",
    }


def summarize_episode_results(results: list[dict]) -> dict:
    episode_count = max(1, len(results))
    wins = sum(result["battle_wins"] for result in results)
    losses = sum(result["battle_losses"] for result in results)
    return {
        "episodes": len(results),
        "avg_reward": sum(result["reward"] for result in results) / episode_count,
        "avg_battle_reward": sum(result["battle_reward"] for result in results) / episode_count,
        "avg_steps": sum(result["steps"] for result in results) / episode_count,
        "avg_battle_steps": sum(result["battle_steps"] for result in results) / episode_count,
        "battle_wins": wins,
        "battle_losses": losses,
        "battle_win_rate": wins / max(1, wins + losses),
        "avg_floor": sum(result["floor"] for result in results) / episode_count,
        "max_floor": max((result["floor"] for result in results), default=0),
        "timeouts": sum(1 for result in results if result["timed_out"]),
    }


def evaluate_checkpoint(
    checkpoint_path: Path,
    episodes: int,
    character: int,
    base_url: str,
    timeout: float,
    max_steps: int,
    sleep_seconds: float,
) -> dict:
    agent = Agent()
    agent.battle_agent.load(str(checkpoint_path))
    if agent.battle_agent.model is not None:
        agent.battle_agent.model.eval()

    game = Game(character=character, base_url=base_url, timeout=timeout)
    episode_results = []
    for episode_index in range(1, episodes + 1):
        episode_results.append(
            evaluate_episode(
                game,
                agent,
                max_steps,
                sleep_seconds,
                checkpoint_path.name,
                episode_index,
            )
        )
    summary = summarize_episode_results(episode_results)
    summary.update(
        {
            "checkpoint": checkpoint_path.name,
            "path": str(checkpoint_path),
            "trained_steps": agent.battle_agent.trained_steps,
            "learn_steps": agent.battle_agent.learn_steps,
            "epsilon": agent.battle_agent.epsilon,
            "ending_steps": ",".join(
                str(result["steps"]) for result in episode_results
            ),
        }
    )
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate each saved battle-agent checkpoint without training."
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--character", type=int, default=0)
    parser.add_argument("--base-url", default="http://localhost:15526/api/v1")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP_SECONDS)
    parser.add_argument("--csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    checkpoints = find_checkpoints(args.checkpoint_dir)
    if not checkpoints:
        logging.warning("No checkpoints found in %s", args.checkpoint_dir)
        return

    rows = []
    for checkpoint_path in checkpoints:
        logging.info("Evaluating checkpoint %s", checkpoint_path)
        try:
            result = evaluate_checkpoint(
                checkpoint_path,
                args.episodes,
                args.character,
                args.base_url,
                args.timeout,
                args.max_steps,
                args.sleep,
            )
        except Exception as exc:
            logging.exception("Could not evaluate checkpoint %s: %s", checkpoint_path, exc)
            continue

        rows.append(result)
        print(
            f"{result['checkpoint']}: "
            f"trained_steps={result['trained_steps']} "
            f"learn_steps={result['learn_steps']} "
            f"avg_reward={result['avg_reward']:.2f} "
            f"avg_battle_reward={result['avg_battle_reward']:.2f} "
            f"avg_steps={result['avg_steps']:.1f} "
            f"ending_steps=[{result['ending_steps']}] "
            f"battle_win_rate={result['battle_win_rate']:.2%} "
            f"wins={result['battle_wins']} losses={result['battle_losses']} "
            f"avg_floor={result['avg_floor']:.1f} max_floor={result['max_floor']} "
            f"timeouts={result['timeouts']}"
        )

    if args.csv is not None:
        write_csv(args.csv, rows)
        logging.info("Wrote evaluation results to %s", args.csv)


if __name__ == "__main__":
    main()
