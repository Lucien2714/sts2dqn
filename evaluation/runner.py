import csv
import math
import time
from pathlib import Path

from ai_agent import Agent, BATTLE_SCREEN_TYPES
from game import Game
from main import should_skip_agent

from evaluation.dashboard import LiveEvaluationDashboard
from evaluation.seeds import evaluation_seed


try:
    import torch
except ImportError:
    torch = None


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


def current_q_values(agent: Agent, raw_state: dict, selected_action: dict | None = None) -> dict:
    state_type = raw_state.get("state_type")
    if state_type not in BATTLE_SCREEN_TYPES:
        return {
            "available": False,
            "reason": f"No Q model is used for screen_type={state_type}",
            "screen_type": state_type,
            "actions": [],
        }

    battle_agent = agent.battle_agent
    if torch is None or battle_agent.model is None:
        return {
            "available": False,
            "reason": "Torch/model is not available",
            "screen_type": state_type,
            "actions": [],
        }

    action_mask = battle_agent.valid_action_mask(raw_state)
    state_vector = battle_agent.encode_state(raw_state, action_mask)
    with torch.no_grad():
        state_tensor = torch.tensor(
            state_vector,
            dtype=torch.float32,
            device=battle_agent.device,
        ).unsqueeze(0)
        q_tensor = battle_agent.model(state_tensor).squeeze(0).detach().cpu()

    selected_action_id = None
    if selected_action is not None:
        try:
            selected_action_id = battle_agent.get_game_action_id(selected_action, raw_state)
        except (KeyError, ValueError):
            selected_action_id = None

    actions = []
    best_valid = None
    for action_id, q_value in enumerate(q_tensor.tolist()):
        valid = bool(action_mask[action_id])
        q_value = safe_float(q_value)
        masked_q = q_value if valid else None
        action = {
            "id": action_id,
            "key": battle_agent.get_action_key(action_id),
            "q": q_value,
            "masked_q": masked_q,
            "valid": valid,
            "selected": action_id == selected_action_id,
        }
        actions.append(action)
        if valid and q_value is not None and (best_valid is None or q_value > best_valid["q"]):
            best_valid = action

    return {
        "available": True,
        "screen_type": state_type,
        "selected_action_id": selected_action_id,
        "best_valid_action": best_valid,
        "actions": actions,
    }


def safe_float(value: float) -> float | None:
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def evaluate_episode(
    game: Game,
    agent: Agent,
    max_steps: int,
    sleep_seconds: float,
    checkpoint_name: str,
    episode_index: int,
    run_seed: str | None,
    dashboard: LiveEvaluationDashboard | None = None,
) -> dict:
    raw_state = game.reset(run_seed=run_seed)
    episode_reward = 0.0
    battle_reward = 0.0
    steps = 0
    battle_steps = 0
    battle_wins = 0
    battle_losses = 0

    while raw_state.get("state_type") != "game_over" and steps < max_steps:
        if should_skip_agent(raw_state):
            action = {"type": "proceed"}
            q_values = current_q_values(agent, raw_state, action)
        else:
            action = choose_eval_action(agent, raw_state)
            q_values = current_q_values(agent, raw_state, action)

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
        if dashboard is not None:
            dashboard.update_step(
                checkpoint_name,
                episode_index,
                run_seed,
                steps,
                raw_state,
                next_raw_state,
                action,
                reward,
                episode_reward,
                battle_reward,
                battle_wins,
                battle_losses,
                done,
                q_values,
            )
        print(
            f"{checkpoint_name} episode={episode_index} step={steps} "
            f"seed={run_seed or 'normal'} "
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
        "seed": run_seed or "normal",
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
    game_mode: str,
    base_seed: str | None,
    max_steps: int,
    sleep_seconds: float,
    dashboard: LiveEvaluationDashboard | None = None,
) -> dict:
    agent = Agent()
    agent.battle_agent.load(str(checkpoint_path))
    if agent.battle_agent.model is not None:
        agent.battle_agent.model.eval()

    game = Game(
        character=character,
        base_url=base_url,
        timeout=timeout,
        game_mode=game_mode,
    )
    episode_results = []
    for episode_index in range(1, episodes + 1):
        run_seed = (
            evaluation_seed(base_seed, episode_index)
            if game_mode in {"custom", "daily"}
            else None
        )
        episode_results.append(
            evaluate_episode(
                game,
                agent,
                max_steps,
                sleep_seconds,
                checkpoint_path.name,
                episode_index,
                run_seed,
                dashboard,
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
            "game_mode": game_mode,
            "seeds": ",".join(result["seed"] for result in episode_results),
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
