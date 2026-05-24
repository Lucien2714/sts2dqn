import logging
from pathlib import Path
import time
from game import Game
from ai_agent import Agent

SAVE_INTERVAL = 10
TRAINING_BACKUP_INTERVAL = 5000
CHECKPOINT_DIR = Path("checkpoints")
BATTLE_MODEL_PATH = CHECKPOINT_DIR / "battle_agent_latest.pt"
COMBAT_SCREEN_TYPES = {"monster", "elite", "boss"}


def should_skip_agent(raw_state: dict) -> bool:
    if raw_state.get("state_type") not in COMBAT_SCREEN_TYPES:
        return False

    battle = raw_state.get("battle", {})
    return not (
        battle.get("turn") == "player"
        and battle.get("is_play_phase") is True
    )


def battle_backup_path(trained_steps: int) -> Path:
    return CHECKPOINT_DIR / f"battle_agent_step_{trained_steps}.pt"


def save_battle_checkpoint(agent: Agent, path: Path, reason: str) -> None:
    agent.battle_agent.save(str(path))
    logging.info(
        "Saved battle agent checkpoint (%s) to %s trained_steps=%d learn_steps=%d",
        reason,
        path,
        agent.battle_agent.trained_steps,
        agent.battle_agent.learn_steps,
    )


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    game = Game(
        character=0,
    )

    agent = Agent()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    if BATTLE_MODEL_PATH.exists():
        try:
            agent.battle_agent.load(str(BATTLE_MODEL_PATH))
            logging.info("Loaded battle agent model from %s", BATTLE_MODEL_PATH)
        except Exception as exc:
            logging.warning(
                "Could not load battle agent model from %s; starting fresh. error=%s",
                BATTLE_MODEL_PATH,
                exc,
            )
    else:
        logging.info("No battle agent checkpoint found at %s; starting fresh", BATTLE_MODEL_PATH)

    last_backup_index = agent.battle_agent.trained_steps // TRAINING_BACKUP_INTERVAL
    episode = 1

    while True:
        raw_state = game.reset()
        done = False
        episode_reward = 0.0
        battle_reward = 0.0
        current_battle_reward = 0.0
        current_battle_steps = 0
        current_battle_hp_lost = 0
        current_battle_potions_used = 0
        battle_number = 0
        episode_steps = 0
        battle_steps = 0
        update_count = 0
        battle_wins = 0
        battle_losses = 0
        losses = []
        action_counts = {}
        epsilon_start = agent.battle_agent.epsilon
        replay_start = len(agent.battle_agent.replay_buffer)

        logging.info("Starting episode %d", episode)

        while raw_state.get("state_type")!="game_over":
            prev_raw_state = raw_state
            encoded_state = game._encode_state(raw_state)

            if should_skip_agent(raw_state):
                action = {"type": "proceed"}
            else:
                policy_state = {
                    "screen_type": raw_state.get("state_type"),
                    "raw_state": raw_state,
                }
                action = agent.choose_action(policy_state)
                
            next_state, reward, done, info = game.step(action, prev_raw_state)
            next_raw_state = info.get("raw_state")
            reward_details = info.get("reward_details", {})
            training_info = agent.train_from_step(
                prev_raw_state,
                action,
                reward,
                next_raw_state,
                done,
                reward_details,
            )
            episode_reward += reward
            episode_steps += 1

            loss = None
            if reward_details.get("type") == "battle":
                step_battle_reward = reward_details.get("total", reward)
                battle_reward += step_battle_reward
                current_battle_reward += step_battle_reward
                current_battle_steps += 1
                current_battle_hp_lost = reward_details.get("hp_lost", current_battle_hp_lost)
                if reward_details.get("potion_used", False):
                    current_battle_potions_used += 1
                if reward_details.get("result") == "won":
                    battle_wins += 1
                if reward_details.get("result") == "lost":
                    battle_losses += 1

                if reward_details.get("result") in {"won", "lost"}:
                    battle_number += 1
                    result = reward_details.get("result")
                    logging.info(
                        "Episode %d battle %d finished: result=%s battle_reward=%.2f "
                        "steps=%d hp=%s->%s hp_lost=%d gold_lost=%d max_hp_lost=%d "
                        "potions_used=%d hp_penalty=%.2f gold_penalty=%.2f "
                        "max_hp_penalty=%.2f win_reward=%.2f",
                        episode,
                        battle_number,
                        result,
                        current_battle_reward,
                        current_battle_steps,
                        reward_details.get("battle_start_hp"),
                        reward_details.get("next_hp"),
                        current_battle_hp_lost,
                        reward_details.get("gold_lost", 0),
                        reward_details.get("max_hp_lost", 0),
                        current_battle_potions_used,
                        reward_details.get("hp_penalty", 0.0),
                        reward_details.get("gold_penalty", 0.0),
                        reward_details.get("max_hp_penalty", 0.0),
                        reward_details.get("win_reward", 0.0),
                    )
                    current_battle_reward = 0.0
                    current_battle_steps = 0
                    current_battle_hp_lost = 0
                    current_battle_potions_used = 0

            if training_info is not None:
                battle_steps += 1
                loss = training_info["loss"]
                action_type = training_info["action_type"]
                action_counts[action_type] = action_counts.get(action_type, 0) + 1
                if training_info["updated"]:
                    update_count += 1
                    losses.append(loss)

                backup_index = agent.battle_agent.trained_steps // TRAINING_BACKUP_INTERVAL
                if backup_index > last_backup_index:
                    last_backup_index = backup_index
                    backup_step = backup_index * TRAINING_BACKUP_INTERVAL
                    try:
                        save_battle_checkpoint(
                            agent,
                            BATTLE_MODEL_PATH,
                            f"training milestone {backup_step}",
                        )
                        save_battle_checkpoint(
                            agent,
                            battle_backup_path(backup_step),
                            f"training milestone backup {backup_step}",
                        )
                    except Exception as exc:
                        logging.exception(
                            "Could not save battle agent milestone checkpoint at trained_steps=%d: %s",
                            agent.battle_agent.trained_steps,
                            exc,
                        )

            print(
                action,
                "reward=", reward,
                "reward_details=", reward_details,
                "done=", done,
                "loss=", loss,
                "replay=", len(agent.battle_agent.replay_buffer),
                "epsilon=", round(agent.battle_agent.epsilon, 4),
            )
            if done:
                break

            raw_state = next_raw_state
            time.sleep(0.3)

        avg_loss = sum(losses) / len(losses) if losses else None
        min_loss = min(losses) if losses else None
        max_loss = max(losses) if losses else None
        epsilon_end = agent.battle_agent.epsilon
        replay_end = len(agent.battle_agent.replay_buffer)
        action_summary = " ".join(
            f"{action_type}:{count}"
            for action_type, count in sorted(action_counts.items())
        ) or "none"
        loss_summary = (
            f"avg_loss={avg_loss:.4f} min_loss={min_loss:.4f} max_loss={max_loss:.4f}"
            if avg_loss is not None
            else "avg_loss=n/a min_loss=n/a max_loss=n/a"
        )

        logging.info(
            "Episode %d study summary: reward=%.2f battle_reward=%.2f steps=%d battle_steps=%d updates=%d "
            "%s epsilon=%.4f->%.4f replay=%d->%d wins=%d losses=%d actions=%s",
            episode,
            episode_reward,
            battle_reward,
            episode_steps,
            battle_steps,
            update_count,
            loss_summary,
            epsilon_start,
            epsilon_end,
            replay_start,
            replay_end,
            battle_wins,
            battle_losses,
            action_summary,
        )

        logging.info(
            "Episode %d ended; returning to main menu and starting a new run",
            episode,
        )
        if episode % SAVE_INTERVAL == 0:
            try:
                save_battle_checkpoint(
                    agent,
                    BATTLE_MODEL_PATH,
                    f"episode {episode}",
                )
            except Exception as exc:
                logging.exception(
                    "Could not save battle agent model after episode %d: %s",
                    episode,
                    exc,
                )

        episode += 1
        time.sleep(0.5)


if __name__ == "__main__":
    main()
