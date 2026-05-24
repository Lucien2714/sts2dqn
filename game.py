import typing as t
import logging
import numpy as np
import random

from mcp_api import STS2Client, STS2ClientError, GameCharacter
from player import Player

COMBAT_STATE_TYPES = {"monster", "elite", "boss"}
BATTLE_STATE_TYPES = COMBAT_STATE_TYPES | {"hand_select"}
BATTLE_REWARD_STATE_TYPES = {"rewards", "card_reward"}
BATTLE_GOLD_LOSS_PENALTY = 0.1
BATTLE_MAX_HP_LOSS_PENALTY = 5.0

logger = logging.getLogger(__name__)


class Game:
    """
    Minimal game skeleton for Deep Q-Learning.
    Uses STS2Client as the backend game client.
    """

    def __init__(
        self,
        character: int=0,
        seed: t.Optional[int] = None,
        base_url: str = "http://localhost:15526/api/v1",
        timeout: float = 20.0,
    ):
        self.character = character
        random.seed(seed)
        self.player=Player(character)
        self._last_player_hp = None
        self._battle_start_hp = None
        self._battle_start_gold = None
        self._battle_start_max_hp = None
        self._battle_reward_closed = False
        # Init game client in singleplayer mode
        self.client = STS2Client(
            base_url=base_url,
            mode="singleplayer",
            timeout=timeout,
        )


    def _init_state(self) -> np.ndarray:
        """Create an initial state. Override for custom state encoding."""
        return np.zeros(self.state_size, dtype=np.float32)

    def _encode_state(self, raw_state: dict) -> np.ndarray:
       '''
       TODO
       '''
       pass

    def reset(self):
        """
        Reset environment and start/create singleplayer mode.

        This tries to move from main menu into:
            menu -> singleplayer -> standard

        Then it returns the current encoded observation.
        """
        raw_state = self.client.get_state()

        if raw_state.get("state_type") == "game_over":
            raw_state = self._menu_select_state("main_menu")

        if raw_state.get("state_type") == "menu":
            menu_screen = raw_state.get("menu_screen")

            if menu_screen == "main":
                if "abandon_run" in raw_state.get("options", []):
                    raw_state = self._menu_select_state("abandon_run")
                    raw_state = self._menu_select_state("yes")
                raw_state = self._menu_select_state("singleplayer")

            if raw_state.get("state_type") == "menu" and raw_state.get("menu_screen") == "singleplayer":
                raw_state = self._menu_select_state("standard")
                
            if raw_state.get("state_type") == "menu" and raw_state.get("menu_screen") == "character_select":
                raw_state = self._menu_select_state(GameCharacter.get(self.character, 0))
                logger.info(f"Selected character: {GameCharacter.get(self.character, 0)}")
                raw_state = self._menu_select_state("confirm")
        self.player=Player(self.character)
        self._last_player_hp = self._player_hp(raw_state, self._last_player_hp)
        self._battle_start_hp = None
        self._battle_start_gold = None
        self._battle_start_max_hp = None
        self._battle_reward_closed = False
        self.act=raw_state.get("run", {}).get("act", 0)
        self.floor=raw_state.get("run", {}).get("floor", 0)
        self.ascension=raw_state.get("run", {}).get("ascension", 0)
        print(f"Started run: act={self.act}, floor={self.floor}, ascension={self.ascension}")
        self._state = self._encode_state(raw_state)

        return raw_state

    def step(self, action: dict, prev_state: dict):
        try:
            api_result = self._apply_action(action, prev_state)
        except Exception as exc:
            raw_state = self.client.get_state()
            reward = 0.0
            done = raw_state.get("state_type") == "game_over"
            info = {
                "error": str(exc),
                "raw_state": raw_state,
                "action": action,
                "reward_details": {
                    "type": "action_error",
                    "error": str(exc),
                    "action_error": True,
                    "total": reward,
                },
            }

            return 0, reward, done, info

        raw_state = self._state_from_action_result(api_result)

        reward, reward_details = self._compute_reward(prev_state, raw_state, action)
        self._last_player_hp = self._player_hp(raw_state, self._last_player_hp)
        done = raw_state.get("state_type") == "game_over"

        info = {
            "api_result": api_result,
            "raw_state": raw_state,
            "action": action,
            "reward_details": reward_details,
        }

        return 0, float(reward), bool(done), info

    def _state_from_action_result(self, api_result):
        if isinstance(api_result, dict):
            if isinstance(api_result.get("state"), dict):
                return api_result["state"]
            if isinstance(api_result.get("raw_state"), dict):
                return api_result["raw_state"]
            if api_result.get("state_type") is not None:
                return api_result

        raise STS2ClientError(f"Action response did not include state: {api_result}")

    def _menu_select_state(self, option: str):
        logger.info("Reset menu_select option=%s", option)
        raw_state = self._state_from_action_result(self.client.menu_select(option))
        logger.info(
            "Reset menu_select option=%s -> state_type=%s menu_screen=%s",
            option,
            raw_state.get("state_type"),
            raw_state.get("menu_screen"),
        )
        return raw_state

    def _apply_action(self, action: dict, raw_state: dict):
        action_type = action.get("type")

        if action_type == "end_turn":
            return self.client.end_turn()

        if action_type == "proceed":
            return self.client.proceed()

        if action_type == "play_card":
            return self.client.play_card(
                card_index=action["card_index"],
                target=action.get("target"),
            )

        if action_type == "use_potion":
            return self.client.use_potion(
                slot=action["slot"],
                target=action.get("target"),
            )

        if action_type == "combat_select_card":
            return self.client.combat_select_card(
                card_index=action["card_index"],
            )

        if action_type == "combat_confirm_selection":
            return self.client.combat_confirm_selection()

        if action_type == "choose_map_node":
            return self.client.choose_map_node(
                index=action["index"],
            )

        if action_type == "claim_reward":
            return self.client.claim_reward(
                index=action["index"],
            )

        if action_type == "select_card_reward":
            return self.client.select_card_reward(
                card_index=action["card_index"],
            )

        if action_type == "skip_card_reward":
            return self.client.skip_card_reward()

        if action_type == "choose_rest_option":
            return self.client.choose_rest_option(
                index=action["index"],
            )

        if action_type == "choose_event_option":
            return self.client.choose_event_option(
                index=action["index"],
            )

        if action_type == "advance_dialogue":
            return self.client.advance_dialogue()

        if action_type == "select_card":
            return self.client.select_card(
                index=action["index"],
            )

        if action_type == "confirm_selection":
            return self.client.confirm_selection()

        if action_type == "cancel_selection":
            return self.client.cancel_selection()

        if action_type == "shop_purchase":
            return self.client.shop_purchase(
                index=action["index"],
            )

        raise ValueError(f"Unknown action type: {action_type}")

    def _compute_reward(
        self,
        prev_state: dict,
        next_state: dict,
        action: dict | None = None,
    ) -> tuple[float, dict]:
        """
        TODO: Create reward function for each different type of state
        """
        if prev_state.get("state_type") in BATTLE_STATE_TYPES:
            return self._compute_battle_reward(prev_state, next_state, action)

        reward = 0.0

        prev_player = prev_state.get("player", {})
        next_player = next_state.get("player", {})

        prev_hp = prev_player.get("hp", 0)
        next_hp = next_player.get("hp", 0)

        prev_floor = prev_state.get("run", {}).get("floor", 0)
        next_floor = next_state.get("run", {}).get("floor", 0)

        # Reward floor progress
        reward += float(next_floor - prev_floor) * 10.0

        # Penalize HP loss
        reward += float(next_hp - prev_hp) * 0.2

        # Terminal reward
        if next_state.get("state_type") == "game_over":
            reward -= 10.0

        return reward, {
            "type": "default",
            "total": reward,
        }

    def _compute_battle_reward(
        self,
        prev_state: dict,
        next_state: dict,
        action: dict | None = None,
    ) -> tuple[float, dict]:
        prev_hp = self._player_hp(prev_state, self._last_player_hp)
        next_hp = self._player_hp(next_state, prev_hp)
        prev_gold = self._player_gold(prev_state)
        next_gold = self._player_gold(next_state, prev_gold)
        prev_max_hp = self._player_max_hp(prev_state)
        next_max_hp = self._player_max_hp(next_state, prev_max_hp)
        prev_has_alive_enemy = self._battle_has_alive_enemy(prev_state)

        if prev_has_alive_enemy:
            self._battle_reward_closed = False

        if self._battle_start_hp is None and prev_has_alive_enemy:
            self._battle_start_hp = prev_hp
        if self._battle_start_gold is None and prev_has_alive_enemy:
            self._battle_start_gold = prev_gold
        if self._battle_start_max_hp is None and prev_has_alive_enemy:
            self._battle_start_max_hp = prev_max_hp

        if self._battle_reward_closed and not prev_has_alive_enemy:
            return 0.0, {
                "type": "battle",
                "prev_state_type": prev_state.get("state_type"),
                "next_state_type": next_state.get("state_type"),
                "result": None,
                "already_resolved": True,
                "prev_hp": prev_hp,
                "next_hp": next_hp,
                "prev_gold": prev_gold,
                "next_gold": next_gold,
                "prev_max_hp": prev_max_hp,
                "next_max_hp": next_max_hp,
                "total": 0.0,
            }

        step_hp_lost = max(0, prev_hp - next_hp)
        battle_start_hp = self._battle_start_hp if self._battle_start_hp is not None else prev_hp
        battle_start_gold = self._battle_start_gold if self._battle_start_gold is not None else prev_gold
        battle_start_max_hp = (
            self._battle_start_max_hp
            if self._battle_start_max_hp is not None
            else prev_max_hp
        )
        total_hp_lost = max(0, battle_start_hp - next_hp)
        total_gold_lost = max(0, battle_start_gold - next_gold)
        total_max_hp_lost = max(0, battle_start_max_hp - next_max_hp)

        potion_used = bool(action and action.get("type") == "use_potion")
        potion_penalty = -5.0 if potion_used else 0.0

        prev_state_type = prev_state.get("state_type")
        next_state_type = next_state.get("state_type")
        battle_result = self._battle_result(prev_state, next_state)
        enemy_hp_lost, enemies_killed = self._enemy_hp_progress(
            prev_state,
            next_state,
            count_missing_as_dead=battle_result != "lost",
        )
        enemy_damage_reward = float(enemy_hp_lost) * 0.2
        enemy_kill_reward = float(enemies_killed) * 10.0
        end_turn_energy_penalty = self._end_turn_energy_penalty(prev_state, action)
        win_reward = 100.0 if battle_result == "won" else 0.0
        hp_penalty = -float(total_hp_lost) if battle_result is not None else 0.0
        gold_penalty = (
            -float(total_gold_lost) * BATTLE_GOLD_LOSS_PENALTY
            if battle_result is not None
            else 0.0
        )
        max_hp_penalty = (
            -float(total_max_hp_lost) * BATTLE_MAX_HP_LOSS_PENALTY
            if battle_result is not None
            else 0.0
        )
        reward = (
            hp_penalty
            + gold_penalty
            + max_hp_penalty
            + potion_penalty
            + win_reward
            + enemy_damage_reward
            + enemy_kill_reward
            + end_turn_energy_penalty
        )
        if battle_result is not None:
            self._battle_start_hp = None
            self._battle_start_gold = None
            self._battle_start_max_hp = None
            self._battle_reward_closed = True

        return reward, {
            "type": "battle",
            "prev_state_type": prev_state_type,
            "next_state_type": next_state_type,
            "result": battle_result,
            "prev_hp": prev_hp,
            "next_hp": next_hp,
            "prev_gold": prev_gold,
            "next_gold": next_gold,
            "prev_max_hp": prev_max_hp,
            "next_max_hp": next_max_hp,
            "battle_start_hp": battle_start_hp,
            "battle_start_gold": battle_start_gold,
            "battle_start_max_hp": battle_start_max_hp,
            "step_hp_lost": step_hp_lost,
            "hp_lost": total_hp_lost,
            "gold_lost": total_gold_lost,
            "max_hp_lost": total_max_hp_lost,
            "hp_penalty": hp_penalty,
            "gold_penalty": gold_penalty,
            "max_hp_penalty": max_hp_penalty,
            "enemy_hp_lost": enemy_hp_lost,
            "enemies_killed": enemies_killed,
            "enemy_damage_reward": enemy_damage_reward,
            "enemy_kill_reward": enemy_kill_reward,
            "end_turn_energy_penalty": end_turn_energy_penalty,
            "potion_used": potion_used,
            "potion_penalty": potion_penalty,
            "win_reward": win_reward,
            "total": reward,
        }

    def _battle_result(self, prev_state: dict, next_state: dict) -> str | None:
        prev_state_type = prev_state.get("state_type")
        next_state_type = next_state.get("state_type")
        if prev_state_type not in BATTLE_STATE_TYPES:
            return None
        if next_state_type == "game_over":
            return "lost"
        if next_state_type in BATTLE_REWARD_STATE_TYPES:
            if self._battle_start_hp is not None or self._battle_has_alive_enemy(prev_state):
                return "won"
            return None
        if (
            next_state_type in BATTLE_STATE_TYPES
            and self._battle_has_alive_enemy(prev_state)
            and not self._battle_has_alive_enemy(next_state)
            and self._enemy_hp_map(prev_state)
        ):
            return "won"
        return None

    def _player_hp(self, state: dict, default: int | None = 0) -> int:
        player = state.get("player", {})
        if isinstance(player, dict) and player.get("hp") is not None:
            return self._parse_int(player.get("hp"), default or 0)
        return default or 0

    def _player_gold(self, state: dict, default: int | None = 0) -> int:
        player = state.get("player", {})
        if isinstance(player, dict) and player.get("gold") is not None:
            return self._parse_int(player.get("gold"), default or 0)
        return default or 0

    def _player_max_hp(self, state: dict, default: int | None = 0) -> int:
        player = state.get("player", {})
        if isinstance(player, dict) and player.get("max_hp") is not None:
            return self._parse_int(player.get("max_hp"), default or 0)
        return default or 0

    def _enemy_hp_progress(
        self,
        prev_state: dict,
        next_state: dict,
        count_missing_as_dead: bool,
    ) -> tuple[int, int]:
        prev_enemies = self._enemy_hp_map(prev_state)
        next_enemies = self._enemy_hp_map(next_state)
        hp_lost = 0
        killed = 0

        for enemy_key, prev_hp in prev_enemies.items():
            next_hp = next_enemies.get(enemy_key)
            if next_hp is None:
                next_hp = 0 if count_missing_as_dead else prev_hp

            hp_lost += max(0, prev_hp - next_hp)
            if prev_hp > 0 and next_hp <= 0:
                killed += 1

        return hp_lost, killed

    def _enemy_hp_map(self, state: dict) -> dict[str, int]:
        enemies = state.get("battle", {}).get("enemies", state.get("enemies", []))
        enemy_hp = {}

        for enemy_index, enemy in enumerate(enemies):
            enemy_key = self._enemy_key(enemy, enemy_index)
            enemy_hp[enemy_key] = self._parse_int(enemy.get("hp", 0))

        return enemy_hp

    def _battle_has_alive_enemy(self, state: dict) -> bool:
        return any(hp > 0 for hp in self._enemy_hp_map(state).values())

    def _enemy_key(self, enemy: dict, enemy_index: int) -> str:
        for key in ("entity_id", "id", "combat_id", "name"):
            value = enemy.get(key)
            if value is not None:
                return str(value)
        return str(enemy_index)

    def _end_turn_energy_penalty(self, prev_state: dict, action: dict | None) -> float:
        if not action or action.get("type") != "end_turn":
            return 0.0

        energy = self._parse_int(prev_state.get("player", {}).get("energy", 0))
        return -0.1 * float(max(0, energy))

    def _parse_int(self, value: object, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def sample_action(self) -> int:
        """Sample a random valid action."""
        return int(self.rng.randint(0, self.action_size))

    def render(self, mode: str = "human"):
        """Optional human/array rendering."""
        if mode == "human":
            raw_state = self.client.get_state()
            print("State type:", raw_state.get("state_type"))
            print("Encoded state:", self._state)
        elif mode == "rgb_array":
            return self._state.copy()
        else:
            raise ValueError(f"Unsupported render mode: {mode}")
    def is_end_state(self)-> bool:
        return self.client.get_state().get("state_type", False) == "game_over"

    def close(self):
        """Cleanup resources if needed."""
        pass
