from __future__ import annotations

from collections import deque
import logging
import random
import re


try:
  import torch
  from torch import nn
except ImportError:
  torch = None
  nn = None


logger = logging.getLogger(__name__)


BATTLE_MAX_HAND = 10
BATTLE_MAX_POTIONS = 10
BATTLE_MAX_ENEMIES = 5
BATTLE_MAX_POWERS = 259
BATTLE_CARD_FEATURES = 10
BATTLE_ENEMY_FEATURES = 11
BATTLE_PLAYER_FEATURES = 10


try:
  from data import (
    get_card_index,
    get_card_map_size,
    get_data_map_size,
    get_intent_index,
    get_monster_index,
    get_potion_index,
    get_power_index,
    get_relic_index,
  )
except ImportError:
  from sts2dqn.data import (
    get_card_index,
    get_card_map_size,
    get_data_map_size,
    get_intent_index,
    get_monster_index,
    get_potion_index,
    get_power_index,
    get_relic_index,
  )



if nn is not None:
  class BattleQNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
      super().__init__()
      self.net = nn.Sequential(
        nn.Linear(state_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, action_size),
      )
    def forward(self, state):
      return self.net(state)
else:
  class BattleQNetwork:
    def __init__(self, *args, **kwargs):
      raise ModuleNotFoundError("torch is required for the DQN network")



class BattleDQNAgent:
  MAX_HAND = BATTLE_MAX_HAND
  MAX_POTIONS = BATTLE_MAX_POTIONS
  MAX_ENEMIES = BATTLE_MAX_ENEMIES
  MAX_POWERS = BATTLE_MAX_POWERS
  CARD_FEATURES = BATTLE_CARD_FEATURES
  ENEMY_FEATURES = BATTLE_ENEMY_FEATURES
  PLAYER_FEATURES = BATTLE_PLAYER_FEATURES

  ACTION_SET = (
    ["end_turn"]
    + [
      f"play_card_{card_index}_target_{enemy_index}"
      for card_index in range(BATTLE_MAX_HAND)
      for enemy_index in range(BATTLE_MAX_ENEMIES)
    ]
    + [f"play_card_{card_index}_self" for card_index in range(BATTLE_MAX_HAND)]
    + [
      f"use_potion_{slot}_target_{enemy_index}"
      for slot in range(BATTLE_MAX_POTIONS)
      for enemy_index in range(BATTLE_MAX_ENEMIES)
    ]
    + [f"use_potion_{slot}_self" for slot in range(BATTLE_MAX_POTIONS)]
    + [f"combat_select_card_{card_index}" for card_index in range(BATTLE_MAX_HAND)]
    + ["combat_confirm_selection"]
  )

  def __init__(
    self,
    gamma=0.99,
    epsilon=1.0,
    learning_rate=0.0005,
    epsilon_decay=0.99995,
    epsilon_min=0.01,
    batch_size=32,
    memory_size=100000,
    update_freq=4,
    update_freq_target=1000,
    hidden_size=256,
    device=None,
  ):
    self.action_set = list(self.ACTION_SET)
    self.act2id = {action: i for i, action in enumerate(self.action_set)}
    self.id2act = {i: action for i, action in enumerate(self.action_set)}
    self.id2game_action = {
      action_id: self._action_key_to_game_action(action)
      for action_id, action in self.id2act.items()
    }
    self.action_size = len(self.action_set)

    self.card_vector_size = get_card_map_size() * 2
    self.intent_vector_size = get_data_map_size("intents")
    self.potion_vector_size = get_data_map_size("potions")
    self.relic_vector_size = get_data_map_size("relics")
    self.potion_slot_size = 1 + self.potion_vector_size + 2
    self.state_size = (
      1
      + self.action_size
      + self.PLAYER_FEATURES
      + self.MAX_HAND * self.CARD_FEATURES
      + self.card_vector_size * 3
      + self.MAX_ENEMIES * self.ENEMY_FEATURES
      + self.MAX_POTIONS * self.potion_slot_size
      + self.MAX_POWERS
      + self.relic_vector_size
    )

    self.update_freq = update_freq
    self.update_freq_target = update_freq_target
    self.gamma = gamma
    self.epsilon = epsilon
    self.epsilon_decay = epsilon_decay
    self.epsilon_min = epsilon_min
    self.batch_size = batch_size
    self.replay_buffer = deque(maxlen=memory_size)
    self.trained_steps = 0
    self.learn_steps = 0

    self.device = self._resolve_device(device)
    self.model = None
    self.target_model = None
    self.optimizer = None
    self.loss_fn = None

    if torch is not None:
      self.model = BattleQNetwork(self.state_size, self.action_size, hidden_size).to(self.device)
      self.target_model = BattleQNetwork(self.state_size, self.action_size, hidden_size).to(self.device)
      self.target_model.load_state_dict(self.model.state_dict())
      self.target_model.eval()
      self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
      self.loss_fn = nn.SmoothL1Loss()

  def choose_action(self, raw_state: dict, training: bool = True) -> dict:
    action_mask = self.valid_action_mask(raw_state)
    valid_action_ids = [index for index, is_valid in enumerate(action_mask) if is_valid]

    if len(valid_action_ids) == 0:
      return self._fallback_action(raw_state)

    if training and random.random() < self.epsilon:
      return self.get_game_action(random.choice(valid_action_ids), raw_state)

    if self.model is None:
      raise ModuleNotFoundError("torch is required for DQN action selection")

    state_vector = self.encode_state(raw_state, action_mask)
    with torch.no_grad():
      state_tensor = torch.tensor(state_vector, dtype=torch.float32, device=self.device).unsqueeze(0)
      q_values = self.model(state_tensor).squeeze(0)
      invalid_mask = torch.tensor(
        [not is_valid for is_valid in action_mask],
        dtype=torch.bool,
        device=self.device,
      )
      q_values = q_values.masked_fill(invalid_mask, float("-inf"))
      action_id = int(torch.argmax(q_values).item())

    return self.get_game_action(action_id, raw_state)

  def remember(
    self,
    state,
    action_id: int,
    reward: float,
    next_state,
    done: bool,
    next_action_mask,
  ) -> None:
    self.replay_buffer.append((
      list(state),
      int(action_id),
      float(reward),
      list(next_state),
      bool(done),
      list(next_action_mask),
    ))
    self.trained_steps += 1

  def train_step(self) -> float | None:
    if self.model is None:
      raise ModuleNotFoundError("torch is required for DQN training")

    if len(self.replay_buffer) < self.batch_size:
      return None

    batch = random.sample(self.replay_buffer, self.batch_size)
    states, actions, rewards, next_states, dones, next_masks = zip(*batch)

    states_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
    actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
    next_states_tensor = torch.tensor(next_states, dtype=torch.float32, device=self.device)
    dones_tensor = torch.tensor(dones, dtype=torch.bool, device=self.device)
    next_masks_tensor = torch.tensor(next_masks, dtype=torch.bool, device=self.device)

    current_q = self.model(states_tensor).gather(1, actions_tensor).squeeze(1)

    with torch.no_grad():
      next_q = self.target_model(next_states_tensor)
      next_q = next_q.masked_fill(~next_masks_tensor, float("-inf"))
      max_next_q = next_q.max(dim=1).values
      max_next_q = torch.where(torch.isfinite(max_next_q), max_next_q, torch.zeros_like(max_next_q))
      target_q = rewards_tensor + self.gamma * max_next_q * (~dones_tensor).float()

    loss = self.loss_fn(current_q, target_q)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    self.learn_steps += 1
    if self.learn_steps % self.update_freq_target == 0:
      self.target_model.load_state_dict(self.model.state_dict())

    if self.learn_steps % self.update_freq == 0:
      self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    return float(loss.item())

  def encode_state(self, raw_state: dict, action_mask=None) -> list[float]:
    battle = raw_state.get("battle", {})
    player = raw_state.get("player", {})

    if action_mask is None:
      action_mask = self.valid_action_mask(raw_state)

    features = []
    features.append(self._scale(raw_state.get("round", battle.get("round", 0)), 100))
    features.extend([1.0 if is_valid else 0.0 for is_valid in action_mask])
    features.extend(self._encode_player(player))

    hand = player.get("hand", raw_state.get("hand", []))
    for card_index in range(self.MAX_HAND):
      card = hand[card_index] if card_index < len(hand) else None
      features.extend(self._encode_hand_card(card, card_index, player))

    features.extend(self._encode_card_pile(player.get("draw_pile", [])))
    features.extend(self._encode_card_pile(player.get("discard_pile", [])))
    features.extend(self._encode_card_pile(player.get("exhaust_pile", [])))

    enemies = battle.get("enemies", raw_state.get("enemies", []))
    for enemy_index in range(self.MAX_ENEMIES):
      enemy = enemies[enemy_index] if enemy_index < len(enemies) else None
      features.extend(self._encode_enemy(enemy))

    potion_slots = self._potion_slots(player.get("potions", []))
    for potion_index in range(self.MAX_POTIONS):
      potion = potion_slots.get(potion_index)
      features.extend(self._encode_potion(potion, potion_index))

    features.extend(self._encode_power_bucket(player.get("status", player.get("powers", [])), self.MAX_POWERS))
    features.extend(self._encode_relic_bucket(player.get("relics", []), self.relic_vector_size))

    return [float(value) for value in features]

  def valid_action_mask(self, raw_state: dict) -> list[bool]:
    mask = [False for _ in range(self.action_size)]
    state_type = raw_state.get("state_type")

    if state_type == "hand_select":
      hand_select = raw_state.get("hand_select", {})
      cards = hand_select.get("cards", [])
      selected_indices = {
        self._parse_int(card.get("index", -1), -1)
        for card in hand_select.get("selected_cards", [])
      }

      for card in cards[: self.MAX_HAND]:
        card_index = self._parse_int(card.get("index", len(cards)))
        if card_index in selected_indices:
          continue
        if 0 <= card_index < self.MAX_HAND:
          mask[self.act2id[f"combat_select_card_{card_index}"]] = True

      if hand_select.get("can_confirm", False):
        mask[self.act2id["combat_confirm_selection"]] = True

      return mask

    if state_type not in {"monster", "elite", "boss"}:
      mask[self.act2id["end_turn"]] = True
      return mask

    battle = raw_state.get("battle", {})
    player = raw_state.get("player", {})
    hand = player.get("hand", raw_state.get("hand", []))
    enemies = battle.get("enemies", raw_state.get("enemies", []))
    potions = player.get("potions", [])
    energy = self._parse_int(player.get("energy", raw_state.get("energy", 0)))
    mask[self.act2id["end_turn"]] = True

    for card_index, card in enumerate(hand[: self.MAX_HAND]):
      if not self._is_playable_card(card, energy):
        continue
      if self._requires_enemy_target(card):
        for enemy_index, enemy in enumerate(enemies[: self.MAX_ENEMIES]):
          if self._parse_int(enemy.get("hp", 0)) > 0:
            mask[self.act2id[f"play_card_{card_index}_target_{enemy_index}"]] = True
      else:
        mask[self.act2id[f"play_card_{card_index}_self"]] = True

    for slot, potion in enumerate(potions[: self.MAX_POTIONS]):
      if not potion or not potion.get("can_use_in_combat", True):
        continue
      potion_slot = self._potion_slot(potion, slot)
      if not 0 <= potion_slot < self.MAX_POTIONS:
        continue
      if self._requires_enemy_target(potion):
        for enemy_index, enemy in enumerate(enemies[: self.MAX_ENEMIES]):
          if self._parse_int(enemy.get("hp", 0)) > 0:
            mask[self.act2id[f"use_potion_{potion_slot}_target_{enemy_index}"]] = True
      else:
        mask[self.act2id[f"use_potion_{potion_slot}_self"]] = True

    return mask

  def get_action_id(self, action_key: str) -> int:
    return self.act2id[action_key]

  def get_action_key(self, action_id: int) -> str:
    return self.id2act[action_id]

  def get_game_action(self, action_id: int, raw_state: dict | None = None) -> dict:
    action = dict(self.id2game_action[action_id])

    if action["type"] in {"play_card", "use_potion"} and raw_state is not None:
      item = self._action_item(action, raw_state)
      if item is not None and self._requires_enemy_target(item):
        enemies = raw_state.get("battle", {}).get("enemies", raw_state.get("enemies", []))
        target = self._enemy_id_by_index(enemies, action.get("target_index"))
        if target is not None:
          action["target"] = target

    return action

  def get_game_action_id(self, action: dict, raw_state: dict | None = None) -> int:
    return self.act2id[self._game_action_to_action_key(action, raw_state)]

  def _action_key_to_game_action(self, action_key: str) -> dict:
    if action_key == "end_turn":
      return {"type": "end_turn"}

    play_card_target = re.fullmatch(r"play_card_(\d+)_target_(\d+)", action_key)
    if play_card_target:
      return {
        "type": "play_card",
        "card_index": int(play_card_target.group(1)),
        "target_index": int(play_card_target.group(2)),
      }

    play_card_self = re.fullmatch(r"play_card_(\d+)_self", action_key)
    if play_card_self:
      return {
        "type": "play_card",
        "card_index": int(play_card_self.group(1)),
      }

    use_potion_target = re.fullmatch(r"use_potion_(\d+)_target_(\d+)", action_key)
    if use_potion_target:
      return {
        "type": "use_potion",
        "slot": int(use_potion_target.group(1)),
        "target_index": int(use_potion_target.group(2)),
      }

    use_potion_self = re.fullmatch(r"use_potion_(\d+)_self", action_key)
    if use_potion_self:
      return {
        "type": "use_potion",
        "slot": int(use_potion_self.group(1)),
      }

    if action_key.startswith("combat_select_card_"):
      return {
        "type": "combat_select_card",
        "card_index": int(action_key.removeprefix("combat_select_card_")),
      }

    if action_key == "combat_confirm_selection":
      return {"type": "combat_confirm_selection"}

    raise ValueError(f"Unknown action key: {action_key}")

  def _game_action_to_action_key(self, action: dict, raw_state: dict | None = None) -> str:
    action_type = action.get("type")

    if action_type == "end_turn":
      return "end_turn"

    if action_type == "play_card":
      target_index = self._action_target_index(action, raw_state)
      if target_index is not None:
        return f"play_card_{action['card_index']}_target_{target_index}"
      return f"play_card_{action['card_index']}_self"

    if action_type == "use_potion":
      target_index = self._action_target_index(action, raw_state)
      if target_index is not None:
        return f"use_potion_{action['slot']}_target_{target_index}"
      return f"use_potion_{action['slot']}_self"

    if action_type == "combat_select_card":
      return f"combat_select_card_{action['card_index']}"

    if action_type == "combat_confirm_selection":
      return "combat_confirm_selection"

    raise ValueError(f"Unknown game action: {action}")

  def _encode_player(self, player: dict) -> list[float]:
    max_hp = max(1, self._parse_int(player.get("max_hp", 1)))
    max_energy = max(1, self._parse_int(player.get("max_energy", 3)))
    return [
      self._scale(player.get("max_hp", 0), 200),
      self._scale(player.get("hp", 0), max_hp),
      self._scale(player.get("block", 0), 100),
      self._scale(player.get("energy", 0), max_energy),
      self._scale(player.get("orb_slots", player.get("orb_slot", 0)), 10),
      self._scale(len(player.get("status", player.get("powers", []))), 50),
      self._scale(len(player.get("hand", [])), self.MAX_HAND),
      self._scale(player.get("draw_pile_count", len(player.get("draw_pile", []))), 60),
      self._scale(player.get("discard_pile_count", len(player.get("discard_pile", []))), 60),
      self._scale(player.get("gold", 0), 999),
    ]

  def _encode_hand_card(self, card: dict | None, card_index: int, player: dict) -> list[float]:
    if card is None:
      return [0.0] * self.CARD_FEATURES

    energy = self._parse_int(player.get("energy", 0))
    card_id = get_card_index(card.get("id") or card.get("name"), default=-1)
    card_type = str(card.get("type", "")).lower()
    target_type = str(card.get("target_type", card.get("target", ""))).lower()

    return [
      1.0,
      self._scale(card_id + 1, max(1, get_card_map_size())),
      self._scale(card_index, self.MAX_HAND),
      self._scale(self._parse_cost(card.get("cost", 0), energy), 5),
      self._scale(self._parse_cost(card.get("star_cost", 0), 0), 5),
      self._card_type_value(card_type),
      1.0 if card.get("is_upgraded", card.get("upgraded", False)) else 0.0,
      1.0 if self._is_playable_card(card, energy) else 0.0,
      self._target_type_value(target_type),
      self._scale(self._estimated_damage(card), 100),
    ]

  def _encode_card_pile(self, pile: list) -> list[float]:
    vector = [0.0 for _ in range(self.card_vector_size)]
    for card in pile:
      if isinstance(card, dict):
        card_id = card.get("id") or card.get("name")
        upgraded = bool(card.get("is_upgraded", card.get("upgraded", False)))
      else:
        card_id = str(card)
        upgraded = False

      card_index = get_card_index(card_id, default=-1)
      if card_index < 0:
        continue

      vector[card_index * 2 + int(upgraded)] += 1.0

    return [min(value / 10.0, 1.0) for value in vector]

  def _encode_enemy(self, enemy: dict | None) -> list[float]:
    if enemy is None:
      return [0.0] * self.ENEMY_FEATURES

    max_hp = max(1, self._parse_int(enemy.get("max_hp", 1)))
    enemy_id = get_monster_index(self._monster_lookup_id(enemy), default=-1)
    attack_intent, support_intent = self._enemy_intents(enemy)
    intent_damage, intent_hit_count = self._intent_attack_parts(enemy)

    return [
      1.0,
      self._scale(enemy_id + 1, 200),
      self._scale(enemy.get("max_hp", 0), 500),
      self._scale(enemy.get("hp", 0), max_hp),
      self._scale_intent(attack_intent),
      self._scale_intent(support_intent),
      self._scale(len(enemy.get("status", enemy.get("powers", []))), 50),
      self._scale(intent_damage, 150),
      self._scale(intent_hit_count, 10),
      self._scale(enemy.get("block", 0), 150),
      1.0 if self._parse_int(enemy.get("hp", 0)) > 0 else 0.0,
    ]

  def _encode_potion(self, potion: dict | None, slot: int) -> list[float]:
    features = [0.0 for _ in range(self.potion_slot_size)]
    if potion is None:
      return features

    potion_id = get_potion_index(potion.get("id") or potion.get("name"), default=-1)
    features[0] = 1.0
    if 0 <= potion_id < self.potion_vector_size:
      features[1 + potion_id] = 1.0
    features[1 + self.potion_vector_size] = self._scale(slot, self.MAX_POTIONS)
    features[1 + self.potion_vector_size + 1] = (
      1.0 if potion.get("can_use_in_combat", True) else 0.0
    )
    return features

  def _encode_power_bucket(self, powers: list, bucket_size: int) -> list[float]:
    vector = [0.0 for _ in range(bucket_size)]
    for power in powers:
      if isinstance(power, dict):
        power_id = power.get("id") or power.get("name")
        amount = max(1, self._parse_int(power.get("amount", 1)))
      else:
        power_id = str(power)
        amount = 1

      index = get_power_index(power_id, default=-1)
      if 0 <= index < bucket_size:
        vector[index] += amount

    return [min(value / 10.0, 1.0) for value in vector]

  def _encode_relic_bucket(self, relics: list, bucket_size: int) -> list[float]:
    vector = [0.0 for _ in range(bucket_size)]
    for relic in relics:
      relic_id = relic.get("id") or relic.get("name") if isinstance(relic, dict) else str(relic)
      index = get_relic_index(relic_id, default=-1)
      if 0 <= index < bucket_size:
        vector[index] = 1.0

    return vector

  def _action_item(self, action: dict, raw_state: dict) -> dict | None:
    player = raw_state.get("player", {})
    if action["type"] == "play_card":
      hand = player.get("hand", raw_state.get("hand", []))
      card_index = action["card_index"]
      return hand[card_index] if card_index < len(hand) else None

    if action["type"] == "use_potion":
      potions = player.get("potions", [])
      slot = action["slot"]
      for potion_index, potion in enumerate(potions):
        if self._potion_slot(potion, potion_index) == slot:
          return potion
      return None

    return None

  def _potion_slot(self, potion: dict, fallback_slot: int) -> int:
    return self._parse_int(potion.get("slot", fallback_slot), fallback_slot)

  def _potion_slots(self, potions: list[dict]) -> dict[int, dict]:
    slots = {}
    for potion_index, potion in enumerate(potions):
      if not potion:
        continue
      slot = self._potion_slot(potion, potion_index)
      if 0 <= slot < self.MAX_POTIONS:
        slots[slot] = potion
    return slots

  def _fallback_action(self, raw_state: dict) -> dict:
    if raw_state.get("state_type") == "hand_select":
      hand_select = raw_state.get("hand_select", {})
      if hand_select.get("can_confirm", False):
        return {"type": "combat_confirm_selection"}

      cards = hand_select.get("cards", [])
      if cards:
        return {
          "type": "combat_select_card",
          "card_index": self._parse_int(cards[0].get("index", 0)),
        }

    return {"type": "end_turn"}

  def _is_playable_card(self, card: dict, energy: int) -> bool:
    if not card.get("can_play", True):
      return False
    if str(card.get("type", "")).lower() in {"status", "curse"}:
      return False
    return self._parse_cost(card.get("cost", 0), energy) <= energy

  def _requires_enemy_target(self, item: dict) -> bool:
    if item.get("requires_target", False):
      return True
    target_type = str(item.get("target_type", item.get("target", ""))).lower()
    return target_type in {"anyenemy", "enemy"}

  def _first_alive_enemy_id(self, enemies: list[dict]) -> str | None:
    for enemy in enemies:
      if self._parse_int(enemy.get("hp", 0)) > 0:
        return enemy.get("entity_id") or enemy.get("id")
    return None

  def _enemy_id_by_index(self, enemies: list[dict], enemy_index: object) -> str | None:
    enemy_index = self._parse_int(enemy_index, -1)
    if not 0 <= enemy_index < min(len(enemies), self.MAX_ENEMIES):
      return None
    enemy = enemies[enemy_index]
    if self._parse_int(enemy.get("hp", 0)) <= 0:
      return None
    return enemy.get("entity_id") or enemy.get("id")

  def _enemy_index_by_id(self, enemies: list[dict], enemy_id: object) -> int | None:
    if enemy_id is None:
      return None
    enemy_id = str(enemy_id)
    for enemy_index, enemy in enumerate(enemies[: self.MAX_ENEMIES]):
      if enemy_id in {str(enemy.get("entity_id")), str(enemy.get("id"))}:
        return enemy_index
    return None

  def _action_target_index(self, action: dict, raw_state: dict | None = None) -> int | None:
    if action.get("target_index") is not None:
      target_index = self._parse_int(action.get("target_index"), -1)
      if 0 <= target_index < self.MAX_ENEMIES:
        return target_index

    if raw_state is None or action.get("target") is None:
      return None

    enemies = raw_state.get("battle", {}).get("enemies", raw_state.get("enemies", []))
    return self._enemy_index_by_id(enemies, action.get("target"))

  def _monster_lookup_id(self, enemy: dict) -> str | None:
    monster_id = enemy.get("id") or enemy.get("entity_id")
    if monster_id is not None:
      return re.sub(r"_\d+$", "", str(monster_id))
    return enemy.get("name")

  def _enemy_intents(self, enemy: dict) -> tuple[str | None, str | None]:
    attack_intent = None
    support_intent = None

    for intent in enemy.get("intents", []):
      intent_id = self._intent_id(intent)
      if intent_id is None:
        continue

      if self._is_attack_intent(intent):
        attack_intent = intent_id
      else:
        support_intent = intent_id

    if attack_intent is None and support_intent is None:
      fallback_intent = enemy.get("intent") or enemy.get("intent_id")
      if fallback_intent is not None:
        if self._is_attack_intent(fallback_intent):
          attack_intent = str(fallback_intent)
        else:
          support_intent = str(fallback_intent)

    return attack_intent, support_intent

  def _intent_id(self, intent: object) -> str | None:
    if isinstance(intent, dict):
      return intent.get("id") or intent.get("type") or intent.get("label")
    if intent is None:
      return None
    return str(intent)

  def _intent_attack_parts(self, enemy: dict) -> tuple[int, int]:
    for intent in enemy.get("intents", []):
      if not isinstance(intent, dict) or not self._is_attack_intent(intent):
        continue

      label = str(intent.get("label", ""))
      repeated_damage = re.fullmatch(r"(\d+)\s*x\s*(\d+)", label, flags=re.IGNORECASE)
      if repeated_damage:
        return int(repeated_damage.group(1)), int(repeated_damage.group(2))

      if label.isdigit():
        return int(label), 1

      return self._parse_int(intent.get("damage", 0)), self._parse_int(intent.get("hit_count", 1), 1)

    return 0, 0

  def _is_attack_intent(self, intent: object) -> bool:
    if isinstance(intent, dict):
      intent_type = intent.get("type", intent.get("id", ""))
    else:
      intent_type = intent
    return str(intent_type).lower() == "attack"

  def _scale_intent(self, intent: str | None) -> float:
    intent_index = get_intent_index(intent, default=-1)
    return self._scale(intent_index + 1, self.intent_vector_size)

  def _estimated_damage(self, card: dict) -> int:
    if card.get("damage") is not None:
      return self._parse_int(card.get("damage"))
    description = str(card.get("description", ""))
    match = re.search(r"Deal\s+(\d+)", description, flags=re.IGNORECASE)
    return self._parse_int(match.group(1)) if match else 0

  def _card_type_value(self, card_type: str) -> float:
    values = {
      "attack": 0.2,
      "skill": 0.4,
      "power": 0.6,
      "status": 0.8,
      "curse": 1.0,
    }
    return values.get(card_type, 0.0)

  def _target_type_value(self, target_type: str) -> float:
    if "enemy" in target_type:
      return 0.33
    if "self" in target_type:
      return 0.66
    if target_type in {"none", ""}:
      return 0.0
    return 1.0

  def _parse_cost(self, cost: object, energy: int) -> int:
    if cost is None:
      return 0
    if isinstance(cost, str) and cost.upper() == "X":
      return energy
    return self._parse_int(cost)

  def _parse_int(self, value: object, default: int = 0) -> int:
    try:
      return int(value)
    except (TypeError, ValueError):
      return default

  def _scale(self, value: object, denominator: int | float) -> float:
    denominator = max(float(denominator), 1.0)
    return max(0.0, min(float(self._parse_int(value)) / denominator, 1.0))

  def _resolve_device(self, device):
    if torch is None:
      return "cpu"
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

  def save(self, path: str) -> None:
    if self.model is None:
      raise ModuleNotFoundError("torch is required to save the DQN")
    torch.save(
      {
        "model_state_dict": self.model.state_dict(),
        "target_model_state_dict": self.target_model.state_dict(),
        "optimizer_state_dict": self.optimizer.state_dict(),
        "epsilon": self.epsilon,
        "trained_steps": self.trained_steps,
        "learn_steps": self.learn_steps,
        "state_size": self.state_size,
        "action_size": self.action_size,
      },
      path,
    )

  def load(self, path: str) -> None:
    if self.model is None:
      raise ModuleNotFoundError("torch is required to load the DQN")
    checkpoint = torch.load(path, map_location=self.device)
    self.model.load_state_dict(checkpoint["model_state_dict"])
    self.target_model.load_state_dict(checkpoint["target_model_state_dict"])
    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    self.epsilon = checkpoint.get("epsilon", self.epsilon)
    self.trained_steps = checkpoint.get(
      "trained_steps",
      checkpoint.get("trained_step", checkpoint.get("learn_steps", self.trained_steps)),
    )
    self.learn_steps = checkpoint.get("learn_steps", self.learn_steps)
    logger.info(
      "Loaded DQN checkpoint from %s: trained_steps=%d learn_steps=%d epsilon=%.4f",
      path,
      self.trained_steps,
      self.learn_steps,
      self.epsilon,
    )
