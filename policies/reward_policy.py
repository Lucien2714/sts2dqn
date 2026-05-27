import logging


logger = logging.getLogger(__name__)


class RewardPolicy:
    def choose_action(self, state: dict) -> dict:
        state_type = state.get("state_type")

        if state_type == "card_reward":
            return self._choose_card_reward_action(state)

        if state_type == "treasure":
            return self._choose_treasure_action(state)

        return self._choose_rewards_action(state)

    def _choose_treasure_action(self, state: dict) -> dict:
        treasure = state.get("treasure", {})
        relics = treasure.get("relics", [])
        logger.debug(
            "RewardPolicy: choosing treasure action relics=%d can_proceed=%s",
            len(relics),
            treasure.get("can_proceed"),
        )

        if relics:
            relic = relics[0]
            action = {
                "type": "claim_treasure_relic",
                "index": relic.get("index", 0),
            }
            logger.debug(
                "RewardPolicy: selected treasure relic index=%s id=%s name=%s",
                relic.get("index"),
                relic.get("id"),
                relic.get("name"),
            )
            logger.debug("RewardPolicy: selected action=%s", action)
            return action

        action = {"type": "proceed"}
        logger.debug("RewardPolicy: selected treasure fallback action=%s", action)
        return action

    def _choose_rewards_action(self, state: dict) -> dict:
        rewards = state.get("rewards", {})
        items = rewards.get("items", [])
        logger.debug(
            "RewardPolicy: choosing rewards action items=%d can_proceed=%s",
            len(items),
            rewards.get("can_proceed"),
        )

        if not items:
            action = {
                "type": "proceed"
            }
            logger.debug("RewardPolicy: selected action=%s", action)
            return action

        # Claim card rewards first. The API then opens the card_reward screen.
        for reward in items:
            logger.debug(
                "RewardPolicy: reward index=%s type=%s",
                reward.get("index"),
                reward.get("type"),
            )
            if reward.get("type") == "card":
                action = {
                    "type": "claim_reward",
                    "index": reward.get("index", 0),
                }
                logger.debug("RewardPolicy: selected action=%s", action)
                return action

        # Otherwise claim first reward
        reward = items[0]
        action = {
            "type": "claim_reward",
            "index": reward.get("index", 0),
        }
        logger.debug("RewardPolicy: selected action=%s", action)
        return action

    def _choose_card_reward_action(self, state: dict) -> dict:
        card_reward = state.get("card_reward", {})
        cards = card_reward.get("cards", [])
        logger.debug(
            "RewardPolicy: choosing card_reward action cards=%d can_skip=%s",
            len(cards),
            card_reward.get("can_skip"),
        )

        chosen_index = self._choose_card(cards)
        if chosen_index is not None:
            action = {
                "type": "select_card_reward",
                "card_index": chosen_index,
            }
            logger.debug("RewardPolicy: selected action=%s", action)
            return action

        if card_reward.get("can_skip", False):
            action = {
                "type": "skip_card_reward"
            }
            logger.debug("RewardPolicy: selected action=%s", action)
            return action

        action = {
            "type": "skip_card_reward"
        }
        logger.debug("RewardPolicy: selected fallback action=%s", action)
        return action

    def _choose_card(self, cards: list[dict]) -> int | None:
        # Simple placeholder:
        # pick first non-curse card
        for i, card in enumerate(cards):
            if card.get("type") != "curse":
                logger.debug(
                    "RewardPolicy: selected card reward index=%d id=%s type=%s",
                    i,
                    card.get("id"),
                    card.get("type"),
                )
                return i

        logger.debug("RewardPolicy: no non-curse card reward found")
        return None
