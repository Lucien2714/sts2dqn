import secrets
import string


SEED_ALPHABET = string.ascii_uppercase + string.digits
SEED_LENGTH = 10


def evaluation_seed(base_seed: str | None, episode_index: int) -> str:
    if base_seed is None:
        return random_evaluation_seed()
    if episode_index == 1:
        return base_seed
    return f"{base_seed}_{episode_index}"


def random_evaluation_seed() -> str:
    return "".join(secrets.choice(SEED_ALPHABET) for _ in range(SEED_LENGTH))
