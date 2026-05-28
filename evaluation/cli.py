import argparse
import logging
from pathlib import Path

from main import CHECKPOINT_DIR

from evaluation.checkpoints import find_checkpoints
from evaluation.config import (
    DEFAULT_EPISODES,
    DEFAULT_GAME_MODE,
    DEFAULT_LIVE_HTTP_HOST,
    DEFAULT_LIVE_HTTP_PORT,
    DEFAULT_LIVE_WS_HOST,
    DEFAULT_LIVE_WS_PORT,
    DEFAULT_MAX_STEPS,
    DEFAULT_SLEEP_SECONDS,
)
from evaluation.dashboard import LiveEvaluationDashboard
from evaluation.runner import evaluate_checkpoint, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate each saved battle-agent checkpoint without training."
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--character", type=int, default=0)
    parser.add_argument("--base-url", default="http://localhost:15526/api/v1")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--game-mode",
        choices=["standard", "daily", "custom"],
        default=DEFAULT_GAME_MODE,
    )
    parser.add_argument(
        "--seed",
        default=None,
        help="Optional fixed base seed for custom/daily mode. Ignored in standard mode.",
    )
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP_SECONDS)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Serve the live evaluation dashboard over HTTP.",
    )
    parser.add_argument(
        "--live-html",
        type=Path,
        default=None,
        help="Optionally write a static copy of the live dashboard HTML to this file.",
    )
    parser.add_argument(
        "--live-http-host",
        default=DEFAULT_LIVE_HTTP_HOST,
        help="Host interface for the live dashboard HTTP server.",
    )
    parser.add_argument(
        "--live-http-port",
        type=int,
        default=DEFAULT_LIVE_HTTP_PORT,
        help="Port for the live dashboard HTTP server. Use 0 to choose a free port.",
    )
    parser.add_argument(
        "--live-ws-host",
        default=DEFAULT_LIVE_WS_HOST,
        help="Host interface for the live dashboard WebSocket server.",
    )
    parser.add_argument(
        "--live-ws-port",
        type=int,
        default=DEFAULT_LIVE_WS_PORT,
        help="Port for the live dashboard WebSocket server. Use 0 to choose a free port.",
    )
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

    dashboard_enabled = args.live or args.live_html is not None
    dashboard = None
    if dashboard_enabled:
        dashboard = LiveEvaluationDashboard(
            args.live_html,
            args.live_ws_host,
            args.live_ws_port,
            args.live_http_host,
            args.live_http_port,
        )
    if dashboard is not None:
        logging.info(
            "Serving live evaluation dashboard at %s websocket=%s",
            dashboard.http_url,
            dashboard.websocket_url,
        )
        if args.live_html is not None:
            logging.info("Wrote static dashboard copy to %s", args.live_html)

    rows = []
    try:
        for checkpoint_path in checkpoints:
            logging.info("Evaluating checkpoint %s", checkpoint_path)
            try:
                result = evaluate_checkpoint(
                    checkpoint_path,
                    args.episodes,
                    args.character,
                    args.base_url,
                    args.timeout,
                    args.game_mode,
                    args.seed,
                    args.max_steps,
                    args.sleep,
                    dashboard,
                )
            except Exception as exc:
                logging.exception("Could not evaluate checkpoint %s: %s", checkpoint_path, exc)
                continue

            rows.append(result)
            if dashboard is not None:
                dashboard.add_checkpoint_result(result)
            print_checkpoint_result(result)

        if args.csv is not None:
            write_csv(args.csv, rows)
            logging.info("Wrote evaluation results to %s", args.csv)

        if dashboard is not None:
            dashboard.finish()
    finally:
        if dashboard is not None:
            dashboard.close()


def print_checkpoint_result(result: dict) -> None:
    print(
        f"{result['checkpoint']}: "
        f"trained_steps={result['trained_steps']} "
        f"learn_steps={result['learn_steps']} "
        f"game_mode={result['game_mode']} "
        f"seeds=[{result['seeds']}] "
        f"avg_reward={result['avg_reward']:.2f} "
        f"avg_battle_reward={result['avg_battle_reward']:.2f} "
        f"avg_steps={result['avg_steps']:.1f} "
        f"ending_steps=[{result['ending_steps']}] "
        f"battle_win_rate={result['battle_win_rate']:.2%} "
        f"wins={result['battle_wins']} losses={result['battle_losses']} "
        f"avg_floor={result['avg_floor']:.1f} max_floor={result['max_floor']} "
        f"timeouts={result['timeouts']}"
    )
