import logging
import os
from pathlib import Path


def setup_logger(log_dir: str, run_name: str) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = os.path.join(log_dir, f"{run_name}.log")

    logger = logging.getLogger(run_name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


def init_wandb(config: dict):
    try:
        import wandb
    except ImportError:
        raise ImportError("wandb is not installed. Run: pip install wandb")

    logging_cfg = config.get("logging", {})
    wandb.init(
        project=logging_cfg.get("project", "radiology-vlm"),
        name=logging_cfg.get("run_name", "run"),
        config=config,
    )
    return wandb.run
