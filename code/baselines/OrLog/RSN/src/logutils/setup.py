import logging
from pathlib import Path
from typing import Tuple
from rich.logging import RichHandler
from .rich_report import Reporter


def setup_logging(
    log_dir: str = "logs",
    mode: str = "html", # or "md" for markdown
) -> Tuple[Reporter, logging.Logger]:
    """
    Configure logging across modules and return a RichHandler Reporter (console) Reporter and a FileHandler (run.log)
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    reporter = Reporter(log_dir=log_dir, mode=mode)

    # capture INFO, WARNING, ERROR
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    #  to avoid duplicates
    root.handlers.clear()

    console_handler = RichHandler(
        console=reporter.record_console,
        markup=(mode == "html"),
        show_time=False,
        rich_tracebacks=True,
    )
    console_fmt = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_fmt)
    root.addHandler(console_handler)

    # to persist all logs
    file_path = Path(log_dir) / "run.log"
    file_handler = logging.FileHandler(file_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(console_fmt)
    root.addHandler(file_handler)

    # logging.getLogger("problog").setLevel(logging.INFO)
    logging.getLogger("problog").setLevel(logging.WARNING)

    # a named logger 
    runner_logger = logging.getLogger("runner")
    runner_logger.setLevel(logging.INFO)

    return reporter, runner_logger.info
