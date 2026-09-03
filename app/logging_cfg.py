"""verbatim_mem 的 stdout 日志。"""

import logging
import sys


def setup_logging() -> None:
    """给 root logger 挂 stdout INFO Formatter。"""
    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    root.addHandler(handler)
    root.setLevel(logging.INFO)
