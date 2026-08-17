"""日志配置：统一使用标准 logging + 轮转，便于生产采集。

Author: MADENG
Reviewer: Li Rongdong
"""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler


# 初始化根日志器。
#
# Args:
#     level: 日志级别。
#     log_file: 可选日志文件路径；为空则仅输出到 stdout（容器场景）。
def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    root = logging.getLogger()
    root.setLevel(level.upper())

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # 清空已有 handler，避免重复
    for h in list(root.handlers):
        root.removeHandler(h)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    if log_file:
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
