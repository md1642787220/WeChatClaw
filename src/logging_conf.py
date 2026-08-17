# 日志配置：统一使用标准 logging + 轮转，便于生产采集。
#
# Author: MADENG
# Reviewer: Li Rongdong
import logging
import sys
from logging.handlers import RotatingFileHandler


# 初始化根日志器。
#
# 参数：
#     level: 日志级别。
#     log_file: 可选日志文件路径；为空则仅输出到 stdout（容器场景）。
def setup_logging(level="INFO", log_file=None):
    root_logger = logging.getLogger()
    root_logger.setLevel(level.upper())

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # 清空已有 handler，避免重复
    existing_handlers = list(root_logger.handlers)
    for old_handler in existing_handlers:
        root_logger.removeHandler(old_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    if log_file:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
