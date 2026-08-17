"""从 HF 缓存复制模型到项目 models/ 目录（离线固化，无需网络）。

优先使用已下载到本地缓存的模型；若缓存不存在，则回退到镜像下载。

Author: MADENG
Reviewer: Li Rongdong
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import scan_cache_dir, snapshot_download  # noqa: E402

MODELS_DIR = Path("models")


# 在 HF 缓存中查找模型，返回最新快照目录。
#
# Args:
#     repo_id: 仓库 id（如 ``BAAI/bge-small-zh-v1.5``）。
#
# Returns:
#     最新快照的本地路径；未找到或扫描失败时返回 ``None``。
def _resolve_cache_path(repo_id: str) -> Path | None:
    try:
        for repo in scan_cache_dir().repos:
            if repo.repo_id == repo_id and repo.revisions:
                latest = sorted(repo.revisions, key=lambda r: r.last_modified, reverse=True)[0]
                return Path(latest.snapshot_path)
    except Exception:  # noqa: BLE001
        return None
    return None


# 递归复制目录（目标存在则先删除）。
#
# Args:
#     src: 源目录。
#     dst: 目标目录。
def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


# 固话模型到 models/ 目录：优先从本地缓存复制，否则镜像下载。
#
# Args:
#     model_name: 模型名。
#     endpoint: 可选自定义镜像地址。
#     force: True 时强制重新固化（即使目标已存在）。
#
# Returns:
#     固化后的本地目录路径。
def download(model_name: str, endpoint: str | None = None, force: bool = False) -> Path:
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint

    target = MODELS_DIR / model_name.replace("/", "__")
    if target.exists() and any(target.iterdir()) and not force:
        print(f"[跳过] 模型已存在：{target}")
        return target

    # 1) 优先从本地缓存复制
    cached = _resolve_cache_path(model_name)
    if cached and cached.exists():
        target.mkdir(parents=True, exist_ok=True)
        _copy_tree(cached, target)
        print(f"[完成] 从本地缓存复制：{cached} -> {target}")
        return target

    # 2) 缓存没有，走镜像下载（禁用 Xet，避免大文件 401/超时）
    print(f"[下载] {model_name} -> {target} (endpoint={os.environ.get('HF_ENDPOINT')})")
    tmp = target.with_suffix(".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    snapshot_download(repo_id=model_name, local_dir=str(tmp))
    if target.exists():
        shutil.rmtree(target)
    tmp.rename(target)
    print(f"[完成] 模型已固化到 {target.resolve()}")
    return target


# 命令行入口：解析参数并执行 :func:`download`。
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="固话 embedding 模型到本地")
    parser.add_argument("--model", default="BAAI/bge-small-zh-v1.5", help="模型名")
    parser.add_argument("--endpoint", default=None, help="自定义镜像地址")
    parser.add_argument("--force", action="store_true", help="强制重新固化")
    args = parser.parse_args()

    download(args.model, args.endpoint, args.force)


if __name__ == "__main__":
    main()
