# 从 HF 缓存复制模型到项目 models/ 目录（离线固化，不用联网）。
#
# 优先用已经下载到本地缓存的模型；缓存里没有就回退到镜像下载。
#
# Author: MADENG
# Reviewer: Li Rongdong
import os
import shutil
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import scan_cache_dir, snapshot_download  # noqa: E402

MODELS_DIR = Path("models")


# 把 repo_id 变成本地目录名（org__name）。
def _model_dir_name(model_name):
    return model_name.replace("/", "__")


# 判断目录存不存在且不是空的。
def _is_non_empty_dir(path):
    if not path.exists():
        return False
    if not path.is_dir():
        return False
    for _ in path.iterdir():
        return True
    return False


# 在 HF 缓存里找模型，返回最新快照目录。
#
# 参数：
#     repo_id: 仓库 id（比如 BAAI/bge-small-zh-v1.5）。
#
# 返回：
#     最新快照的本地路径；找不到或扫描失败时返回 None。
def _resolve_cache_path(repo_id: str):
    try:
        for repo in scan_cache_dir().repos:
            if repo.repo_id == repo_id:
                if not repo.revisions:
                    continue
                # 按 last_modified 倒序找最新的
                revisions = list(repo.revisions)
                revisions.sort(key=lambda r: r.last_modified, reverse=True)
                latest = revisions[0]
                return Path(latest.snapshot_path)
    except Exception:  # noqa: BLE001
        return None
    return None


# 递归复制目录（目标存在就先删除）。
#
# 参数：
#     src: 源目录。
#     dst: 目标目录。
def _copy_directory_tree(src: Path, dst: Path):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


# 把模型固化到 models/ 目录：优先从本地缓存复制，否则镜像下载。
#
# 参数：
#     model_name: 模型名。
#     endpoint: 可选自定义镜像地址。
#     force: True 时强制重新固化（即使目标还在）。
#
# 返回：
#     固化后的本地目录路径。
def download_model(model_name: str, endpoint=None, force=False):
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint

    target = MODELS_DIR / _model_dir_name(model_name)

    # 目标已存在且非空 而且不强制：直接跳过
    if not force and _is_non_empty_dir(target):
        print("[跳过] 模型已存在：" + str(target))
        return target

    # 1) 优先从本地缓存复制
    cached = _resolve_cache_path(model_name)
    if cached is not None and cached.exists():
        target.mkdir(parents=True, exist_ok=True)
        _copy_directory_tree(cached, target)
        print("[完成] 从本地缓存复制：" + str(cached) + " -> " + str(target))
        return target

    # 2) 缓存没有，走镜像下载（禁用 Xet，避免大文件 401/超时）
    print("[下载] " + model_name + " -> " + str(target) + " (endpoint=" + str(os.environ.get("HF_ENDPOINT")) + ")")
    tmp = target.with_suffix(".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    snapshot_download(repo_id=model_name, local_dir=str(tmp))
    if target.exists():
        shutil.rmtree(target)
    tmp.rename(target)
    print("[完成] 模型已固化到 " + str(target.resolve()))
    return target


# 命令行入口：解析参数并执行下载。
def main():
    import argparse

    parser = argparse.ArgumentParser(description="固化 embedding 模型到本地")
    parser.add_argument("--model", default="BAAI/bge-small-zh-v1.5", help="模型名")
    parser.add_argument("--endpoint", default=None, help="自定义镜像地址")
    parser.add_argument("--force", action="store_true", help="强制重新固化")
    args = parser.parse_args()

    download_model(args.model, args.endpoint, args.force)


if __name__ == "__main__":
    main()
