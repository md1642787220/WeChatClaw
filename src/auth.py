"""管理员鉴权：保护 /kb/* 管理接口。

设计：
  - token 来自 settings.admin.token（通过环境变量 ADMIN_TOKEN 或 .env 注入）。
  - 为空 token 表示关闭鉴权（仅本地开发允许）。
  - 客户端在请求头携带 ``X-Admin-Token: <token>``。
  - 校验失败返回 403，避免泄露接口细节（不返回 401 以弱化接口暴露）。

Author: MADENG
Reviewer: Li Rongdong
"""
from __future__ import annotations

from fastapi import Header, HTTPException, status

from src.config import load_settings

_FORBIDDEN = HTTPException(
    status_code=status.HTTP_403_FORBIDDEN,
    detail="需要管理员令牌：请在请求头携带 X-Admin-Token。",
)


# FastAPI 依赖：校验管理员令牌。
#
# Args:
#     x_admin_token: 请求头 ``X-Admin-Token`` 的值（由 FastAPI 自动注入）。
#
# Returns:
#     无返回值；校验通过则正常返回，失败则抛出 403 异常。
#
# Notes:
#     - 未配置 token（开发模式）时直接放行。
#     - 返回 403 而非 401，弱化接口暴露。
def require_admin(x_admin_token: str | None = Header(default=None)) -> None:
    expected = load_settings().admin.token
    # 开发模式：未配置 token 则放行（warning 已在启动日志提示）
    if not expected:
        return
    if not x_admin_token or x_admin_token != expected:
        raise _FORBIDDEN
