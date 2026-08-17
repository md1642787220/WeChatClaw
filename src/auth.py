"""管理员鉴权：保护 /kb/* 管理接口。

设计：
  - token 来自 settings.admin.token（通过环境变量 ADMIN_TOKEN 或 .env 注入）。
  - token 为空表示关闭鉴权（仅本地开发允许）。
  - 客户端在请求头携带 X-Admin-Token: <token>。
  - 校验失败返回 403，避免泄露接口细节（不返回 401 以弱化接口暴露）。

Author: MADENG
Reviewer: Li Rongdong
"""
from fastapi import Header, HTTPException, status

from src.config import read_settings

# 403 异常单例：校验失败时抛出（不返回 401，弱化接口暴露）
_FORBIDDEN_EXCEPTION = HTTPException(
    status_code=status.HTTP_403_FORBIDDEN,
    detail="需要管理员令牌：请在请求头携带 X-Admin-Token。",
)


# FastAPI 依赖：校验管理员令牌。
#
# 参数：
#     x_admin_token: 请求头 X-Admin-Token 的值（由 FastAPI 自动注入）。
#
# 注意：
#     - 没配置 token（开发模式）时直接放行。
#     - 返回 403 而非 401，弱化接口暴露。
def require_admin(x_admin_token=Header(default=None)):
    settings = read_settings()
    expected_token = settings.admin.token
    # 开发模式：没配置 token 就放行（warning 已在启动日志提示）
    if not expected_token:
        return
    # 已配置 token：必须匹配
    if x_admin_token is None:
        raise _FORBIDDEN_EXCEPTION
    if x_admin_token != expected_token:
        raise _FORBIDDEN_EXCEPTION
