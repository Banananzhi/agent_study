"""业务配置，不读取或初始化 Agent、模型和外部服务。"""

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from media_operations.schemas import OwnerScope


class MediaSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    database_path: Path = Path(".agent_data/media_operations.sqlite3")
    owner: OwnerScope = OwnerScope()

    @classmethod
    def from_env(cls):
        # 环境由入口加载，业务配置不隐式读取 .env 或用户凭据。
        return cls(
            database_path=os.getenv("MEDIA_DATABASE_PATH", str(cls().database_path)),
            owner=OwnerScope(
                tenant_id=os.getenv("AGENT_TENANT_ID", "local"),
                user_id=os.getenv("AGENT_USER_ID", "local-user"),
            ),
        )

