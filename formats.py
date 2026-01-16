from typing import Any
from pydantic import BaseModel, HttpUrl

class EvalRequest(BaseModel):
    participants: dict[str, HttpUrl] # role-endpoint mapping
    config: dict[str, Any]

class EvalResult(BaseModel):
    logs: Any
    resuts: Any
    elo: Any
