# Shared AgentState schema. Do not edit without team review.

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List


class AgentState(BaseModel):
    task_domain: str
    raw_input: str
    round_number: int = 0
    is_validated: bool = False
    error_log: Optional[str] = None
    analysis_payload: Dict[str, Any] = Field(default_factory=dict)
    sanitized_tool_calls: List[str] = Field(default_factory=list)
