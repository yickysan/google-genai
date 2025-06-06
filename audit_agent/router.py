from typing import Literal
from pydantic import BaseModel, Field


class Router(BaseModel):
    """
    A simple router to route between states.
    """
    step: Literal["audit_assistant", "search_web", "vectorstore"] = Field(None, description="The step to route to")