from typing import List, Optional

from pydantic import BaseModel


class PromptTemplate(BaseModel):
    system: str
    user: Optional[str] = None
