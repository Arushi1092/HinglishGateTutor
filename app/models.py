from pydantic import BaseModel
from typing import Optional

class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5

class AskResponse(BaseModel):
    question: str
    answer: str
    language: str
    sources: list[str]

class IngestResponse(BaseModel):
    filename: str
    chunks_added: int
    status: str