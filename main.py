from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from agent import run_turn

app = FastAPI(title="SHL Conversational Assessment Recommender")

MAX_TURNS = 8


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool = False


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    messages = [m.model_dump() for m in req.messages][-MAX_TURNS:]
    result = run_turn(messages)
    return ChatResponse(**result)
