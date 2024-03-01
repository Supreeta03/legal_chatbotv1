import json

from fastapi import FastAPI, APIRouter
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from fastapi.encoders import jsonable_encoder
from LLM_mistral7b_setup import answer_query

app = FastAPI(title="LegalChatbot")


# class Request(BaseModel):
#     query: str

class Response(BaseModel):
    answer: str

@app.post("/api/chat", response_model=Response)
async def chat(request: str):
    print("In API endpoint")
    result = answer_query(request)
    return Response(answer=result)

