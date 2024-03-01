from fastapi import FastAPI
from pydantic import BaseModel
from LLM_mistral7b_setup import answer_query, load_llm
from embeddings import get_embeddings_from_existing_index

app = FastAPI(title="LegalChatbot")

llm = load_llm()
docsearch = get_embeddings_from_existing_index()

class Response(BaseModel):
    answer: str


@app.post("/api/chat", response_model=Response)
async def chat(request: str):
    print("In API endpoint")
    result = answer_query(request, llm, docsearch)
    return Response(answer=result)

