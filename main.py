import uvicorn

from LLM_mistral7b_setup import load_llm
from embeddings import get_embeddings

if __name__ == "__main__":
    uvicorn.run("app:app", host='localhost', port=3000, reload=True)
