from LLM_mistral7b_setup import answer_query, load_llm
from embeddings import get_embeddings_from_existing_index

llm = load_llm()
docsearch = get_embeddings_from_existing_index()

while True:
    print("Ask your query. If you wish to exit type 'exit'")
    query = input()
    if query == "exit":
        exit()
    else:
        answer = answer_query(query,llm,docsearch)
        print(answer)
