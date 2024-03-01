import torch
from torch import cuda
from transformers import pipeline, AutoTokenizer, BitsAndBytesConfig,AutoModelForCausalLM, AutoConfig
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA


def load_llm():

    device = 'cuda:0' if cuda.is_available() else 'cpu'

    if device == "cpu":
        model_id = "models/mistral-7b-instruct-v0.1.Q2_K.gguf"
    else:
        model_id = "mistralai/Mistral-7B-Instruct-v0.1"

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True
    )
    model_config = AutoConfig.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        max_new_tokens=500
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    return llm


def retriever_engine(llm, docsearch):
    retriever = docsearch.as_retriever(search_type="mmr", search_kwargs={"k": 4})
    QnA = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type="stuff"
    )
    return QnA


def answer_query(question, llm, docsearch):
    QnA = retriever_engine(llm, docsearch)
    response = QnA.invoke(question)
    return response['result']


