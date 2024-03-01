import random

import torch
from huggingface_hub import hf_hub_download
from ctransformers import  AutoModelForCausalLM
from transformers import pipeline, AutoTokenizer, BitsAndBytesConfig
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from embeddings import get_embeddings


def load_llm():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


    # for cpu
    model = AutoModelForCausalLM.from_pretrained(
        "models/mistral-7b-instruct-v0.1.Q2_K.gguf",
        model_type="mistral",
        gpu_layers=0,
        hf=True
    )

    # for gpu
    # model_path = "mistralai/Mistral-7B-Instruct-v0.1"
    # print("Setting up model")
    # model_4bit = AutoModelForCausalLM.from_pretrained(
    #     model_path,
    #     quantization_config=bnb_config,
    #     torch_dtype=torch.bfloat16,
    #     device_map="auto",
    #     trust_remote_code=True,
    #     hf=True
    # )

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_new_tokens=100,
        # repetition_penalty=1.1
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    return llm


def retriever_engine():
    docsearch = get_embeddings()
    llm = load_llm()
    retriever = docsearch.as_retriever(search_type="mmr", search_kwargs={"k": 4})
    # create a retrieval QA Chain
    QnA = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type="stuff"
    )
    return QnA


def answer_query(question):
    QnA = retriever_engine()
    response = QnA.invoke(question)
    return response['result']


