import torch
from ctransformers import AutoModelForCausalLM as cAutoModelForCausalLM
from torch import cuda
from transformers import pipeline, AutoTokenizer, BitsAndBytesConfig,AutoModelForCausalLM, AutoConfig
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA


def load_llm():

    device = 'cuda:0' if cuda.is_available() else 'cpu'
    model_path_gpu = "mistralai/Mistral-7B-Instruct-v0.1"
    model_path_cpu = "models/mistral-7b-instruct-v0.1.Q2_K.gguf"

    if device == "cpu":
        print("Setting up model for cpu")
        model = cAutoModelForCausalLM.from_pretrained(
            model_path_cpu,
            model_type="mistral",
            gpu_layers=0,
            hf=True
        )
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True
        )
        print("Setting up model for gpu")

        model_config = AutoConfig.from_pretrained(model_path_gpu)
        model = AutoModelForCausalLM.from_pretrained(
            model_path_gpu,
            trust_remote_code=True,
            config=model_config,
            quantization_config=bnb_config,

        )

    tokenizer = AutoTokenizer.from_pretrained(model_path_gpu)


    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_new_tokens=500,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    return llm


def retriever_engine(llm, docsearch):
    retriever = docsearch.as_retriever(search_type="mmr", search_kwargs={"k": 4})
    # create a retrieval QA Chain
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


