import os
from torch import cuda
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from pinecone import Pinecone as PC
from langchain_community.vectorstores import Pinecone
from sentence_transformers import SentenceTransformer


def load_pdf():
    pdf_folder_path = "texts_contract_law"
    data = []
    for file in os.listdir(pdf_folder_path):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder_path, file)
            loader = PyPDFLoader(pdf_path)
            data.extend(loader.load())
    return data


def split_data():
    data = load_pdf()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    docs = text_splitter.split_documents(data)
    return docs


def generate_embeddings():

    embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'

    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

    embeddings = HuggingFaceEmbeddings(
        model_name=embed_model_id,
        model_kwargs={'device': device},
        encode_kwargs={'device': device, 'batch_size': 32}
    )

    return embeddings


def get_embeddings():
    embeddings = generate_embeddings()
    os.environ['PINECONE_API_KEY'] = 'api_key'
    # pc = PC(
    #     api_key='9a27892a-f502-4fe2-937b-a19d177faa25',
    #     environment='gcp-starter'
    # )
    index = "contract-law"
    # for the 1st time when index is empty
    # docsearch = Pinecone.from_texts(
    #     [t.page_content for t in docs],
    #     embeddings,
    #     index_name=index)

    # Loading from existing index
    print("Getting Embeddings......")
    docsearch = Pinecone.from_existing_index(index, embeddings)
    print("Embeddings loaded")
    return docsearch
