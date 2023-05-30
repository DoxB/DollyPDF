from langchain.embeddings import Embeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from dollypdf import load_pdf

from transformers import AutoModel, AutoTokenizer

def load_local_embedding_model(model_path):
    # 모델 로드
    model = AutoModel.from_pretrained(model_path)
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

model_path = '../../local_models/ko-sroberta-multitask'

documents = load_pdf.loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

model, tokenizer = load_local_embedding_model(model_path)
embeddings = Embeddings(model, tokenizer)


result_db = Chroma.from_documents(docs, embeddings)