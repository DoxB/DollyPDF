from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

from dollypdf import load_pdf

documents = load_pdf.loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model = '../embedding_models/ko-sroberta-multitask')

result_db = Chroma.from_documents(docs, embeddings)