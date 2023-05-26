from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("../../test.pdf")
pages = loader.load_and_split()