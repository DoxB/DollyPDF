from langchain.chains.question_answering import load_qa_chain
from apis.dollypdf import hf_pipeline
from apis.dollypdf import split_and_embedding




chain = load_qa_chain(hf_pipeline.hf_pipeline, chain_type="stuff")

query = ''

while query != 'exit': 
    query = input('질문 입력하세여: ')
    chain.run(input_documents=split_and_embedding.result_db, question=query)