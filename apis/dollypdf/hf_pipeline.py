from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline
from apis.dollypdf import load_pipeline

# template for an instrution with no input
prompt = PromptTemplate(
    input_variables=["instruction"],
    template="{instruction}")

# template for an instruction with input
prompt_with_context = PromptTemplate(
    input_variables=["instruction", "context"],
    template="{instruction}\n\nInput:\n{context}")

hf_pipeline = HuggingFacePipeline(pipeline=load_pipeline.generate_text)