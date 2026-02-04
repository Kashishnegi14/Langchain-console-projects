from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

Prompt=PromptTemplate(
    template="Give the summary of each curriculum \n {curriculum}",
    input_variables=['curriculum']
)

model=ChatOpenAI()

parser=StrOutputParser()

chain=Prompt|model|parser
loader=PyPDFLoader("dl-curriculum.pdf")
docs=loader.load()


result=chain.invoke({"curriculum":docs[0].page_content})
print(result)
print(len(docs))