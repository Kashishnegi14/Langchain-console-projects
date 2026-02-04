from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

prompt=PromptTemplate(
    template="Answer the following question \n {question}from the following text - \n {text}",
    input_variables=['question','text']
)

model=ChatOpenAI()

parser=StrOutputParser()

chain=prompt|model|parser

url='https://marscosmetics.in/products/matte-muse-mousse-lipstick?variant=46867020349664'
loader=WebBaseLoader(url)
docs=loader.load()
result = chain.invoke({"question":"What is the name of the lipcstick and its price",'text':docs[0].page_content})
print(result)