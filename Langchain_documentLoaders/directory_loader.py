from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv

loader=DirectoryLoader(
    path="books",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)
docs=loader.load()
for document in docs:
    print(document.metadata)