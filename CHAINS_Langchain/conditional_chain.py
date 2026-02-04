from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatOpenAI()
parsers = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description="Give the sentiment of the feedback")

parsers2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template="classify the sentiment of the following feedback text into positive or negative \n {feedback}\n {format_instruction}",
    input_variables=['feedback'],
    partial_variables={'format_instruction': parsers2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parsers2

prompt2 = PromptTemplate(
    template="Write an appropriate response to this positive feedback \n {feedback}",
    input_variables=['feedback']
)
prompt3 = PromptTemplate(
    template="Write an appropriate response to this negative feedback \n {feedback}",
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt2 | model | parsers),
    (lambda x: x.sentiment == 'negative', prompt3 | model | parsers),
    RunnableLambda(lambda x: "could not find sentiment")
)

chain = classifier_chain | branch_chain
print(chain.invoke({"feedback": "my name is kashish"}))
