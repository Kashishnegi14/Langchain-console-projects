import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_community.output_parsers import StructuredOutputParser, ResponseSchema




load_dotenv()

model=ChatOpenAI()

schema=[
    ResponseSchema(name='fact_1',description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2',description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3',description='Fact 3 about the topic')
]

parser=StructuredOutputParser.from_response_schemas(schema)

template=PromptTemplate(
    template='Give 3 facts about the {topic} \n {format_instructions}',
    input_variables=['topic'],
    partial_variables={'format_instructions':parser.get_format_instructions}
)

chain=template|model|parse
result=chain.invoke({"topic":"black hole"})
print(result)