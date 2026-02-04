from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableLambda


def word_counter(text):
    return len(text.split())

load_dotenv()

prompt1=PromptTemplate(
    template="Write a joke on {topic}",
    input_variables=['topic']

)
model=ChatOpenAI()
parser=StrOutputParser()

joke_gen_chain=RunnableSequence(prompt1,model,parser)

parallel_chain=RunnableParallel({
    "joke":RunnablePassthrough(),
    "word_counter":RunnableLambda(word_counter)
})
final_chain=RunnableSequence(joke_gen_chain,parallel_chain)
print(final_chain.invoke({"topic":"ai"}))
