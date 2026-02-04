from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity 
import numpy as np

load_dotenv()
document=""

query='tell me about bumrah'

query_embedding=embedding.embed_query(query)
scores=cosine_similarity([query_embedding],doc_embeddings[0])
index,score=sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]
print(query)
print(documents[index])
