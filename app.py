from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA, StuffDocumentsChain
# from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import os
from fastapi import FastAPI
from pydantic import BaseModel


load_dotenv()  # take environment variables from .env.


def load_pinecone_index():
    pinecone_api_key = os.getenv("PINECONE_API")
    index_collection = os.getenv("INDEX_COLLECTION")
    pc = Pinecone(api_key=pinecone_api_key)
    pinecone_index = pc.Index(index_collection)
    return pinecone_index


llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=1,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=os.getenv("DEEPSEEK_API")
)


embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

index = load_pinecone_index()

vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Create retriever
retriever = vector_store.as_retriever()

# Define prompt for the LLMChain
prompt_template = """You are an helpful assistant that will suggest recipes based on the user queries. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {input}
Helpful Answer:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "input"]
)
llm_chain = llm

# Create StuffDocumentsChain
combine_documents_chain = create_stuff_documents_chain(
    llm=llm_chain,
    document_variable_name="context",
    prompt=PROMPT
)

# Setup LangChain Retrieval QA
# Removed the llm parameter as it's already defined in the combine_documents_chain
qa_chain = create_retrieval_chain(retriever=retriever,
                                  combine_docs_chain=combine_documents_chain)

# load fast api
app = FastAPI()


class RecipeQuery(BaseModel):
    input: str


@app.post("/recommend")
async def recommend_recipe(req: RecipeQuery):
    response = qa_chain.invoke({"input": req.input})
    return response
