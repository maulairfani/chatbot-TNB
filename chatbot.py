from operator import itemgetter

from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb
from dotenv import load_dotenv
import os
load_dotenv()

persistent_client = chromadb.PersistentClient()
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
vectorstore = Chroma(
    client=persistent_client,
    collection_name="TNB",
    embedding_function=embedding_function,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
model = ChatOpenAI(openai_api_key=os.environ.get('OPENAI_API_KEY'))

# Prompts
template = """Answer the question based only on the following context:
{context}

Question: {question}

Answer in the following language: {language}
"""
prompt = ChatPromptTemplate.from_template(template)

# Chains
chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
        "language": itemgetter("language"),
    }
    | prompt
    | model
    | StrOutputParser()
)

def generate_answer(question):
    return chain.invoke({"question": question, "language": "indonesia"})