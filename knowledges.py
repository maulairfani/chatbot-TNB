from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb
import uuid
from utils import *
import os
from langchain_community.vectorstores import Chroma

knowledge_folder = 'knowledges'

def add_documents():
    loader = DirectoryLoader(os.path.join(knowledge_folder, 'unsaved'), 
                             glob="**/*.pdf", 
                             show_progress=True, 
                             use_multithreading=True)
    docs = loader.load()

    splitter = TokenTextSplitter(chunk_size=2000, chunk_overlap=0)
    chunks = splitter.split_documents(docs)

    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    client = chromadb.PersistentClient()
    collection = client.get_or_create_collection(name="TNB", metadata={"hnsw:space": "cosine"})

    ids = [str(uuid.uuid1()) for i in range(len(chunks))]
    documents = [chunks[i].page_content for i in range(len(chunks))]
    metadata = [chunks[i].metadata for i in range(len(chunks))]

    print("Generating embeddings...")
    embeddings = embedding_function.embed_documents(documents)
    print("DONE")

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadata
    )

    source_dir = os.path.join(knowledge_folder, 'unsaved')
    destination_dir = os.path.join(knowledge_folder, 'saved')
    for file in os.listdir(source_dir):
        move_pdf(source_dir, destination_dir, file)

    print("Adding Documents Successfull")

def delete_document(filename):
    persistent_client = chromadb.PersistentClient()
    vectorstore = Chroma( client=persistent_client, collection_name="TNB" )
    
    data = vectorstore.get()
    metadatas = data['metadatas']
    ids = data['ids']

    ids_to_delete = [ids[i] for i in range(len(metadatas)) if filename in metadatas[i]['source']]

    for i in range(len(ids_to_delete)):
        vectorstore.delete(ids_to_delete[i])

    source_dir = os.path.join(knowledge_folder, 'saved')
    destination_dir = os.path.join(knowledge_folder, 'deleted')
    move_pdf(source_dir, destination_dir, filename)

    print(f"Successfully deleting {filename}")

def count_documents():
    persistent_client = chromadb.PersistentClient()
    vectorstore = Chroma( client=persistent_client, collection_name="TNB" )
    count = vectorstore._collection.count()
    print("There are", count, "in the collection")
    return count