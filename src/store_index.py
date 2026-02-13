#loading ApI keys for vector data base where we will store our embeddings

from dotenv import load_dotenv
import os
from pinecone import ServerlessSpec
from pinecone import Pinecone
from src.helper import load_pdf_files, text_split, download_embeddings, filter_to_minimal_docs
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

extracted_data = load_pdf_files("../data")
new_filtered_doc=filter_to_minimal_docs(extracted_data)
chunk_text = text_split(new_filtered_doc)
embeddings = download_embeddings()



pinecone_api_key = PINECONE_API_KEY

pc = Pinecone(api_key=pinecone_api_key)


#Creating an Index upstream or space that we will store our vectors

index_name = "medical-rag"
#sanity check
if not pc.has_index(index_name):
    pc.create_index(
        name= index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud='aws',region='us-east-1')

    )

index =pc.Index(index_name)

#Now let store the actual chunks in the vector store upstream

from langchain_pinecone import PineconeVectorStore

docsearch = PineconeVectorStore.from_documents(
    documents=chunk_text,
    embedding=embeddings,
    index_name =index_name
)