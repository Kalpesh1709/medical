from src.helper import load_pdf_file,text_split,Download_hugging_face_embedding
from langchain_pinecone.vectorstores import Pinecone
from pinecone import ServerlessSpec,pinecone
from dotenv import load_dotenv
load_dotenv()
import os
PINECONE_API_KEY=os.environ.get("PINECONE_API_KEY")
os.environ("PINECONE_API_KEY") =PINECONE_API_KEY


extract_data =load_pdf_file(data="Data/")
text_chunk =text_split()
embedding= Download_hugging_face_embedding()
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medicalbot"


pc.create_index(
    name=index_name,
    dimension=384, 
    metric="cosine", 
    spec=ServerlessSpec(
        cloud="aws", 
        region="us-east-1"
    ) 
) 
docsearch = Pinecone.from_documents(
    documents= text_chunks,
    index_name =index_name,
    embedding= embeddings
)