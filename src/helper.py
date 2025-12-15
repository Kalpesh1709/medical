from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

def load_pdf_file(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents
def text_split(extract_data):
    text_spliter =RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=20)
    text_chunk =text_spliter.split_documents(extract_data)
    return text_chunk
def Download_hugging_face_embedding():
    embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model