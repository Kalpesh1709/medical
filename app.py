import os
from langchain_groq import ChatGroq
from flask import Flask,render_template,jsonify,request
from src.helper import Download_hugging_face_embedding
from langchain_pinecone.vectorstores import Pinecone
from langchain_classic.chains import create_retrieval_chain,RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from src.prompt import *
from src.helper import *


app =Flask(__name__)

load_dotenv()
PINECONE_API_KEY=os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY =os.environ.get("GROQ_API_KEY")
os.environ["PINECONE_API_KEY"]= PINECONE_API_KEY
os.environ["GROQ_API_KEY"]= GROQ_API_KEY

embeddings=Download_hugging_face_embedding()

index_name="medicalbot"
docsearch =Pinecone.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever =  docsearch.as_retriever(
        search_type="similarity",
        search_kwargs={"k":3}
    )

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    max_tokens=500,
    temperature=0
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt), 
    ("human", "{input}")
])

# Helper function to format retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Build the RAG chain correctly
rag_chain = (load_dotenv(),
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
@app.route("/")
def index():
    return render_template('chat.html')
@app.route("/get",methods=["GET","POST"])
def chat():
    msg = request.form["msg"]
    print(msg)

    data = ""
    for chunk in rag_chain.stream(msg):
        data += str(chunk)

    return data



if __name__=='__main__':
    app.run(host="0.0.0.0",port=8080,debug=True)