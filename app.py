from flask import Flask, render_template, jsonify, request
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
load_dotenv()


# initialise flask
app = Flask(__name__)

# Groq api key load 

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY


#embedding models

embeddings = download_embeddings()


index_name = "medical-rag"
docsearch = PineconeVectorStore.from_existing_index(
    embedding=embeddings,
    index_name =index_name
)


#Now since we have the vectors in the db lets create the retriever for it , it picks top 3 that matches the questions

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

#now create the llm we will use to bind the retriever
chatModel = ChatGroq(model='llama-3.3-70b-versatile')


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human","{input}")
    ]
)

# now lets chain them 

question_answer_chain = create_stuff_documents_chain(chatModel,prompt)

rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/chat", methods =["GET","POST"])
def chat():
    msg = request.form["msg"]
    input =msg
    print(input)
    response = rag_chain.invoke({"input":msg})
    print("Response: ", response["answer"])
    return str(response["answer"])

if __name__=="__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
