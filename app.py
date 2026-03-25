from flask import Flask, render_template, request
from dotenv import load_dotenv
import os

# LangChain components
from langchain_community.chat_models import ChatOllama
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Your modules
from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt


# -------------------- App Init --------------------
app = Flask(__name__)
load_dotenv()

# -------------------- Env Setup --------------------
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


# -------------------- Embeddings --------------------
embeddings = download_hugging_face_embeddings()

# -------------------- Vector Store --------------------
index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

from langchain_groq import ChatGroq
import os

chatModel = ChatGroq(
    model="mixtral-8x7b-32768",   # best for RAG
    groq_api_key=os.environ["GROQ_API_KEY"],
    temperature=0.2
)

# -------------------- Prompt --------------------
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


# -------------------- RAG Chain --------------------
question_answer_chain = create_stuff_documents_chain(chatModel, prompt)

rag_chain = create_retrieval_chain(
    retriever,
    question_answer_chain
)


# -------------------- Routes --------------------
@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]

    response = rag_chain.invoke({
        "input": msg
    })

    return str(response["answer"])


# -------------------- Run --------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)