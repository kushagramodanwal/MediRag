from dotenv import load_dotenv
import os

from src.helper import (
    load_pdf_file,
    filter_to_minimal_docs,
    text_split,
    download_hugging_face_embeddings
)

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore


# -------------------- ENV --------------------
load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

# -------------------- LOAD DATA --------------------
extracted_data = load_pdf_file(data="data/")
filtered_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filtered_data)

# -------------------- EMBEDDINGS --------------------
embeddings = download_hugging_face_embeddings()

# -------------------- PINECONE INIT --------------------
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medirag"

# -------------------- CREATE INDEX (SAFE CHECK) --------------------
existing_indexes = [index.name for index in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=384,   # MUST match embedding dim
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

# -------------------- UPSERT --------------------
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name,
    namespace="default"   # optional but recommended
)

print("✅ Data successfully indexed into Pinecone")