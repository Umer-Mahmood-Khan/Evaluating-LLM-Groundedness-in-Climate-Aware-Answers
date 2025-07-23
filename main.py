import os
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from response_logger import save_responses_as_json
from langchain.prompts import PromptTemplate

# Load API Key from .env file
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load all .txt files inside data/ folder
file_paths = glob.glob('data/*.txt')
all_docs = []

for file_path in file_paths:
    loader = TextLoader(file_path)
    docs = loader.load()
    all_docs.extend(docs)

print(f"Loaded {len(all_docs)} documents from {len(file_paths)} files.\n")

# Split and embed documents
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
chunks = splitter.split_documents(all_docs)
print(f"Total chunks created: {len(chunks)}\n")

# Build FAISS vector store
vectordb = FAISS.from_documents(chunks, OpenAIEmbeddings())
retriever = vectordb.as_retriever()

# ✅ Set top-k retrieved chunks
retriever.search_kwargs["k"] = 5  # Retrieve top 2 most relevant chunks

# Initialize ChatGPT model
llm = ChatOpenAI(model_name="gpt-4.1-nano", temperature=0)

# Setup RAG chain with custom prompt
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    You are a Python tutor.

    Answer the following question based only on the provided context. Use your own words if needed, but only draw information from the context.

    If the context does not contain relevant information, you can say so.

    Question:
    {question}

    Context:
    {context}

    Answer:
    """
)

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": custom_prompt}
)

# Example question (changeable)
question = "What is the quantified impact of a 1°C temperature increase on cotton lint yield in Pakistan?"

print(f"User Question: {question}\n")

# Pure ChatGPT Response (No Retrieval)
pure_response = llm.predict(question)
print("=== Pure ChatGPT Response ===\n")
print(pure_response)

# Manual Retrieval to View Chunks
retrieved_docs = retriever.get_relevant_documents(question)

print("\n=== Retrieved Chunks (Before RAG Answer) ===\n")
for i, doc in enumerate(retrieved_docs):
    print(f"Chunk {i+1}:\n{doc.page_content}\n{'-'*50}")

# RAG Response (With Retrieved Context)
rag_response = rag_chain.run(question)

print("\n=== RAG-Powered Response ===\n")
print(rag_response)

# Save responses to JSON
save_responses_as_json(
    filename="responses.json",
    question=question,
    retrieved_chunks=retrieved_docs,
    pure_response=pure_response,
    rag_response=rag_response
)
