import os
import google.generativeai as genai
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
import streamlit as st
import time

# Load environment variables (ensure you have a .env file with GOOGLE_API_KEY)
from dotenv import load_dotenv
load_dotenv()

# Configure the Google Generative AI API key securely
genai.configure(api_key="AIzaSyAlP1dXEG4MEWXLlEVA1F7aCumVo08zwD8")

# Create the model with the generation configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

# Load the CSV file into a DataFrame
df = pd.read_csv("knowledge_base.csv")

# Convert the DataFrame into a list of Documents
documents = [Document(page_content=f"Q: {row['Question']} A: {row['Answer']}") for _, row in df.iterrows()]

# Extract questions for reference
questions = df['Question'].tolist()

# Initialize the HuggingFace embedding function
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Extract raw text from documents
texts = [doc.page_content for doc in documents]

# Create FAISS vector store directly from texts using the embedding function
vector_store = FAISS.from_texts(texts=texts, embedding=embedding_function)

# Set up a retriever using FAISS
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Define the prompt for generation
def build_prompt(question, context):
    return f"""
    You are an intelligent assistant helping answer questions based on the provided documents.

    Question: {question}
    Context:
    {context}

    Answer:
    """

# Function to generate a response using the generative model
def generate_response(prompt):
    chat_session = model.start_chat(history=[])
    response = chat_session.send_message(prompt)
    return response.text

# Define a function to query the bot
def ask_bot(question):
    retrieved_docs = retriever.get_relevant_documents(question)

    if not retrieved_docs:
        return "I couldn't find relevant information in the knowledge base."

    context = "\n".join([doc.page_content for doc in retrieved_docs])
    formatted_prompt = build_prompt(question, context)
    response = generate_response(formatted_prompt)
    
    return response

# Initialize session state for chat history
if "history" not in st.session_state:
    st.session_state["history"] = []

# Streamlit UI Layout
st.set_page_config(page_title="RAG Chatbot", layout="wide")

# Creating three vertical sections
col1, col2, col3 = st.columns([1, 2, 1])  # Adjust ratio as needed

# Left Zone - About & Data
with col1:
    st.header("📌 About Us")
    st.write('''TechNova is an AI-driven technology company specializing in intelligent chatbots, knowledge management systems, and automation solutions. Our mission is to enhance information retrieval and customer interactions using advanced AI models. With expertise in Retrieval-Augmented Generation (RAG) and deep learning, we develop innovative solutions that provide accurate, context-aware responses, helping businesses and users access the right information effortlessly. ''')
    
    st.subheader("📂 Knowledge Base")
    with st.expander("View Knowledge Base"):
        st.dataframe(df)

    st.download_button("Download Knowledge Base", df.to_csv(index=False), "knowledge_base.csv", "text/csv")

# Middle Zone - Chatbot
with col2:
    st.header("💬 RAG Chatbot")
    
    question = st.text_input("Ask a question:", placeholder="Type your question here...")
    
    if st.button("Get Answer"):
        if question:
            with st.spinner("Fetching answer..."):
                answer = ask_bot(question)
            
            # Store the question and answer in history
            st.session_state["history"].append((question, answer))
        else:
            st.warning("Please enter a question.")
    
    # Display chat history
    if st.session_state["history"]:
        st.subheader("📜 Chat History")
        for q, a in reversed(st.session_state["history"]):
            st.markdown(f"**Q:** {q}")
            st.markdown(f"**A:** {a}")
            st.markdown("---")

# Right Zone - Additional Info
with col3:
    st.header("🛠️ How It Works")
    st.write(
        """
        1. Enter your question in the chat.
        2. The bot retrieves relevant knowledge base content.
        3. It generates a response using Google Gemini.
        4. The answer is displayed here!
        """
    )

    with st.spinner("Processing..."):
        time.sleep(3)
        st.success("Ready to answer your queries!")

    # Tabs for additional information
    tab1, tab2 = st.tabs(["📖 About RAG", "⚙️ Technical Details"])
    with tab1:
        st.write("RAG combines retrieval-based and generative AI to improve chatbot accuracy.")
    with tab2:
        st.write("Using FAISS for retrieval and Gemini for generation.")
