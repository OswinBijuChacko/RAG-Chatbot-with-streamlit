# 🤖 RAG Chatbot with Streamlit  

This is a **Retrieval-Augmented Generation (RAG) chatbot** that uses **FAISS for retrieval** and **Google Gemini API** for intelligent response generation. The chatbot allows users to query a **knowledge base (CSV file)** and receive **context-aware AI-generated answers**.  

---

## 📌 Features  
- **Retrieval-Augmented Generation (RAG)** for intelligent responses  
- **FAISS-based vector search** for efficient document retrieval  
- **Google Gemini API** for generating human-like answers  
-  **Streamlit UI** with an interactive chatbot  
-  **Downloadable knowledge base** for transparency  
- **Fast, scalable, and easy to use**  

---

## 🛠️ How It Works  
1. **User enters a query** in the chatbot.  
2. The system **retrieves relevant context** from the knowledge base using FAISS.  
3. The **query + retrieved data** are passed to **Google Gemini API**.  
4. The chatbot **displays the generated answer** interactively.  

---

## 🚀 Installation  

### 1️⃣ Clone the repository  
```bash
git clone https://github.com/your-username/rag-chatbot.git
cd rag-chatbot
