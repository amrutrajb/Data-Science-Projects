import streamlit as st 
from langchain.text_splitter import CharacterTextSplitter 
from langchain.vectorstores import FAISS 
from langchain.chains import RetrievalQA 
from langchain.embeddings.base import Embeddings 
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline 
from langchain.llms import HuggingFacePipeline 
from langchain.prompts import PromptTemplate 
from langchain_core.documents import Document
import os 
import json 
from typing import List


class RAGChatbotApp:
    class TfidfEmbedder(Embeddings):
        def __init__(self, texts: List[str]):
            self.vectorizer = TfidfVectorizer()
            self.vectorizer.fit(texts)

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            return self.vectorizer.transform(texts).toarray().tolist()

        def embed_query(self, text: str) -> List[float]:
            return self.vectorizer.transform([text]).toarray()[0].tolist()

    @st.cache_resource 
    def prepare_rag_pipeline(txt_path): 
        with open(txt_path, "r", encoding="utf-8") as f: 
            text = f.read()

        documents = [Document(page_content=text)]
        splitter = CharacterTextSplitter(separator="#", chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        texts = [doc.page_content for doc in chunks]
        embedder = RAGChatbotApp.TfidfEmbedder(texts)
        vectors = embedder.embed_documents(texts)
        db = FAISS.from_embeddings(list(zip(texts, vectors)), embedder)

        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
        pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)
        llm = HuggingFacePipeline(pipeline=pipe)

        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=""" 
    You are an AI assistant specialized in answering questions based on provided context.
    only use the context to answer the question.from the document given 

    Context: {context}

    Question: {question}

    Answer: """ )

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=db.as_retriever(),
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt_template}
        )
        return qa

    def answer_from_dict(user_query, qa_dict): 
        return qa_dict.get(user_query.lower().strip())

    def run():
        st.title("📄  Personal AI-Powered Document RAG Chatbot")

        uploaded_file = st.file_uploader("Upload a QA Dictionary (.json)", type="json")

        if "qa_dict" not in st.session_state: 
            st.session_state.qa_dict = {}  # Safe initialization

        if uploaded_file: 
            content = uploaded_file.read().decode("utf-8")
            try: 
                st.session_state.qa_dict = json.loads(content) 
                st.success("📄 Dictionary loaded successfully.") 
            except Exception as e:
                st.error("Failed to parse JSON: " + str(e))

        if "qa" not in st.session_state and st.session_state.qa_dict: 
            with open("temp.txt", "w", encoding="utf-8") as f: json.dump(st.session_state.qa_dict, f) 
            st.session_state.qa = RAGChatbotApp.prepare_rag_pipeline("temp.txt")

        if "chat_active" not in st.session_state: st.session_state.chat_active = True

        query = None
        if st.session_state.chat_active and "qa" in st.session_state and st.session_state.qa_dict:
            query = st.text_input("Ask a question based on the document (type 'quit' to exit):")

        if query:
            if query.lower() == "quit":
                st.session_state.chat_active = False
                st.success("Chat ended. You can refresh to start again.")
            else:
                st.markdown("### 📤 Prompt:")
                st.code(query)
                # Try dictionary-based answer first
                response = RAGChatbotApp.answer_from_dict(query, st.session_state.qa_dict)
                if response is None:
                    response = st.session_state.qa.run(query)
                st.markdown("### 💬 Answer:")
                st.write(response)

# Run the app
if __name__ == "__main__":
    RAGChatbotApp.run()
