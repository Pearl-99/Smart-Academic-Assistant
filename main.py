import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import tempfile
import os
import json

# -------------------- Page Configuration --------------------
st.set_page_config(page_title="Smart Academic Assistant", layout="centered")

# -------------------- Title --------------------
st.title("Smart Academic Assistant")
st.write("Upload your academic documents and ask questions to get structured answers.")

# -------------------- File Upload Section --------------------
uploaded_files = st.file_uploader(
    "Upload academic documents (PDF, DOCX, or TXT):",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

# -------------------- Question Input --------------------
question = st.text_input("Enter your academic question:")

# -------------------- Submit Button --------------------
if st.button("Get Answer"):
    if not uploaded_files or not question:
        st.warning("Please upload at least one document and enter a question.")
    else:
        # -------------------- PLACEHOLDER: RAG Pipeline Logic --------------------
        
        # 1. Load documents using LangChain document loaders
        documents = []

        for file in uploaded_files:
            file_name = file.name.lower()

            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name)[1]) as tmp_file:
                tmp_file.write(file.read())
                tmp_path = tmp_file.name

            if file_name.endswith(".pdf"):
                loader = PyPDFLoader(tmp_path)
            elif file_name.endswith(".docx"):
                loader = Docx2txtLoader(tmp_path)
            elif file_name.endswith(".txt"):
                loader = TextLoader(tmp_path)

            docs = loader.load()

            # Save original uploaded filename
            for doc in docs:
                doc.metadata["source"] = file.name

            documents.extend(docs)

        st.success("Document Loaded")
        info_box = st.empty()
        info_box.info("Thinking...")

        # 2. Split documents using RecursiveCharacterTextSplitter or similar

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

        chunks = splitter.split_documents(documents)

        # 3. Create embeddings and store in vector store (e.g., FAISS, Chroma)
       
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectorstore = FAISS.from_documents(chunks, embedding_model)

        # 4. Retrieve relevant chunks based on the question
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.invoke(question)

        # 5. Use Groq-hosted LLM via LangChain (e.g., Mixtral, Gemma, Llama3)
        load_dotenv()

        model=ChatGroq(model="openai/gpt-oss-120b")

        # Build the prompt template
        template = """
        You are an academic assistant.

        Answer the question ONLY using the given context.

        If the answer is not available in the context, reply:
        "I could not find the answer in the uploaded documents."

        Context:
        {context}

        Question:
        {question}

        Return ONLY the JSON object.

        Do NOT wrap the JSON inside markdown code blocks.

        {{
            "question":"{question}",
            "answer":"",
            "source_document":"",
            "confidence_score":"High/Medium/Low"
        }}
        """

        prompt = PromptTemplate.from_template(template)

        # 6. Use Output Parser to format structured response

        parser=StrOutputParser()
        
        chain = prompt | model | parser

        context = ""

        for doc in relevant_docs:
            source = doc.metadata.get("source", "Unknown")
            context += f"Source: {source}\n"
            context += doc.page_content + "\n\n"

        response = chain.invoke({"question": question, "context": context})
        info_box.empty()
        st.subheader("Answer")

        try:
            result = json.loads(response)

            st.write(result["question"])
            st.write(result["answer"])
            st.success(f"Confidence: {result['confidence_score']}")
            st.info(f"Source Document: {result['source_document']}")

        except:
            st.write(response)

        st.subheader("Sources")

        for doc in relevant_docs:

            st.write(doc.metadata.get("source"))

            st.caption(doc.page_content[:250]+"...")


        
