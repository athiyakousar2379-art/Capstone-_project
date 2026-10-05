import streamlit as st
from pypdf import PdfReader
from pathlib import Path
import requests

from rag import add_document, search_documents


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MEETINGS_DIR = BASE_DIR.parent / "meetings"

MEETINGS_DIR.mkdir(exist_ok=True)


# -----------------------------
# Page
# -----------------------------
st.set_page_config(
    page_title="Meeting Agenda Assistant",
    page_icon="🤖"
)

st.title("🤖 Meeting Agenda Assistant Bot")
st.write("Upload meeting notes and ask questions about them.")


# -----------------------------
# PDF Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a meeting PDF",
    type=["pdf"]
)


if uploaded_file is not None:

    # Save PDF
    pdf_path = MEETINGS_DIR / uploaded_file.name

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"Uploaded: {uploaded_file.name}")

    # Extract text
    reader = PdfReader(str(pdf_path))

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"


    # Split text into chunks
    chunk_size = 1000

    chunks = [
        text[i:i + chunk_size]
        for i in range(0, len(text), chunk_size)
    ]


    # Add chunks to ChromaDB
    for i, chunk in enumerate(chunks):

        document_id = f"{uploaded_file.name}_{i}"

        add_document(
            chunk,
            document_id
        )

    st.success(
        f"Meeting document processed successfully! "
        f"{len(chunks)} chunks added."
    )


# -----------------------------
# Question
# -----------------------------
st.subheader("Ask about the meeting")

question = st.text_input(
    "Enter your question"
)


if st.button("Ask Assistant"):

    if not question:
        st.warning("Please enter a question.")

    else:

        # RAG retrieval
        results = search_documents(
            question,
            n_results=3
        )

        documents = results["documents"][0]

        context = "\n\n".join(documents)


        # Prompt for Ollama
        prompt = f"""
You are a helpful Meeting Assistant.

Use ONLY the meeting information provided below.

Meeting information:
{context}

User question:
{question}

Give a clear and concise answer.
If the information is not available in the meeting notes,
say that it is not mentioned in the meeting notes.
"""


        # Ask Ollama
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": prompt,
                "stream": False
            }
        )


        if response.status_code == 200:

            answer = response.json()["response"]

            st.subheader("Assistant Answer")
            st.write(answer)

        else:

            st.error(
                "Could not connect to Ollama. "
                "Make sure Ollama is running."
            )