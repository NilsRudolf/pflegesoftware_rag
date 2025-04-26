# app.py
import os, re, string, unicodedata
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# ───────────────────────────────────────────────────────
# Streamlit-Page-Config
# ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pflegedienst Software Assistent",
    page_icon="🧠",
)

# ───────────────────────────────────────────────────────
# Sidebar: OpenAI-Key
# ───────────────────────────────────────────────────────
st.sidebar.title("🔑 OpenAI-API-Key")

api_key_input = st.sidebar.text_input(
    "Bitte gültigen OpenAI-Key eingeben",
    type="password",
    placeholder="sk-...",
)

def valid_openai_key(key: str) -> bool:
    key = key.strip()
    return key.startswith("sk-") and len(key) >= 24 and all(32 <= ord(c) < 127 for c in key)

if api_key_input and valid_openai_key(api_key_input):
    os.environ["OPENAI_API_KEY"] = api_key_input.strip()
elif not os.getenv("OPENAI_API_KEY"):
    st.sidebar.error("Ungültiger Key – bitte korrekt einfügen.")
    st.stop()

# ───────────────────────────────────────────────────────
# System-Nachricht für jedes LLM-Prompt
# ───────────────────────────────────────────────────────
SYSTEM_MSG = (
    "Du bist ein hilfsbereiter Assistent für die Pflegedienst-Software. "
    "Antworte immer auf Deutsch, kurz und präzise, mit Aufzählungen wenn sinnvoll."
)

# ───────────────────────────────────────────────────────
# Embeddings & Vectorstore
# ───────────────────────────────────────────────────────
embeddings  = OpenAIEmbeddings()
vectorstore = FAISS.load_local(
    "doc_faiss_index",
    embeddings,
    allow_dangerous_deserialization=True,
)

# ───────────────────────────────────────────────────────
# Memory – nur die Antwort speichern
# ───────────────────────────────────────────────────────
class MemoryStripExtras(ConversationBufferMemory):
    def save_context(self, inputs, outputs):
        outputs = {"answer": outputs.get("answer", next(iter(outputs.values())))}
        super().save_context(inputs, outputs)

if "memory" not in st.session_state:
    st.session_state.memory = MemoryStripExtras(
        memory_key="chat_history",
        input_key="question",
        return_messages=True,
    )

# ───────────────────────────────────────────────────────
# LLM + Prompts
# ───────────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

answer_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(SYSTEM_MSG),
        HumanMessagePromptTemplate.from_template("{context}\n\nFrage: {question}"),
    ]
)

condense_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(SYSTEM_MSG),
        HumanMessagePromptTemplate.from_template(
            "Bisheriger Verlauf:\n{chat_history}\n\n"
            "Neue Rückfrage: {question}\n\n"
            "Formuliere eine eigenständige Frage:"
        ),
    ]
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=st.session_state.memory,
    return_source_documents=True,
    condense_question_prompt=condense_prompt,
    combine_docs_chain_kwargs={"prompt": answer_prompt},
)

# ───────────────────────────────────────────────────────
# Slug-Helfer →  Überschrift → anchor-id (HTML)
# ───────────────────────────────────────────────────────
def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^\w\s-]", "", text.lower())
    text = re.sub(r"\s+", "-", text).strip("-")
    return text

# ───────────────────────────────────────────────────────
# Haupt-UI
# ───────────────────────────────────────────────────────
st.title("🧠 Pflegedienst Software Assistent")

if "chat_history_ui" not in st.session_state:
    st.session_state.chat_history_ui = []

user_input = st.chat_input("Stelle eine Frage zur Software…")

if user_input:
    with st.spinner("Antwort wird generiert…"):
        result      = qa_chain({"question": user_input})
        answer_text = result["answer"]
        source_docs = result["source_documents"]

        st.session_state.chat_history_ui.append(
            {
                "user":    user_input,
                "bot":     answer_text,
                "sources": source_docs,
            }
        )

# ───────────────────────────────────────────────────────
# Chat-Verlauf anzeigen
# ───────────────────────────────────────────────────────
for msg in st.session_state.chat_history_ui:
    with st.chat_message("user"):
        st.markdown(msg["user"])

    with st.chat_message("assistant"):
        st.markdown(msg["bot"])
        if msg["sources"]:
            with st.expander("🔎 Quellen anzeigen", expanded=False):
                for doc in msg["sources"]:
                    title   = doc.metadata.get("title", "Unbekannte Seite")
                    section = doc.metadata.get("section", "")
                    base    = doc.metadata.get("url", "#")
                    anchor  = slugify(section) if section else ""
                    url     = f"{base}#{anchor}" if anchor else base
                    label   = f"{title} – {section}" if section else title
                    st.markdown(f"• [{label}]({url})")

# ───────────────────────────────────────────────────────
# Reset-Button
# ───────────────────────────────────────────────────────
if st.button("🗑️ Chatverlauf löschen"):
    st.session_state.chat_history_ui.clear()
    st.session_state.memory = MemoryStripExtras(
        memory_key="chat_history",
        input_key="question",
        return_messages=True,
    )
    st.rerun()
