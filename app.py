import os, re, string
import streamlit as st

from langchain_community.vectorstores import FAISS   # new import path
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.memory import ConversationBufferMemory

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# ───────────────────────────────────────────────────────
# FIRST Streamlit command → page layout
# ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pflegedienst Chatbot mit Gedächtnis",
    page_icon="🧠",
)

# ───────────────────────────────────────────────────────
# SIDEBAR: OpenAI-Key input + simple validation
# ───────────────────────────────────────────────────────
st.sidebar.title("🔑 OpenAI-API-Key")

api_key_input = st.sidebar.text_input(
    "Bitte gültigen OpenAI-Key eingeben",
    type="password",
    placeholder="sk-...",
)

def valid_openai_key(key: str) -> bool:
    key = key.strip()
    return (
        key.startswith("sk-")
        and len(key) >= 24                     # minimum length
        and all(32 <= ord(c) < 127 for c in key)  # printable ASCII
    )

if api_key_input and valid_openai_key(api_key_input):
    os.environ["OPENAI_API_KEY"] = api_key_input.strip()
elif not os.getenv("OPENAI_API_KEY"):
    st.sidebar.error("Ungültiger Key – bitte korrekt einfügen.")
    st.stop()

# ───────────────────────────────────────────────────────
# System message that guides every response
# ───────────────────────────────────────────────────────
SYSTEM_MSG = (
    "Du bist ein hilfsbereiter Assistent für die Pflegedienst-Software. Dir werden Dokumente aus der Dokumentation bereitgestellt."
    "Antworte immer auf Deutsch, präzise und mit Aufzählungen wenn sinnvoll."
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
# Memory that stores only the LLM answer (drops extras)
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
# Build LLM + custom prompts
# ───────────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

answer_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(SYSTEM_MSG),
        HumanMessagePromptTemplate.from_template(
            "{context}\n\nFrage: {question}"
        ),
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
    condense_question_prompt=condense_prompt,            # system msg applied
    combine_docs_chain_kwargs={"prompt": answer_prompt}, # system msg applied
)

# ───────────────────────────────────────────────────────
# MAIN UI
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

        # keep list of (doc, score) although we don't score here
        doc_tuples = [(doc, 0) for doc in source_docs]

        # console logging
        print("\n=== Retrieved Documents ===")
        for i, (doc, _) in enumerate(doc_tuples, 1):
            print(f"\nDocument {i}\n{doc.page_content}\n" + "-" * 60)

        st.session_state.chat_history_ui.append(
            {"user": user_input, "bot": answer_text, "sources": doc_tuples}
        )

# ───────────────────────────────────────────────────────
# RENDER CHAT HISTORY
# ───────────────────────────────────────────────────────
for msg in st.session_state.chat_history_ui:
    with st.chat_message("user"):
        st.markdown(msg["user"])

    with st.chat_message("assistant"):
        st.markdown(msg["bot"])
        if msg["sources"]:
            with st.expander("🔎 Quellen anzeigen", expanded=False):
                for doc, _ in msg["sources"]:
                    title   = doc.metadata.get("title", "Unbekannte Seite")
                    section = doc.metadata.get("section", "")
                    url     = doc.metadata.get("url", "#")
                    label   = f"{title} – {section}" if section else title
                    st.markdown(f"• [{label}]({url})")

# ───────────────────────────────────────────────────────
# RESET BUTTON
# ───────────────────────────────────────────────────────
if st.button("🗑️ Chatverlauf löschen"):
    st.session_state.chat_history_ui.clear()
    st.session_state.memory = MemoryStripExtras(
        memory_key="chat_history",
        input_key="question",
        return_messages=True,
    )
    st.rerun()
