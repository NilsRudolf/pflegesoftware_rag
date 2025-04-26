import os, re
import streamlit as st
from langchain_community.vectorstores import FAISS          # ← NEW import path!
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.memory import ConversationBufferMemory

# ──────────────────────────────────────────────────────────
# FIRST Streamlit command → Page layout
# ──────────────────────────────────────────────────────────
st.set_page_config(page_title="Pflegedienst Chatbot mit Gedächtnis", page_icon="🧠")

# ──────────────────────────────────────────────────────────
# SIDEBAR: secure OpenAI key input & validation
# ──────────────────────────────────────────────────────────
st.sidebar.title("🔑 OpenAI-API-Key")

api_key_input = st.sidebar.text_input(
    "Bitte gültigen OpenAI-Key eingeben (beginnt mit 'sk-')",
    type="password",
    placeholder="sk-...",
)

# simple ASCII+pattern check (prevents the “UnicodeEncodeError”)
def valid_openai_key(key: str) -> bool:
    return bool(re.fullmatch(r"sk-[A-Za-z0-9]{20,}", key))

if api_key_input and valid_openai_key(api_key_input):
    os.environ["OPENAI_API_KEY"] = api_key_input
elif not os.getenv("OPENAI_API_KEY"):
    st.sidebar.error("Kein gültiger OpenAI-Key gesetzt! Bitte einfügen und ENTER drücken.")
    st.stop()

# ──────────────────────────────────────────────────────────
# Embeddings & VectorStore
# ──────────────────────────────────────────────────────────
embeddings  = OpenAIEmbeddings()
vectorstore = FAISS.load_local(
    "doc_faiss_index", embeddings, allow_dangerous_deserialization=True
)

# ──────────────────────────────────────────────────────────
# Memory that drops non-text outputs
# ──────────────────────────────────────────────────────────
class MemoryStripExtras(ConversationBufferMemory):
    def save_context(self, inputs, outputs):
        outputs = {"answer": outputs.get("answer", next(iter(outputs.values())))}
        super().save_context(inputs, outputs)

if "memory" not in st.session_state:
    st.session_state.memory = MemoryStripExtras(
        memory_key="chat_history", input_key="question", return_messages=True
    )

# ──────────────────────────────────────────────────────────
# LLM + ConversationalRetrievalChain
# ──────────────────────────────────────────────────────────
llm       = ChatOpenAI(model="gpt-4o", temperature=0.3)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=st.session_state.memory,
    return_source_documents=True,
)

# ──────────────────────────────────────────────────────────
# MAIN UI
# ──────────────────────────────────────────────────────────
st.title("🧠 Pflegedienst Software Assistent")

if "chat_history_ui" not in st.session_state:
    st.session_state.chat_history_ui = []

user_input = st.chat_input("Stelle eine Frage zur Software…")

if user_input:
    with st.spinner("Antwort wird generiert…"):
        result       = qa_chain({"question": user_input})   # memory-aware
        answer_text  = result["answer"]
        source_docs  = result["source_documents"]

        # just show docs; no manual re-ranking (memory intact)
        doc_tuples = [(doc, 0) for doc in source_docs]

        # console log
        print("\n=== Retrieved Documents ===")
        for i, (doc, _) in enumerate(doc_tuples, 1):
            print(f"\nDocument {i}\n{doc.page_content}\n" + "-" * 60)

        st.session_state.chat_history_ui.append(
            {"user": user_input, "bot": answer_text, "sources": doc_tuples}
        )

# ──────────────────────────────────────────────────────────
# RENDER CHAT HISTORY
# ──────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────
# RESET BUTTON
# ──────────────────────────────────────────────────────────
if st.button("🗑️ Chatverlauf löschen"):
    st.session_state.chat_history_ui.clear()
    st.session_state.memory = MemoryStripExtras(
        memory_key="chat_history", input_key="question", return_messages=True
    )
    st.rerun()
