import os
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.memory import ConversationBufferMemory

# ────────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  ➜  must be the first Streamlit command
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Pflegedienst Chatbot mit Gedächtnis", page_icon="🧠")

# ────────────────────────────────────────────────────────────────────────────────
# SIDEBAR: OpenAI API key input
# ────────────────────────────────────────────────────────────────────────────────
st.sidebar.title("🔑 OpenAI-API-Key")
api_key = st.sidebar.text_input(
    "Trage hier deinen OpenAI-API-Key ein:",
    type="password",
    placeholder="sk-...",
    key="api_key_input",
)

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
elif "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"]:
    st.sidebar.error("Bitte zuerst einen gültigen Key eingeben, sonst funktioniert der Chat nicht.")
    st.stop()

# ────────────────────────────────────────────────────────────────────────────────
# DATA & HELPERS  (unchanged)
# ────────────────────────────────────────────────────────────────────────────────
embeddings  = OpenAIEmbeddings()
vectorstore = FAISS.load_local("doc_faiss_index", embeddings, allow_dangerous_deserialization=True)

def reranked_sources(question: str, top_k: int = 3):
    query_emb = embeddings.embed_query(question)
    results   = vectorstore.similarity_search_with_score_by_vector(query_emb, k=10)
    results.sort(key=lambda x: x[1])
    return results[:top_k]

class ConversationBufferMemoryIgnoreExtras(ConversationBufferMemory):
    def save_context(self, inputs, outputs):
        outputs = {"answer": outputs.get("answer", next(iter(outputs.values())))}
        super().save_context(inputs, outputs)

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemoryIgnoreExtras(
        memory_key="chat_history", input_key="question", return_messages=True
    )

llm       = ChatOpenAI(model="gpt-4o", temperature=0.3)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=st.session_state.memory,
    return_source_documents=True,
)

# ────────────────────────────────────────────────────────────────────────────────
# MAIN PAGE UI
# ────────────────────────────────────────────────────────────────────────────────
st.title("🧠 MeinPflegedienst Dokumentation Chatbot")

if "chat_history_ui" not in st.session_state:
    st.session_state.chat_history_ui = []

user_input = st.chat_input("Stelle eine Frage zur Software…")

if user_input:
    with st.spinner("Antwort wird generiert…"):
        # 1) model answer (with memory)
        result       = qa_chain({"question": user_input})
        answer_text  = result["answer"]
        source_docs  = result["source_documents"]

        # 2) score docs for UI
        scored_docs = reranked_sources(user_input, top_k=len(source_docs))
        score_map   = {d.page_content: s for d, s in scored_docs}
        doc_tuples  = [(doc, score_map.get(doc.page_content, 999)) for doc in source_docs]

        # 3) console log
        print("\n=== Retrieved Documents ===")
        for i, (doc, sc) in enumerate(doc_tuples, 1):
            print(f"\nDocument {i}  (Score: {sc})")
            print(doc.page_content)
            print("-" * 60)

        # 4) save for UI
        st.session_state.chat_history_ui.append({
            "user":    user_input,
            "bot":     answer_text,
            "sources": doc_tuples,
        })

# ────────────────────────────────────────────────────────────────────────────────
# RENDER CHAT HISTORY
# ────────────────────────────────────────────────────────────────────────────────
for msg in st.session_state.chat_history_ui:
    with st.chat_message("user"):
        st.markdown(msg["user"])

    with st.chat_message("assistant"):
        st.markdown(msg["bot"])
        if msg["sources"]:
            best_score = min(sc for _, sc in msg["sources"])
            with st.expander("🔎 Quellen anzeigen", expanded=False):
                for doc, sc in msg["sources"]:
                    title   = doc.metadata.get("title", "Unbekannte Seite")
                    section = doc.metadata.get("section", "")
                    url     = doc.metadata.get("url", "#")
                    label   = f"{title} – {section}" if section else title
                    if sc == best_score:
                        st.markdown(f"• [**{label}**]({url})")   # bold best
                    else:
                        st.markdown(f"• [{label}]({url})")

# ────────────────────────────────────────────────────────────────────────────────
# RESET BUTTON
# ────────────────────────────────────────────────────────────────────────────────
if st.button("🗑️ Chatverlauf löschen"):
    st.session_state.chat_history_ui = []
    st.session_state.memory = ConversationBufferMemoryIgnoreExtras(
        memory_key="chat_history", input_key="question", return_messages=True
    )
    st.rerun()
