import os, re, string                          # string added                     ### NEW
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import (              # prompt helpers imported        ### NEW
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# ──────────────────────────────────────────────────────────
# PAGE CONFIG  …
# ──────────────────────────────────────────────────────────
st.set_page_config(page_title="Pflegedienst Chatbot mit Gedächtnis", page_icon="🧠")

# ( … unchanged sidebar key validation … )

# ──────────────────────────────────────────────────────────
# System message you want the LLM to obey                  ### NEW
# ──────────────────────────────────────────────────────────
SYSTEM_MSG = (
    "Du bist ein hilfsbereiter Assistent für die Pflegedienst-Software. "
    "Antworte immer auf Deutsch, kurz und präzise, mit Aufzählungen wenn sinnvoll."
)

# ──────────────────────────────────────────────────────────
# Embeddings / Vectorstore  … (unchanged)
# ──────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────
# Memory class  … (unchanged)
# ──────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────
# Build LLM + custom prompts                               ### NEW
# ──────────────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

# 1) Prompt that the LLM will use to **generate the final answer**
answer_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(SYSTEM_MSG),       # system
        HumanMessagePromptTemplate.from_template(                    # human
            "{context}\n\nFrage: {question}"
        ),
    ]
)

# 2) Prompt that rewrites follow-up questions to standalone form
condense_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(SYSTEM_MSG),       # same system
        HumanMessagePromptTemplate.from_template(
            "Bisheriger Verlauf:\n{chat_history}\n\n"
            "Neue Rückfrage: {question}\n\n"
            "Formuliere eine eigenständige Frage:"
        ),
    ]
)

retriever = FAISS.load_local("doc_faiss_index", embeddings, allow_dangerous_deserialization=True
).as_retriever(search_kwargs={"k": 3})

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=st.session_state.memory,
    return_source_documents=True,
    condense_question_prompt=condense_prompt,          # <-- inject condense prompt ### NEW
    combine_docs_chain_kwargs={"prompt": answer_prompt} # <-- inject answer prompt   ### NEW
)

# ──────────────────────────────────────────────────────────
# Rest of your UI logic stays exactly the same
# ──────────────────────────────────────────────────────────
