import streamlit as st
import warnings

warnings.filterwarnings("ignore")

from manifesto_qa.app import (
    retriever,
    llm,
    prompt_template,
    # prompt_template_with_history,
    # run_rag_chain,
    run_rag_chain_with_sources,
    # run_rag_chain_with_history,
)

from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory,
)

st.title("General Election 2024 Party Manifesto Q&A")

# Initialize chat history (not working)
# msgs = StreamlitChatMessageHistory(key="langchain_messages")
# for msg in msgs.messages:
#     st.chat_message(msg.type).write(msg.content)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("What's up?"):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # completion = run_rag_chain(retriever, llm, prompt_template, prompt)
        completion = run_rag_chain_with_sources(retriever, llm, prompt_template, prompt)
        response = st.write(completion)

    st.session_state.messages.append({"role": "assistant", "content": completion})
