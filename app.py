import os
import streamlit as st
from llama_index.llms.groq import Groq
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from twilio.rest import Client
import tempfile

st.title("PDF Q&A to WhatsApp Sender")

groq_key = st.text_input("Enter GROQ API Key", type="password")
check = st.checkbox("Send answer to WhatsApp")
if check:
    tw_sid =st.text_input("Twilio SID",type="password")
    tw_token =st.text_input("Twilio Token",type="password")
    number=st.text_input("Enter your WhatsApp number",placeholder="+91XXXXXXXXXX")
uploaded_file =st.file_uploader("Upload PDF",type="pdf")
question =st.text_input("Ask your question")
if st.button("Submit"):
    if uploaded_file and question and groq_key and (not check or (tw_sid and tw_token and number)):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            file_path =tmp.name
        os.environ['GROQ_API_KEY'] = groq_key
        llm = Groq(model="llama-3.1-8b-instant")
        embed_model =HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
    )
        documents = SimpleDirectoryReader(
            input_files=[file_path]
        ).load_data()
        index = VectorStoreIndex.from_documents(
            documents,
            llm=llm,
            embed_model=embed_model
        )
        query_engine =index.as_query_engine(llm=llm)
        response = query_engine.query(question)
        st.success("Answer Generated")
        # st.write(response)
        st.write("### Answer:")
        st.write(str(response))
        st.write("#### Source Page Numbers:")
        for node in response.source_nodes:
            st.write(f"Page: {node.node.metadata.get('page_label')}")
        if check:
            client = Client(tw_sid, tw_token)
            try:
                message = client.messages.create(
                    to=f'whatsapp:{number}',
                    from_='whatsapp:+14155238886',
                    # body=str(response)
                    body=response.response
                )
                st.success("Message sent to WhatsApp successfully!")
            except Exception as e:
                st.error(f"Error sending message: {e}")
    else:
        st.warning("Please fill all fields!")