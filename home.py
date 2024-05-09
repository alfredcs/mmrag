import streamlit as st
import sys
import os
import io
from PIL import Image
from io import BytesIO
import base64

module_paths = ["./", "./configs"]
file_path = "./data/"

for module_path in module_paths:
    sys.path.append(os.path.abspath(module_path))

from utility import *
from utils import *

st.set_page_config(page_title="Advanced RAG",page_icon="🩺",layout="wide")
st.title("Advanced RAG Demo")

aoss_host = read_key_value(".aoss_config.txt", "AOSS_host_name")
aoss_index = read_key_value(".aoss_config.txt", "AOSS_index_name")


#@st.cache_data
#@st.cache_resource(TTL=300)
with st.sidebar:
    #----- RAG  ------ 
    st.header(':green[Search RAG] :file_folder:')
    rag_search = rag_update = rag_retrieval = False
    rag_on = st.select_slider(
        'Activate RAG',
        value='None',
        options=['None', 'Search', 'Insert', 'Retrieval'])
    if 'Search' in rag_on:
        doc_num = st.slider('Choose max number of documents', 1, 8, 3)
        embedding_model_id = st.selectbox('Choose Embedding Model',('amazon.titan-embed-g1-text-02', 'amazon.titan-embed-image-v1'))
        rag_search = True
    elif 'Insert' in rag_on:
        upload_docs = st.file_uploader("Upload your doc here", accept_multiple_files=True, type=['pdf', 'doc', 'jpg', 'png'])
        # Amazon Bedrock KB only supports titan-embed-text-v1 not g1-text-02
        embedding_model_id = st.selectbox('Choose Embedding Model',('amazon.amazon.titan-embed-text-v1', 'amazon.titan-embed-image-v1'))
        rag_update = True
    elif 'Retrieval' in rag_on:
        rag_retrieval = True
                
    #----- Choose models  ------ 
    st.divider()
    st.title(':orange[Model Config] :pencil2:') 
    if 'Search' in rag_on:
        option = st.selectbox('Choose Model',('anthropic.claude-3-haiku-20240307-v1:0', 
                                              'anthropic.claude-3-sonnet-20240229-v1:0'
                                             ))
    else:
        option = st.selectbox('Choose Model',('anthropic.claude-3-haiku-20240307-v1:0', 
                                              'anthropic.claude-3-sonnet-20240229-v1:0',
                                              #'anthropic.claude-3-opus-20240229-v1:0',
                                              'meta.llama3-70b-instruct-v1:0',
                                              'finetuned:llama-3-8b-instruct'))
        
    st.write("------- Default parameters ----------")
    temperature = st.number_input("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
    max_token = st.number_input("Maximum Output Token", min_value=0, value=1024, step=64)
    top_p = st.number_input("Top_p: The cumulative probability cutoff for token selection", min_value=0.1, value=0.85)
    top_k = st.number_input("Top_k: Sample from the k most likely next tokens at each step", min_value=1, value=40)
    #candidate_count = st.number_input("Number of generated responses to return", min_value=1, value=1)
    stop_sequences = st.text_input("The set of character sequences (up to 5) that will stop output generation", value="\n\nHuman")

        
    # ---- Clear chat history ----
    st.divider()
    if st.button("Clear Chat History"):
        st.session_state.messages.clear()
        st.session_state["messages"] = [{"role": "assistant", "content": "I am your assistant. How can I help today?"}]
        record_audio = None
        voice_prompt = ""
        #del st.session_state[record_audio]


###
# Streamlist Body
###
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "I am your assistant. How can I help today?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

#
if rag_search:
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        documents, urls = google_search(prompt, num_results=doc_num)
        msg = retrieval_faiss(prompt, documents, option, embedding_model_id, 6000, 600, max_token, temperature, top_p, top_k, doc_num)
        msg += "\n\n ✧ Sources:\n\n" + '\n\n\r'.join(urls)
        msg += "\n\n ✒︎ Content created by using: " + option
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("ai", avatar='🦙').write(msg)

elif rag_update:        
    # Update AOSS
    if upload_docs:
        empty_directory(file_path)
        upload_doc_names = [file.name for file in upload_docs]
        for upload_doc in upload_docs:
            bytes_data = upload_doc.read()
            with open(file_path+upload_doc.name, 'wb') as f:
                f.write(bytes_data)
        stats, status, kb_id = bedrock_kb_injection(file_path)
        msg = f'Total {stats}  to Amazon Bedrock Knowledge Base: {kb_id}  with status: {status}.'
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("ai", avatar='🎙️').write(msg)

elif rag_retrieval:
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        #Do retrieval
        msg = bedrock_kb_retrieval(prompt, option)
        #msg = bedrock_kb_retrieval_advanced(prompt, option, max_token, temperature, top_p, top_k, stop_sequences)
        #msg = bedrock_kb_retrieval_decomposition(prompt, option, max_token, temperature, top_p, top_k, stop_sequences)
        msg += "\n\n ✒︎Content created by using: " + option
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("ai", avatar='🦙').write(msg)
        
else:
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        if 'generate image' in classify_query(prompt, 'generate image, news, others', 'anthropic.claude-3-haiku-20240307-v1:0'):
            option = 'amazon.titan-image-generator-v1' #'stability.stable-diffusion-xl-v1:0' # Or 'amazon.titan-image-generator-v1'
            base64_str = bedrock_imageGen(option, prompt, iheight=1024, iwidth=1024, src_image=None, image_quality='premium', image_n=1, cfg=7.5, seed=452345)
            new_image = Image.open(io.BytesIO(base64.decodebytes(bytes(base64_str, "utf-8"))))
            st.image(new_image,width=512)#use_column_width='auto')
            msg = ' '
        else:
            msg=bedrock_textGen(option, prompt, max_token, temperature, top_p, top_k, stop_sequences)
        msg += "\n\n ✒︎Content created by using: " + option
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("ai", avatar='🦙').write(msg)
        
