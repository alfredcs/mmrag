# Version: v0.05: adding RAG solution without MM
#
import streamlit as st
from PIL import Image
import typing
import os
import sys
import io
from operator import itemgetter
import google.generativeai as genai
#from vertexai.preview.generative_models import (GenerativeModel, Part)
from openai import OpenAI
import base64
import requests
from io import BytesIO
import urllib.request
from brain import get_index_for_pdf
import hmac
#from streamlit_mic_recorder import mic_recorder, speech_to_text
#from audiorecorder import audiorecorder
from audio_recorder_streamlit import audio_recorder as audiorecorder
from multiprocessing.pool import ThreadPool
import concurrent.futures


module_paths = ["./", "./configs"]
for module_path in module_paths:
    sys.path.append(os.path.abspath(module_path))
from claude_bedrock_134 import *
from rad_tools_13 import *
from mmrag_tools_133 import *
from perplexity_tools_134 import *


from utils.gemini_generative_models import _GenerativeModel as GenerativeModel
from utils.gemini_generative_models import Part 

pool = ThreadPool(processes=8)
chroma_pers_dir = "/home/alfred/data/chroma_03122024_02"
df_pers_dir = "/home/alfred/df_03122024/df_images_02.csv"
st.set_page_config(page_title="MM-RAG Demo",page_icon="🩺",layout="wide")
st.title("Multimodal RAG Demo")

video_file_name = "download_video.mp4"
client = OpenAI(api_key=os.getenv('openai_api_token'))
voice_prompt = ""
chat_history = []
temp_audio_file = 'temp_input_audio.mp3'
sys.path.append('/home/alfred/anaconda3/envs/dui/bin/ffmpeg')

## VectorDB
profile_name = 'default'
os.environ["AWS_DEFAULT_REGION"] = "us-west-2"
my_region = os.environ.get("AWS_DEFAULT_REGION", None)
collection_name = 'bedrock-workshop-rag'
collection_id = '967j1chec55256z804lj'
aoss_host = "{}.{}.aoss.amazonaws.com:443".format(collection_id, my_region)
#credentials = boto3.Session(profile_name='default').get_credentials()
#auth = AWSV4SignerAuth(credentials, my_region, 'aoss')

## Chat memory
memory = ConversationBufferMemory(  
    return_messages=True, output_key="answer", input_key="question"  
)

## Empty a search cache file
cache_file = "/home/alfred/data/search_cache.csv"
with open(cache_file, "w") as file:
    # Write an empty string to clear the file contents
    file.write("")
    
## Image generation prompt rewrite
prefix = "Your role as an expert prompt engineer involves accuratly and meticulously rewriting the input text without altering original meaning, transforming it into a precised, detailed and enriched text prompt. This refined prompt is destined for a text-to-image generation model. Your primary objective is to strickly and precisely maintain the key elements and core semantic essence of the original text while infusing it with rich, descriptive elements. Such detailed guidance is crucial for steering the image generation model towards producing images of superior quality, characterized by their vivid and expressive visual nature. Your adeptness in prompt crafting is instrumental in ensuring that the final images not only captivate visually but also resonate deeply with the original textual concept. Please rewrite this prompt: "

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the passward is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False


#@st.cache_data
@st.cache_resource
def create_vectordb(files, filenames):
    # Show a spinner while creating the vectordb
    with st.spinner("Vector database"):
        vectordb = get_index_for_pdf(
            [file.getvalue() for file in files], filenames, openai_api_key
        )
    return vectordb


def get_asr(audio_filename):
    # Set the API endpoint
    url = 'http://infs.cavatar.info:8081/asr?task=transcribe&encode=true&output=txt'
    # Define headers
    headers = {
        'Accept': 'application/json',
        #'Content-Type': 'multipart/form-data'
    }

    # Define the file to be uploaded
    files = {
        'audio_file': (audio_filename, open(audio_filename, 'rb'), 'audio/mpeg')
    }

    # Make the POST request
    response = requests.post(url, headers=headers, files=files)
    output = response.text.rstrip()
    if output == "Thank you." or output == "Bye.":
        return ""
    else:
        return output

# ++++++++++++++ Local deployed models with TGI>=1.4 ++++++++++++++++++++++
    
    

# Check password
if not check_password():
    st.stop()  # Do not continue if check_password is not True.

with st.sidebar:
    st.title(':orange[Multimodal Config] :pencil2:')
    option = st.selectbox('Choose Model',('anthropic.claude-3-haiku-20240307-v1:0', 'anthropic.claude-3-sonnet-20240229-v1:0', 'anthropic.claude-v2:1', 'mistral.mistral-large-2402-v1:0', 'gemini-pro', 'gemini-pro-vision', 'gpt-4-1106-preview', 'gpt-4-vision-preview', 'stability.stable-diffusion-xl-v1:0', 'amazon.titan-image-generator-v1', 'dall-e-3'))

    if 'model' not in st.session_state or st.session_state.model != option:
        st.session_state.chat = genai.GenerativeModel(option).start_chat(history=[])
        st.session_state.model = option

    if 'stability.stable-diffusion' in option or 'amazon.titan-image-generator' in option or 'dall-e-3' in option:
        st.write("------- Image Generation----------")
        image_n = st.number_input("Choose number of images", min_value=1, value=1, max_value=1)
        image_size = st.selectbox('Choose image size (wxh)', ('512x512', '1024x1024', '768x1024'))
        image_quality = st.selectbox('Choose image quality', ('standard', 'hd'))
        cfg =  st.number_input("Choose CFG Scale for freedom", min_value=1.0, value=7.5, max_value=15.0)
        seed = st.slider('Choose seed for noise pattern', -1, 214783647, 452345)
    else:
        st.write("------- Default parameters ----------")
        temperature = st.number_input("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
        max_token = st.number_input("Maximum Output Token", min_value=0, value=1024, step=64)
        top_p = st.number_input("Top_p: The cumulative probability cutoff for token selection", min_value=0.1, value=0.85)
        top_k = st.number_input("Top_k: Sample from the k most likely next tokens at each step", min_value=1, value=40)
        #candidate_count = st.number_input("Number of generated responses to return", min_value=1, value=1)
        stop_sequences = st.text_input("The set of character sequences (up to 5) that will stop output generation", value="\n\n\n")
        gen_config = genai.types.GenerationConfig(max_output_tokens=max_token,temperature=temperature, top_p=top_p, top_k=top_k) #, candidate_count=candidate_count, stop_sequences=stop_sequences)
        #text_embedding_option = st.selectbox('Choose Embedding Model',('titan', 'tian-image', 'openai', 'hf-tei'))

    # --- Perplexity query -----#
    st.divider()
    st.header(':green[Perplexity] :confused:')
    #perplexity_on = st.toggle('Activate Perplexity query')
    perplexity_on = st.select_slider(
        'Activate Perplexity query',
        value='Naive',
        options=['Naive', 'Text', 'Multimodal(TBA)'])
    if 'Text' in perplexity_on or 'Multimodal' in perplexity_on:
        text_embedding_option = st.selectbox('Choose Embedding Model',('titan', 'tian-image', 'openai', 'hf-tei'))
    if 'Multimodal' in perplexity_on:
        upload_image = st.file_uploader("Upload your image here", accept_multiple_files=False, type=['jpg', 'png'])
        image_url_p = st.text_input("Or Input Image URL", key="image_url", type="default")
        if upload_image:
            bytes_data = upload_image.read()
            image =  (io.BytesIO(bytes_data))
            st.image(image)
        elif image_url_p:
            try:
                stream = fetch_image_from_url(image_url)
                st.image(stream)
                image = Image.open(stream)
            except:
                msg = 'Failed to download image, please check permission.'
                st.session_state.messages.append({"role": "assistant", "content": msg})
                st.chat_message("ai").write(msg)

    #----- RAG ----#
    st.divider()
    st.header(':green[Multimodal RAG] :file_folder:')
    rag_on = st.toggle('Activate RAG (Only tested with Claude 3)')
    if rag_on and 'anthropic.claude-3' in option:
        text_embedding_option = st.selectbox('Choose Embedding Model',('titan', 'tian-image', 'openai', 'hf-tei'))
        upload_docs = st.file_uploader("Upload pdf files", accept_multiple_files=True, type=['pdf'])
        doc_urls = st.text_input("Or input URLs seperated by ','", key="doc_urls", type="default")
        if text_embedding_option == 'titan':
            embd_model_id = "amazon.titan-embed-image-v1"
            text_embedding = BedrockEmbeddings(client=boto3_bedrock, model_id=embd_model_id)
            chunk_size = 8000
        elif text_embedding_option == 'openai':
            text_embedding =  OpenAIEmbeddings(openai_api_key=os.getenv('openai_api_token'))
            chunk_size = 8000
        elif text_embedding_option == 'hf-tei':
            text_embedding = HuggingFaceHubEmbeddings(model='http://infs.cavatar.info:8084')
            chunk_size = 500
        upload_doc_names = []
        if upload_docs:
            upload_doc_names = [file.name for file in upload_docs]
            for upload_doc in upload_docs:
                bytes_data = upload_doc.read()
                with open(upload_doc.name, 'wb') as f:
                    f.write(bytes_data)
            #st.session_state["vectordb"] = create_vectordb(upload_docs, upload_doc_names)
            #docs, avg_doc_length = data_prep(upload_doc_names, text_embedding, chunk_size=chunk_size)
            #print(f'Docs:{upload_doc_names} and avg sizes:{avg_doc_length}')
            #vlen = update_vdb(docs, text_embedding, aoss_host, collection_name, profile_name, my_region)
            async_result = pool.apply_async(insert_into_chroma, ([], upload_doc_names, chroma_pers_dir, embd_model_id, chunk_size, int(chunk_size*0.05), option, max_token, temperature, top_k, top_p, df_pers_dir))
            vlen = async_result.get()
            msg = f'Total {vlen} papges of document was added to vectorDB.'
            st.session_state.messages.append({"role": "assistant", "content": msg})
            #st.chat_message("ai").write(msg)
        if doc_urls:
            try:
                #docs, avg_doc_length = data_prep(doc_urls.split(","), text_embedding, chunk_size=chunk_size)
                async_result = pool.apply_async(insert_into_chroma, (doc_urls.split(","), [], chroma_pers_dir, embd_model_id, chunk_size, int(chunk_size*0.05), option, max_token, temperature, top_k, top_p, df_pers_dir))
                vlen = async_result.get()
                #vlen = update_vdb(docs, text_embedding, aoss_host, collection_name, profile_name, my_region)
                msg = f'Total {vlen} papges of document was added to vectorDB.'
            except:
                msg = f'Incorrect url format.'
                pass
            st.session_state.messages.append({"role": "assistant", "content": msg})
            #st.chat_message("ai").write(msg)

    
     #----- Image ----#
    st.divider()
    st.header(':green[Image Understanding] :camera:')
    image_on = st.select_slider(
        'Activate image uploading',
        value='None',
        options=['None', 'Single', 'Multiple'])
    if 'Single' in image_on or  'Multiple' in image_on:
        upload_images = st.file_uploader("Upload your Images Here", accept_multiple_files=True, type=['jpg', 'png', 'pdf'])
        image_url = st.text_input("Or Input Image URL", key="image_url", type="default")
        if upload_images:
            #image = Image.open(upload_image)
            for upload_file in upload_images:
                bytes_data = upload_file.read()
                image =  (io.BytesIO(bytes_data))
                st.image(image)
                #image_path = upload_file
                #base64_image = convert_image_to_base64(upload_file)
                #base64_image = base64.b64encode(upload_file.read())
        elif image_url:
            try:
                stream = fetch_image_from_url(image_url)
                st.image(stream)
                image = Image.open(stream)
            except:
                msg = 'Failed to download image, please check permission.'
                st.session_state.messages.append({"role": "assistant", "content": msg})
                st.chat_message("ai").write(msg)
    if 'Multiple' in image_on:
        st.divider()
        st.caption('Image comparisons')
        upload_image2 = st.file_uploader("Upload the 2nd Images for comparison", accept_multiple_files=False, type=['jpg', 'jpeg', 'png'])
        if upload_image2:
            image2 = Image.open(upload_image2)
            #bytes_data2 = upload_file2.read()
            #image22 = Image.open(io.BytesIO(bytes_data))
            st.image(image2)

    #---- Video -----#
    st.divider()
    st.header(':green[Video Understanding] :video_camera:')
    video_on = st.toggle('Activate video uploading')
    if video_on:
        upload_video = st.file_uploader("Upload your video Here", accept_multiple_files=False, type=['mp4'])
        video_url = st.text_input("Or Input Video URL with mp4 type", key="video_url", type="default")
        if video_url:
            urllib.request.urlretrieve(video_url, video_file_name)
            video = Part.from_uri(
                uri=video_url,
                mime_type="video/mp4",
            )
            video_file = open(video_file_name, 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)
        elif upload_video:
            video_bytes = upload_video.getvalue()
            with open(video_file_name, 'wb') as f:
                f.write(video_bytes)
            video = Part.from_uri(
                uri=video_file_name,
                mime_type="video/mp4",
            )
            st.video(video_bytes)
    
    # --- Audio query -----#
    st.divider()
    st.header(':green[Enable voice input]')# :microphone:')
    voice_on = st.toggle('Activate microphone')
    if voice_on:
        #record_audio=audiorecorder(start_prompt="Voice input start:  ▶️ ", stop_prompt="Record stop: ⏹️", pause_prompt="", key=None)
        record_audio_bytes = audiorecorder(icon_name="fa-solid fa-microphone-slash", recording_color="#cc0000", neutral_color="#666666",icon_size="2x",)
        #if len(record_audio)>3:
        if record_audio_bytes:
            #record_audio_bytes = record_audio.export().read()
            st.audio(record_audio_bytes, format="audio/wav")#, start_time=0, *, sample_rate=None)
            #record_audio.export(temp_audio_file, format="mp3")
            with open(temp_audio_file, 'wb') as audio_file:
                audio_file.write(record_audio_bytes)
            if os.path.exists(temp_audio_file):
                voice_prompt = get_asr(temp_audio_file)
                #os.remove(temp_audio_file)
            #record_audio.empty()
            #if os.path.exists(temp_audio_file):
            #    os.remove(temp_audio_file)
        st.caption("Press space and hit ↩️ for voice & agent activation")
        
     # --- Clear history -----#
    st.divider()

    if st.button("Clear Chat History"):
        st.session_state.messages.clear()
        st.session_state["messages"] = [{"role": "assistant", "content": "I am your assistant. How can I help today?"}]
        record_audio = None
        voice_prompt = ""
        #del st.session_state[record_audio]

# =============================================

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "I am your assistant. How can I help today?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

#if upload_images or image_url:
if "Single" in image_on or "Multiple" in image_on:
    if option != "gemini-pro-vision" and option != "gpt-4-vision-preview" and option != 'llava-v1.5-13b-vision' and option != 'stability.stable-diffusion-xl-v1:0' and option != 'amazon.titan-image-generator-v1' and "anthropic.claude-3" not in option:
        st.info("Please switch to a vision model")
        st.stop()
    if prompt := st.chat_input(placeholder=voice_prompt, on_submit=None, key="user_input"):
        prompt=voice_prompt if prompt==' ' else prompt
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        #print(f'In image on {option}')
        try:
            if option == "gemini-pro-vision":
                if isinstance(image, io.BytesIO):
                    image = Image.open(image)
                context = [prompt,image]
                #context = [prompt, image] if upload_images or image_url else ([prompt, video] if video_url else None)
                if "Multiple" in image_on:
                    if isinstance(image2, io.BytesIO):
                        image2 = Image.open(image2)
                    context = [prompt,image, image2]
                response=st.session_state.chat.send_message(context,stream=True,generation_config = gen_config)
                response.resolve()
                msg=response.text
            elif "anthropic.claude-3" in option.lower():
                if "Multiple" in image_on:
                    msg = bedrock_get_img_description2(option, prompt, image, image2, max_token, temperature, top_p, top_k, stop_sequences)
                else:
                    msg = bedrock_get_img_description(option, prompt, image, max_token, temperature, top_p, top_k, stop_sequences)
            elif option == 'llava-v1.5-13b-vision' and image_url:
                image_prompt = f'({image_url}) {prompt}'
                msg=tgi_imageGen('http://infs.cavatar.info:8085/generate', image_prompt, max_token, temperature, top_p, top_k)
            elif option == "gpt-4-vision-preview":
                msg = getDescription(option, prompt, image, max_token, temperature, top_p)
                if "Multiple" in image_on:
                     msg = getDescription2(option, prompt, image, image2, max_token, temperature, top_p)
            elif option == "amazon.titan-image-generator-v1" or option =='stability.stable-diffusion-xl-v1:0':
                if record_audio and len(voice_prompt) > 1:
                    new_prompt = tgi_textGen('http://infs.cavatar.info:8080', f'{prefix} {prompt}', max_token, temperature, top_p, top_k)
                else:
                    new_prompt = prompt
                src_image = image if 'image' in locals() else None
                image_quality = 'premium' if image_quality == 'hd' else image_quality
                try:
                    height = int(image_size.split('x')[1])
                    width = int(image_size.split('x')[0])
                    base64_str = bedrock_imageGen(option, new_prompt, iheight=height, iwidth=width, src_image=image, image_quality=image_quality, image_n=image_n, cfg=cfg, seed=seed)
                    new_image = Image.open(io.BytesIO(base64.decodebytes(bytes(base64_str, "utf-8"))))
                    #st.image(new_image,use_column_width='auto')
                    st.image(new_image,use_column_width='auto', width=256)
                    msg = new_prompt 
                except:
                    msg = 'Server error encountered. Please try again later!.'
                    pass
            else:
                msg = "Please choose a correct model."
        except:
            msg = "Server error encountered. Please try again later."
            pass
        msg += "\n\n✒︎Content created by using: " + option
        st.session_state.chat = genai.GenerativeModel(option).start_chat(history=[])
        st.session_state.messages.append({"role": "assistant", "content": msg})
        
        st.image(image)
        if "Multiple" in image_on and upload_image2:
            st.image(image2)
        st.chat_message("assistant", avatar='🌈').write(msg)
elif video_on:
    if option != "gemini-pro-vision" and option != "gpt-4-vision-preview" and "anthropic.claude-3" not in option:
        st.info("Please switch to a vision model")
        st.stop()
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        try:
            if option == "gemini-pro-vision":
                multimodal_model = GenerativeModel(option)
                context = [prompt, video]
                responses = multimodal_model.generate_content(context, stream=True)
                #response=st.session_state.chat.send_message(context,stream=True,generation_config = gen_config)
                #response.resolve()
                for response in responses:
                    msg += response.text
            elif 'anthropic.claude-3' in option.lower():
                msg = videoCaptioning_claude(option, prompt, getBase64Frames(video_file_name), max_token, temperature, top_p)
            elif option == "gpt-4-vision-preview":
                msg = videoCaptioning(option, prompt, getBase64Frames(video_file_name), max_token, temperature, top_p)
            elif option == 'llava-v1.5-13b-vision' and  image_url:
                image_prompt = f'({image_url}){prompt}'
                msg=tgi_imageGen('http://infs.cavatar.info:8085/generate', image_prompt, max_token, temperature, top_p, top_k)
        except:
            msg = "Server error encountered. Please try again."
            pass
        msg += "\n\n✒︎Content created by using: " + option
        st.session_state.chat = genai.GenerativeModel(option).start_chat(history=[])
        st.session_state.messages.append({"role": "assistant", "content": msg})

        #video_file = open(video_file_name, 'rb')
        #video_bytes = video_file.read()
        st.video(video_bytes, start_time=0)
        st.chat_message("assistant", avatar='🎞️').write(msg)
elif rag_on:
    if "anthropic.claude-3" not in option:
        st.info("Please switch to claude-3 model")
        st.stop()
    if prompt := st.chat_input(placeholder=voice_prompt, on_submit=None, key="user_input"):
        prompt=voice_prompt if prompt==' ' else prompt
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            try:
                if text_embedding_option == 'titan':
                    embd_model_id = "amazon.titan-embed-image-v1"
                    text_embedding = BedrockEmbeddings(client=boto3_bedrock, model_id=embd_model_id)
                elif text_embedding_option == 'openai':
                    text_embedding =  OpenAIEmbeddings(openai_api_key=os.getenv('openai_api_token'))
                elif text_embedding_option == 'hf-tei':
                     ext_embedding = HuggingFaceHubEmbeddings(model='http://infs.cavatar.info:8084')
    
                #print(f'RAG:{prompt}, model to use:{option}')
                #msg = do_query(prompt, option, text_embedding, aoss_host, collection_name, profile_name, max_token, temperature, top_p, top_k, my_region)
                if voice_prompt:
                    answer1 = executor.submit(retrieval_from_chroma_decompose, chroma_pers_dir, embd_model_id, prompt.replace("'s ", " "), option, max_token, temperature, top_k, top_p)
                    #answer2 = executor.submit(top_2_images, prompt, df_pers_dir, embd_model_id="amazon.titan-embed-g1-text-02")
                    #msg = retrieval_from_chroma_decompose(chroma_pers_dir, text_embedding, prompt, option, max_token, temperature, top_k, top_p)
                    msg = answer1.result()
                    #images = answer2.result()
                else:
                    #msg = retrieval_from_chroma_fusion(chroma_pers_dir, text_embedding, prompt, option, max_token, temperature, top_k, top_p)
                    answer1 = executor.submit(retrieval_from_chroma_fusion, chroma_pers_dir, embd_model_id, prompt.replace("'s ", " "), option, max_token, temperature, top_k, top_p)
                    #answer2 = executor.submit(top_2_images, prompt, df_pers_dir, embd_model_id="amazon.titan-embed-g1-text-02")
                    #msg = retrieval_from_chroma_decompose(chroma_pers_dir, text_embedding, prompt, option, max_token, temperature, top_k, top_p)
                    msg = answer1.result()
                    #images = answer2.result()
                #msg,_ = do_faiss_query(option, prompt, temperature=temperature, max_tokens=max_token, top_p=top_p)
            except:
                msg = "Server error encountered. Please try again."
                pass
            msg += "\n\n✒︎Content created by using: RAG with " + option
            st.session_state.chat = genai.GenerativeModel(option).start_chat(history=[])
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.chat_message("ai", avatar="📄").write(msg)
            images = top_2_images(prompt, df_pers_dir, embd_model_id=embd_model_id)
            if len(images) > 0:
                for img_path in images:
                    image_data = Image.open(img_path[0])
                    st.image(image_data, width=500, caption=f"Source: {img_path[1]}")

elif 'naive' not in perplexity_on.lower():
    if prompt := st.chat_input(placeholder=voice_prompt, on_submit=None, key="user_input"):
        prompt=voice_prompt if prompt==' ' else prompt
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        #try:
        if text_embedding_option == 'titan':
            embd_model_id = "amazon.titan-embed-g1-text-02"
            text_embedding = BedrockEmbeddings(client=boto3_bedrock, model_id=embd_model_id)
        elif  text_embedding_option == 'titan-image':
            embd_model_id = "amazon.titan-embed-image-v1"
        elif text_embedding_option == 'openai':
            text_embedding =  OpenAIEmbeddings(openai_api_key=os.getenv('openai_api_token'))
        elif text_embedding_option == 'hf-tei':
                     ext_embedding = HuggingFaceHubEmbeddings(model='http://infs.cavatar.info:8084')
            
        if "anthropic.claude" in option.lower() and 'text' in perplexity_on.lower():
            msg,_ = bedrock_textGen_perplexity_memory(option, prompt, max_token, temperature, top_p, top_k, stop_sequences, embd_model_id)
        elif 'multimodal' in perplexity_on.lower():
            if image:
                if 'anthropic.claude' in option:
                    prompt_msg = bedrock_get_img_description(option, prompt, image, max_token, temperature, top_p, top_k, stop_sequences)
                elif "gemini-pro-vision" in option:
                    if isinstance(image, io.BytesIO):
                        image = Image.open(image)
                    context = [prompt,image]
                    response=st.session_state.chat.send_message(context,stream=True,generation_config = gen_config)
                    response.resolve()
                    prompt_msg=response.text
                elif 'gpt-4-vision' in option:
                    prompt_msg = getDescription(option, prompt, image, max_token, temperature, top_p)
                    
                keywords = extract_keywords(prompt_msg)
                search_prompt = '+'.join(str(item) for item in keywords)
                prompt = f"{prompt}::{search_prompt}"
                #prompt = f"{prompt}::{prompt_msg}"
            msg,_ = bedrock_imageGen_perplexity(option, prompt, max_token, temperature, top_p, top_k, stop_sequences, embd_model_id)
            #images= 
        elif "mistral.mistral-large" in option.lower() and 'text' in perplexity_on.lower():
            msg,_ = bedrock_textGen_perplexity_memory(option, prompt, max_token, temperature, top_p, top_k, stop_sequences, embd_model_id)
        else:
            msg = "Please choose a correct model."
        #except:
        #    msg = "Server error encountered. Please try again later."
        #    pass
        msg += "\n\n✒︎Content created by using: Perplexity query with " + option
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("ai", avatar='🤔').write(msg)
        #images = top_2_images(prompt, df_pers_dir, embd_model_id=embd_model_id)
        #if 'multimodal' in perplexity_on.lower():
        #    images = ['./images/under_construct.jpg, https://t3.ftcdn.net/jpg/03/53/83/92/360_F.jpg']
        #    if len(images) > 0:
        #        for img_path in images:
        #            image_data = Image.open(img_path.split(',')[0])
        #            left_co, cent_co,last_co = st.columns(3)
        #            with cent_co:
        #                st.image(image_data, width=400, caption=f"Source: {img_path.split(',')[1]}")
                   

elif voice_on:
#elif (record_audio_bytes and len(voice_prompt) > 1):
    if prompt := st.chat_input(placeholder=voice_prompt, on_submit=None, key="user_input"):
        prompt=voice_prompt if prompt==' ' else prompt
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        #try:
        if "anthropic.claude" in option.lower() :
            msg=bedrock_textGen(option, prompt, max_token, temperature, top_p, top_k, stop_sequences)
        elif option == "gemini-pro":
            response=st.session_state.chat.send_message(prompt,stream=True,generation_config = gen_config)
            response.resolve()
            msg=response.text
        elif option == "gpt-4-1106-preview":
            msg=textGen_agent(option, prompt, max_token, temperature, top_p)
        elif option == "mistral-7b":
            msg=tgi_textGen('http://infs.cavatar.info:8080', prompt, max_token, temperature, top_p, top_k)
        elif option == 'llava-v1.5-13b-vision' and image_url:
            msg=tgi_textGen('http://infs.cavatar.info:8085', prompt, max_token, temperature, top_p, top_k)
        elif option == "amazon.titan-image-generator-v1" or option == "stability.stable-diffusion-xl-v1:0":
            src_image = image if 'image' in locals() else None
            image_quality = 'premium' if image_quality == 'hd' else image_quality
            new_prompt =tgi_textGen('http://infs.cavatar.info:8080', f'{prefix} {prompt}', max_token, temperature, top_p, top_k)
            width = int(image_size.split('x')[0])
            height = int(image_size.split('x')[1])
            base64_str = bedrock_imageGen(option, new_prompt, iheight=height, iwidth=width, src_image=src_image, image_quality=image_quality, image_n=image_n, cfg=cfg, seed=seed)
            new_image = Image.open(io.BytesIO(base64.decodebytes(bytes(base64_str, "utf-8"))))
            st.image(new_image,use_column_width='auto')
            msg = new_prompt
        else:
            msg = "Please choose a correct model."
        #except:
        #    msg = "Server error encountered. Please try again later."
        #    pass
        msg += "\n\n✒︎Content created by using: LangChain Agent and " + option
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("ai", avatar='🎙️').write(msg)

else:
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        #try:
        if option == "gemini-pro":
            response=st.session_state.chat.send_message(prompt,stream=True,generation_config = gen_config)
            response.resolve()
            msg=response.text
        elif option == "gpt-4-1106-preview":
            msg=textGen(option, prompt, max_token, temperature, top_p)
        elif "anthropic.claude" in option.lower():
            msg=bedrock_textGen(option, prompt, max_token, temperature, top_p, top_k, stop_sequences)
        elif "mistral.mistral-large" in option.lower():
            msg=bedrock_textGen_mistral(option, prompt, max_token, temperature, top_p, top_k)
        elif option == "mistral-7b":
            print(f'IN {option} with {prompt}')
            msg=tgi_textGen('http://infs.cavatar.info:8080', prompt, max_token, temperature, top_p, top_k)
        elif option == 'llava-v1.5-13b-vision':
            msg=tgi_textGen('http://infs.cavatar.info:8085', prompt, max_token, temperature, top_p, top_k)
        elif option == "amazon.titan-image-generator-v1" or option == "stability.stable-diffusion-xl-v1:0":
            src_image = image if 'image' in locals() else None
            image_quality = 'premium' if image_quality == 'hd' else image_quality
            height = int(image_size.split('x')[1])
            width = int(image_size.split('x')[0])
            base64_str = bedrock_imageGen(option, prompt, iheight=height, iwidth=width, src_image=src_image, image_quality=image_quality, image_n=image_n, cfg=cfg, seed=seed)
            new_image = Image.open(io.BytesIO(base64.decodebytes(bytes(base64_str, "utf-8"))))
            st.image(new_image,use_column_width='auto')
            msg = ' '
        elif option == "dall-e-3":
            image_url = dalle3_imageGen(option, prompt, size=image_size, quality=image_quality, n_image=image_n)
            st.image(image_url)
            msg = ''
        else:
            msg = "Please choose a correct model."
        #except Exception as err:
        #    msg = "Server error encountered. Please try again later."
        #    pass
        msg += "\n\n ✒︎Content created by using: " + option
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("ai", avatar='🦙').write(msg)
