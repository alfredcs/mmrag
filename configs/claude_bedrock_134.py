import os
import sys
import boto3
import logging
import io
import json
import base64
import struct
from io import BytesIO
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
#from langchain.embeddings import BedrockEmbeddings
from langchain_community.embeddings import BedrockEmbeddings
#from langchain_community.chat_models.bedrock import BedrockChat
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from PIL import Image
from botocore.exceptions import ClientError
# Langchian agent
from langchain.tools import DuckDuckGoSearchRun
from langchain.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.chat_models import BedrockChat
from langchain.agents import initialize_agent, AgentType, load_tools
#from langchain import FewShotPromptTemplate
# Keywords
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

## Rewrite
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
#from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)

module_paths = ["./", "../", "./configs"]
for module_path in module_paths:
    sys.path.append(os.path.abspath(module_path))
    
from utils import bedrock
from mmrag_tools_133 import  resize_base64_image,  resize_bytes_image

boto3_bedrock = bedrock.get_bedrock_client(
    assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
    region=os.environ.get("AWS_DEFAULT_REGION", None)
)
logger = logging.getLogger(__name__)

os.environ["AWS_DEFAULT_REGION"] = region = "us-west-2"

def get_bedrock_client(region):
    bedrock_client = boto3.client("bedrock-runtime", region_name=region)
    return bedrock_client

def image_to_base64(img) -> str:
    """Converts a PIL Image or local image file path to a base64 string"""
    if isinstance(img, str):
        if os.path.isfile(img):
            print(f"Reading image from file: {img}")
            with open(img, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        else:
            raise FileNotFoundError(f"File {img} does not exist")
    elif isinstance(img, Image.Image):
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    else:
        raise ValueError(f"Expected str (filename) or PIL Image. Got {type(img)}")

def bedrock_imageGen(model_id:str, prompt:str, iheight:int, iwidth:int, src_image, image_quality:str, image_n:int, cfg:float, seed:int):
    negative_prompts = [
                "poorly rendered",
                "poor background details",
                "poorly drawn objects",
                "poorly focused objects",
                "disfigured object features",
                "cartoon",
                "animation"
            ]
    titan_negative_prompts = ','.join(negative_prompts)
    try:
        if model_id == "amazon.titan-image-generator-v1":
            if cfg > 10.0:
               cfg = 10.0
            if src_image:
                src_img_b64 = image_to_base64(src_image)
                body = json.dumps(
                    {
                        "taskType": "IMAGE_VARIATION",
                        "imageVariationParams": {
                            "text":prompt,   # Required
                            "negativeText": titan_negative_prompts,  # Optional
                            "images": [src_img_b64]
                        },
                        "imageGenerationConfig": {
                            "numberOfImages": image_n,   # Range: 1 to 5 
                            "quality": image_quality,  # Options: standard or premium
                            "height": iheight,         # Supported height list in the docs 
                            "width": iwidth,         # Supported width list in the docs
                            "cfgScale": cfg,       # Range: 1.0 (exclusive) to 10.0
                            "seed": seed             # Range: 0 to 214783647
                        }
                    }
                )
            else:
                body = json.dumps(
                    {
                        "taskType": "TEXT_IMAGE",
                        "textToImageParams": {
                            "text":prompt,   # Required
                            "negativeText": titan_negative_prompts  # Optional
                        },
                        "imageGenerationConfig": {
                            "numberOfImages": image_n,   # Range: 1 to 5 
                            "quality": image_quality,  # Options: standard or premium
                            "height": iheight,         # Supported height list in the docs 
                            "width": iwidth,         # Supported width list in the docs
                            "cfgScale": cfg,       # Range: 1.0 (exclusive) to 10.0
                            "seed": seed             # Range: 0 to 214783647
                        }
                    }
                )
        elif model_id == "stability.stable-diffusion-xl-v1:0":
            style_preset = "photographic"  # (e.g. photographic, digital-art, cinematic, ...)
            clip_guidance_preset = "FAST_GREEN" # (e.g. FAST_BLUE FAST_GREEN NONE SIMPLE SLOW SLOWER SLOWEST)
            sampler = "K_DPMPP_2S_ANCESTRAL" # (e.g. DDIM, DDPM, K_DPMPP_SDE, K_DPMPP_2M, K_DPMPP_2S_ANCESTRAL, K_DPM_2, K_DPM_2_ANCESTRAL, K_EULER, K_EULER_ANCESTRAL, K_HEUN, K_LMS)
            if src_image:
                src_img_b64 = image_to_base64(src_image)
                body = json.dumps({
                    "text_prompts": (
                            [{"text": prompt, "weight": 1.0}]
                            + [{"text": negprompt, "weight": -1.0} for negprompt in negative_prompts]
                        ),
                    "cfg_scale": cfg,
                    "seed": seed,
                    "steps": 60,
                    "style_preset": style_preset,
                    "clip_guidance_preset": clip_guidance_preset,
                    "sampler": sampler,
                    "width": iwidth,
                    "init_image": src_img_b64,
                })
            else:
                body = json.dumps({
                    "text_prompts": (
                            [{"text": prompt, "weight": 1.0}]
                            + [{"text": negprompt, "weight": -1.0} for negprompt in negative_prompts]
                        ),
                    "cfg_scale": cfg,
                    "seed": seed,
                    "steps": 60,
                    "style_preset": style_preset,
                    "clip_guidance_preset": clip_guidance_preset,
                    "sampler": sampler,
                    "width": iwidth,
                })
            
        response = get_bedrock_client(region).invoke_model(
            body=body, 
            modelId=model_id,
            accept="application/json", 
            contentType="application/json"
        )
        response_body = json.loads(response["body"].read())
        if model_id == "amazon.titan-image-generator-v1":
            base64_image_data = response_body["images"][0]
        elif model_id == "stability.stable-diffusion-xl-v1:0":
            base64_image_data = response_body["artifacts"][0].get("base64")

        return base64_image_data

    except ClientError:
        logger.error("Couldn't invoke Titan Image Generator Model")
        raise
            
def bedrock_textGen(model_id, prompt, max_tokens, temperature, top_p, top_k, stop_sequences):
    stop_sequence = [stop_sequences]
    if  "anthropic.claude-v2" in model_id.lower():
        inference_modifier = {
            "max_tokens_to_sample": max_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "stop_sequences": stop_sequence,
        }
    
        textgen_llm = Bedrock(
            model_id=model_id,
            client=boto3_bedrock,
            model_kwargs=inference_modifier,
        )     
        return textgen_llm(prompt)
    elif "anthropic.claude-3" in model_id.lower():
        payload = {
            "modelId": model_id,
            "contentType": "application/json",
            "accept": "application/json",
            "body": {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                #"stop_sequences": stop_sequence,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            }
                        ]
                    }
                ]
            }
        }
        
        # Convert the payload to bytes
        body_bytes = json.dumps(payload['body']).encode('utf-8')
        # Invoke the model
        response = get_bedrock_client(region).invoke_model(
            body=body_bytes,
            contentType=payload['contentType'],
            accept=payload['accept'],
            modelId=payload['modelId']
        )
        
        # Process the response
        response_body = response['body'].read().decode('utf-8')
        data = json.loads(response_body)
        return data['content'][0]['text']
    
    else:
        return f"Incorrect Bedrock model ID {model_id.lower()} selected!"

def bedrock_textGen_mistral(model_id, prompt, max_tokens, temperature, top_p, top_k):
    inference_modifier = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
    }

    textgen_llm = Bedrock(
        model_id=model_id,
        client=boto3_bedrock,
        model_kwargs=inference_modifier,
    )    
    return textgen_llm(prompt)
    
#------ Chat with Claude 3 ----
'''
class BedrockChatAnthropicMessagesAPI(BedrockChat):
    def _prepare_message_input(
        self, messages: List[BaseMessage], model_kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        formatted_messages = [
            {"role": "user" if isinstance(msg, HumanMessage) else "assistant", "content": msg.content}
            for msg in messages
        ]
        input_body = {
            "messages": formatted_messages,
            "anthropic_version": "bedrock-2023-05-31",
            **model_kwargs,
        }
        return input_body

    def _parse_message_output(self, response: Any) -> str:
        response_body = json.loads(response.get("body").read())
        return response_body.get("content", "")

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = ["Human"],
        **kwargs: Any,
    ) -> ChatResult:
        _model_kwargs = self.model_kwargs or {}
        params = {**_model_kwargs, **kwargs}

        input_body = self._prepare_message_input(messages, params)
        body = json.dumps(input_body)

        request_options = {
            "body": body,
            "modelId": self.model_id,
            "accept": "application/json",
            "contentType": "application/json",
        }
        try:
            response = self.client.invoke_model(**request_options)
            print(response)
            completion = self._parse_message_output(response)
        except Exception as e:
            raise ValueError(f"Error raised by bedrock service: {e}")
    
            message = AIMessage(content=completion)
            return ChatResult(generations=[ChatGeneration(message=message)])

def BedrockChatClaude3(model_id, prompt, max_tokens, temperature, top_p, top_k, stop_sequences):
    chat = BedrockChatAnthropicMessagesAPI(model_id=model_id, model_kwargs={"temperature": temperature, "max_tokens":max_tokens, 
                                                                            "top_p":top_p, "top_k": top_k, "stop_sequences": stop_sequences})
    messages = [HumanMessage(content=prompt),
                AIMessage(content=""),
                ]
    return chat.invoke(messages).content[0]["text"]
'''

def convert_image_to_base64(BytesIO_image):
    # Convert the image to RGB (optional, depends on your requirements)
    rgb_image = BytesIO_image.convert('RGB')
    # Prepare the buffer
    buffered = BytesIO()
    # Save the image to the buffer
    rgb_image.save(buffered, format="JPEG")
    # Get the byte data
    img_data = buffered.getvalue()
    # Encode to base64
    base64_encoded = base64.b64encode(img_data)
    return base64_encoded.decode()

def get_image_type(bytesio):
    """
    Detects the type of an image file based on its magic number.

    Args:
        file_path (str): The path to the image file.

    Returns:
        str: The image type in the format "image/type" or "Unknown" if the type cannot be determined.
    """
    # Define the magic numbers for different image types
    magic_numbers = {
        b'\xff\xd8\xff': 'image/jpeg',
        b'\x89PNG\r\n\x1a\n': 'image/png',
        b'GIF87a': 'image/gif',
        b'GIF89a': 'image/gif',
        b'BM': 'image/bmp',
        b'RIFF': 'image/webp'
    }

    try:
        bytesio.seek(0)
        header = bytesio.read(16)

        # Check if the header matches any of the known magic numbers
        for magic_number, image_type in magic_numbers.items():
            if header.startswith(magic_number):
                return image_type
    except Exception as e:
        print(f"Error: {e}")

    return "image/Unknown"
    
def bedrock_get_img_description(option, prompt, image, max_token, temperature, top_p, top_k, stop_sequences):
    stop_sequence = [stop_sequences]
    #encoded_string = base64.b64encode(image)
   
    # format conversation
    if isinstance(image, io.BytesIO):
        image = Image.open(image)
    #base64_string = encoded_string.decode('utf-8')

    # Resize to image resolution by half to make sure input to Claude 3 is < 5M
    width, height = image.size
    if width > 2048 or height > 2024:
        new_size = (int(width/2), int(height/2))
        image = image.resize(new_size, Image.Resampling.LANCZOS) # or Image.ANTIALIAS for Pillow < 7.0.0

    payload = {
        "modelId": option,
        "contentType": "application/json",
        "accept": "application/json",
        "body": {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_token,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            #"stop_sequences": stop_sequence,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": 'image/png', #get_image_type(image),
                                "data": convert_image_to_base64(image)
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        }
    }
    
    # Convert the payload to bytes
    body_bytes = json.dumps(payload['body']).encode('utf-8')
    
    # Invoke the model
    response = get_bedrock_client(region).invoke_model(
        body=body_bytes,
        contentType=payload['contentType'],
        accept=payload['accept'],
        modelId=payload['modelId']
    )
    
    # Process the response
    response_body = response['body'].read().decode('utf-8')
    data = json.loads(response_body)
    return data['content'][0]['text']

def bedrock_get_img_description2(option, prompt, image, image2, max_token, temperature, top_p, top_k, stop_sequences):
    stop_sequence = [stop_sequences]
    #encoded_string = base64.b64encode(image)
    #base64_string = encoded_string.decode('utf-8')
    
    if isinstance(image, io.BytesIO):
        image = Image.open(image)
    if isinstance(image2, io.BytesIO):
        image2 = Image.open(image2)
    
    payload = {
        "modelId": option,
        "contentType": "application/json",
        "accept": "application/json",
        "body": {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_token,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            #"stop_sequences": stop_sequence,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": 'image/png', #get_image_type(image),
                                "data": convert_image_to_base64(image)
                            }
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": 'image/png', #get_image_type(image),
                                "data": convert_image_to_base64(image2)
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        }
    }
    
    # Convert the payload to bytes
    body_bytes = json.dumps(payload['body']).encode('utf-8')
    
    # Invoke the model
    response = get_bedrock_client(region).invoke_model(
        body=body_bytes,
        contentType=payload['contentType'],
        accept=payload['accept'],
        modelId=payload['modelId']
    )
    
    # Process the response
    response_body = response['body'].read().decode('utf-8')
    data = json.loads(response_body)
    return data['content'][0]['text']
    
def bedrock_textGen_agent(model_id, prompt, max_tokens, temperature, top_p, top_k, stop_sequences):
    stop_sequence = [stop_sequences]
    inference_modifier = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        #"stop_sequences": stop_sequence,
    }

    #textgen_llm = Bedrock(
    textgen_llm = BedrockChat(
        model_id=model_id,
        client=boto3_bedrock,
        model_kwargs=inference_modifier,
    )

    ## Using Dickduckgo as search engine
    #wrapper = DuckDuckGoSearchAPIWrapper(region="wt-wt", safesearch='Moderate', time=None, max_results=3)
    #duckduckgo_search = DuckDuckGoSearchRun()
    #duckduckgo_tool = DuckDuckGoSearchResults()
    #Use serp search
    serp_tools = load_tools(["serpapi"], serpapi_api_key=os.getenv('serp_api_token'))

    # initialize the agent
    agent_chain = initialize_agent(
        #[duckduckgo_tool],
        serp_tools,
        textgen_llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, #STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
    )

    # run the agent
    output = agent_chain.run(
        prompt,
    )
    
    return output

def create_vector_db_chroma_index(bedrock_clinet, chroma_db_path: str, pdf_file_names: str, bedrock_embedding_model_id:str):
    #replace the document path here for pdf ingestion
    loader = PyPDFLoader(pdf_file_name)
    doc = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=4000, chunk_overlap=200, separator="\n")
    chunks = text_splitter.split_documents(doc)
    emb_model = "sentence-transformers/all-MiniLM-L6-v2"
    '''
    embeddings = HuggingFaceEmbeddings(
        model_name=emb_model,
        cache_folder="./cache/"
    )
    '''
    embeddings = create_langchain_vector_embedding_using_bedrock(bedrock_client=bedrock_client, bedrock_embedding_model_id=bedrock_embedding_model_id)
    db = Chroma.from_documents(chunks,
                               embedding=embeddings,
                               persist_directory=chroma_db_path)
    db.persist()
    return db

def create_langchain_vector_embedding_using_bedrock(bedrock_client, bedrock_embedding_model_id):
    bedrock_embeddings_client = BedrockEmbeddings(
        client=bedrock_client,
        model_id=bedrock_embedding_model_id)
    return bedrock_embeddings_client


def create_opensearch_vector_search_client(index_name, opensearch_password, bedrock_embeddings_client, opensearch_endpoint, _is_aoss=False):
    docsearch = OpenSearchVectorSearch(
        index_name=index_name,
        embedding_function=bedrock_embeddings_client,
        opensearch_url=f"https://{opensearch_endpoint}",
        http_auth=(index_name, opensearch_password),
        is_aoss=_is_aoss
    )
    return docsearch

def create_bedrock_llm(bedrock_client, model_version_id, temperature):
    bedrock_llm = Bedrock(
        model_id=model_version_id,
        client=bedrock_client,
        model_kwargs={'temperature': temperature}
        )
    return bedrock_llm

def bedrock_chroma_rag(llm_model_id, embed_model_id, temperature, max_token):
    bedrock_client = get_bedrock_client(region)
    bedrock_embedding_model_id = embed_model_id
    chroma_db= create_vector_db_chroma_index(bedrock_clinet, os.path.join("./","chroma_rag_db", pdf_file_names, bedrock_embedding_model_id))
    retriever = chroma_db.as_retriever()
    llm = Bedrock(model_id=llm_model_id, client=bedrock_client, model_kwargs={"max_tokens_to_sample": max_toekn, "temperature": temperature})

    template = """\n\nHuman:Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    {context}
    Question: {question}
    \n\nAssistant:"""
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=False)
    conv_qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=False)
    returnval = conv_qa_chain("is application development covered?")
    print(returnval["answer"])



def bedrock_llm(model_id, max_tokens, temperature):
    bedrock_embedding_model_id = 'amazon.titan-embed-text-v1'
    index_name = ''

    bedrock_client = get_bedrock_client(region)
    bedrock_llm = create_bedrock_llm(bedrock_client, model_id)
    bedrock_embeddings_client = create_langchain_vector_embedding_using_bedrock(bedrock_client, bedrock_embedding_model_id)
    opensearch_endpoint = opensearch.get_opensearch_endpoint(index_name, region)
    opensearch_password = secret.get_secret(index_name, region)
    opensearch_vector_search_client = create_opensearch_vector_search_client(index_name, opensearch_password, bedrock_embeddings_client, opensearch_endpoint)
    



    llm = Bedrock(model_id=model_id, client=boto3_bedrock, model_kwargs={'max_tokens_to_sample':max_tokens, 'temperature': temperature})
    bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=boto3_bedrock)
    prompt_template = """

    Human: Use the following pieces of context to provide a concise answer to the question at the end. Please think before answering and provide answers only when you find supported evidence. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    <context>
    {context}
    </context

    Question: {question}

    Assistant:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answewr['result']


def prompt_rewrite(distracted_query, temperature, top_p, max_len):
    bedrock_client = get_bedrock_client(region)
    inference_modifier = {
        #"max_tokens_to_sample": 512,
        "max_gen_len": max_len,
        "temperature": temperature,
        #"top_k": 0.85,
        "top_p": top_p,
    }
    
    bedrock_model = Bedrock(
        model_id="meta.llama2-70b-chat-v1",
        client=bedrock_client,
        model_kwargs=inference_modifier,
    )

    chat_llm = BedrockChat(
            model_id="meta.llama2-70b-chat-v1", client=boto3_bedrock, model_kwargs={"max_tokens_to_sample":max_len, "temperature":temperature, "top_p": top_p}
        )

    # create our examples
    examples = [
        {
            "input": "A cat sitting on a windowsill.",
            "output": "Imagine a cozy, sunlit room, with sheer curtains gently swaying in the breeze. On the wooden windowsill, there's a fluffy, ginger tabby cat lounging lazily. The cat's green eyes are half-closed, basking in the warm sunlight filtering through the window, casting a soft glow on its fur. Potted plants are placed around the windowsill, adding a touch of greenery to the serene scene."
        }, {
            "input": "A futuristic cityscape at night.",
            "output": "Envision a sprawling futuristic cityscape under the cloak of night, illuminated by the neon glow of skyscrapers. Hover cars zip through the skyways, leaving trails of light in their wake. The architecture is a blend of high-tech structures and eco-friendly green buildings with vertical gardens. In the sky, a giant hologram advertisement plays, reflecting off the glossy surface of a nearby tower, while the moon looms large in the starry sky."
        }, {
            "input": "A medieval knight on a quest.",
            "output": "Picture a valiant medieval knight, clad in shining armor, embarking on a noble quest through an ancient forest. The knight rides a majestic, well-armored steed. The dense forest is shrouded in mist, with rays of sunlight piercing through the canopy, creating a mystical ambiance. The knight holds aloft a banner with a crest symbolizing their noble cause, and in the background, an imposing, mysterious castle can be seen atop a distant hill, its flags fluttering in the wind."
        }
    ]
    
    # This is a prompt template used to format each individual example.
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )

    # now break our previous prompt into a prefix and suffix
    # the prefix is our instructions
    prefix = """Your role as an expert prompt engineer involves meticulously refining the input text, transforming it into a detailed and enriched prompt. This refined prompt is destined for a text-to-image generation model. Your primary objective is to maintain the core semantic essence of the original text while infusing it with rich, descriptive elements. Such detailed guidance is crucial for steering the image generation model towards producing images of superior quality, characterized by their vivid and expressive visual nature. Your adeptness in prompt crafting is instrumental in ensuring that the final images not only captivate visually but also resonate deeply with the original textual concept. Here are some examples: 
    """

    # now create the few shot prompt template
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        #prefix=prefix,
        #suffix=suffix,
        #input_variables=["query"],
        #example_separator="\n\n****"
    )

    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prefix),
            few_shot_prompt,
            ("human", "{input}"),
        ]
    )

    output_parser = StrOutputParser()
    chain = final_prompt | bedrock_model | output_parser

    return chain.invoke({"input": distracted_query})

# --- top n keywords --
def extract_keywords(text):
    # Tokenize the text into words
    words = word_tokenize(text)

    # Get the set of English stopwords
    stop_words = set(stopwords.words('english'))

    # Remove stopwords and any non-alphabetic tokens, like numbers and punctuation
    keywords = [word for word in words if word.isalpha() and word.lower() not in stop_words]

    return keywords
    

if __name__ == "__main__":
    response = prompt_rewrite("A man walks his dog toward the camera in a park.", 0.5, 0.85, 512)
    print(response)
