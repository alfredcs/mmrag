import requests
from PIL import Image
from bs4 import BeautifulSoup as Soup
import os, sys, glob
import fitz
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders.csv_loader import CSVLoader
#import tabula
import pandas as pd
import matplotlib.pyplot as plt
import tiktoken
from operator import itemgetter
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import BedrockEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads
from langchain.document_loaders.json_loader import JSONLoader
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from langchain.vectorstores import OpenSearchVectorSearch
#from langchain import hub
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
#from io import BytesIO
from base64 import b64decode
from sklearn.metrics.pairwise import cosine_similarity
import threading
import concurrent.futures
import ast
import numpy as np 
from urllib.parse import unquote # Required to unquote URLs

#module_path = "../"
#sys.path.append(os.path.abspath(module_path))
#from claude_bedrock_13 import *
os.environ['AWS_PROFILE'] = 'default'
os.environ['AWS_DEFAULT_REGION'] = region = 'us-west-2'
module_paths = ["./", "./configs"]
for module_path in module_paths:
    sys.path.append(os.path.abspath(module_path))
from utils import bedrock
from claude_bedrock_134 import *
from multiprocessing.pool import ThreadPool

boto3_bedrock = bedrock.get_bedrock_client(
    #assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
    region=os.environ.get("AWS_DEFAULT_REGION", None)
)

#-- Embeddings ----
def get_text_embedding(image_base64=None, text_description=None,  embd_model_id:str="amazon.titan-embed-image-v1"):
    input_data = {}

    if image_base64 is not None:
        input_data["inputImage"] = image_base64
    if text_description is not None:
        input_data["inputText"] = text_description

    if not input_data:
        raise ValueError("At least one of image_base64 or text_description must be provided")

    body = json.dumps(input_data)

    response = boto3_bedrock.invoke_model(
        body=body,
        modelId=embd_model_id,
        accept="application/json",
        contentType="application/json"
    )

    response_body = json.loads(response.get("body").read())
    return response_body.get("embedding")

def resize_base64_image(base64_string, new_size):
    # Decode the base64 string
    image_data = b64decode(base64_string)
    # Open the image using Pillow
    image = Image.open(BytesIO(image_data))

    # Resize the image
    resized_image = image.resize(new_size)

    # Convert the resized image back to base64
    buffered = BytesIO()
    resized_image.save(buffered, format="PNG")
    resized_base64 = b64encode(buffered.getvalue()).decode('utf-8')
    print(f"Done with image resize: {resized_base64[:10]}")
    return resized_base64

def resize_bytes_image(image_bytes, target_width, target_height):
    # Load the image bytes into a PIL image
    image = Image.open(io.BytesIO(image_bytes))
    # Resize the image
    resized_image = image.resize((target_width, target_height))
    # Save the resized image back to bytes
    img_byte_arr = io.BytesIO()
    resized_image.save(img_byte_arr, format=image.format)  
    # Get the bytes of the resized image
    resized_image_bytes = img_byte_arr.getvalue()
    return resized_image_bytes
    
def titan_multimodal_embedding(
    image_path:str=None,  # maximum 2048 x 2048 pixels
    description:str=None, # English only and max input tokens 128
    dimension:int=1024,   # 1,024 (default), 384, 256
    embd_model_id:str="amazon.titan-embed-image-v1"
):
    payload_body = {}
    embedding_config = {
        "embeddingConfig": { 
             "outputEmbeddingLength": dimension
         }
    }

    # You can specify either text or image or both
    print(f"In image embedding {image_path}....")
    if image_path:
        with open(image_path, "rb") as image_file:
            input_image = base64.b64encode(image_file.read()).decode('utf8')
            '''
            print(f"here000: {type(input_image)}")
            img_data_tmp = base64.b64decode(input_image)
            print("here111")
            # Convert binary data to an image object
            img = Image.open(io.BytesIO(img_data_tmp))
            print(f"image {image_path} size: {img.size(0)}")
            if (img.size[0] > 2047 or img.size[1] > 2047):
                input_image = resize_base64_image(input_image, 1024)
            '''
            image_tt = Image.open(image_file)
            # Get image dimensions
            width, height = image_tt.size
            payload_body["inputImage"] = resize_base64_image(input_image, 2048) if width > 2048 else input_image

    if description:
        payload_body["inputText"] = description

    assert payload_body, "please provide either an image and/or a text description"
    print("\n".join(payload_body.values()))
    try:
        response = boto3_bedrock.invoke_model(
            body=json.dumps({**payload_body, **embedding_config}), 
            modelId= embd_model_id,
            accept="application/json", 
            contentType="application/json"
        )
    except  Exception as e:
            print(f'Error embedding image {image_path}: {e}')
    print("Done with embedding...")
    return json.loads(response.get("body").read())
    
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf8')

# ----------  Object store and AOSS  ---------#
aoss_host = f'rua94ts82ynqy19co875.{region}.aoss.amazonaws.com:443'
aoss_index_name = "bedrock-sample-rag-248"
service = 'aoss'
credentials = boto3.Session().get_credentials()
auth = AWSV4SignerAuth(credentials, os.environ.get("AWS_DEFAULT_REGION", region), service)

### Upload the image files to S3
s3_client = boto3.client("s3")

def uploadDirectory(path,bucket_name: str="mmrag-images"):
        for root,dirs,files in os.walk(path):
            for file in files:
                s3_client.upload_file(os.path.join(root,file),bucket_name,file)

def create_new_bucket(bucket_name:str, region_name:str):
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        return f"Bucket '{bucket_name}' already exists. Skipping creation."
    except s3_client.exceptions.ClientError as e:
        error_code = int(e.response['Error']['Code'])
        if error_code == 404:
            # Bucket doesn't exist, create it
            s3bucket = s3_client.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={ 'LocationConstraint': region_name }
            )
            return s3bucket

def insertImage2AOSS(image_path: str, image_url:str, bedrock_image_embeddings, host:str, new_index_name:str, auth):
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
        image_byteio = Image.open(io.BytesIO(image_data))
        # Resize if needed
        width, height = image_byteio.size
        print(f"Image size:{width}x{height}, types: {type(image_data)} and {type(image_byteio)}")
        if width > 2048 or height > 2048:
            image_data = resize_bytes_image(image_data, int(width/2), int(height/2))
            image_byteio = Image.open(io.BytesIO(image_data))
            
        image_base64 = base64.b64encode(image_data).decode('utf8')
        #image_vectors = get_image_embedding(image_base64=image_base64, text_description=image_path,  embd_model_id=embd_model_id)

    # Upload to S3
    last_slash_pos = image_path.rfind('/')
    # Extract the substring from the beginning to the last '/' character
    directory_path = image_path[:last_slash_pos]
    s3_client.upload_file(image_path,bucket_name,image_path)
    s3_image_path = f"s3://{bucket_name}/{image_path}"
    
    # Form a json
    document = {
        "doc_source": image_url,
        "image_filename": s3_image_path,
        "embedding": image_base64
    }
    
    filename = f"{os.path.dirname(image_path)}/{image_path.split('/')[-1].split('.')[0]}.json"
    
    # Writing JSON data
    with open(filename, 'w') as file:
        json.dump(document, file, indent=4)

    loader = DirectoryLoader(os.path.dirname(image_path), glob='**/*.json', show_progress=False, loader_cls=TextLoader)

    #loader = DirectoryLoader("./jsons", glob='**/*.json', show_progress=True, loader_cls=JSONLoader, loader_kwargs = {'jq_schema':'.content'})
    new_documents = loader.load()
    new_docs = text_splitter.split_documents(new_documents)

    # Insert into AOSS
    new_docsearch = OpenSearchVectorSearch.from_documents(
        new_docs,
        bedrock_image_embeddings,
        opensearch_url=host,
        http_auth=auth,
        timeout = 100,
        use_ssl = True,
        verify_certs = True,
        connection_class = RequestsHttpConnection,
        index_name=new_index_name,
        engine="faiss",
    )

    # ### Clear out the local temp files
    [os.remove(f) for f in glob.glob(f'{os.path.dirname(image_path)}/*.json')]

    return True

# --- parsing --- #
def parse_tables_images(url:str, model_id, max_token, temperature, top_k, top_p, df_pers_file, embd_model_id):
    # Send a GET request to the URL 
    response = requests.get(url)

    # Parse the HTML content using BeautifulSoup
    soup = Soup(response.content, 'html.parser')

    # Find all table elements
    tables = soup.find_all('table')
    # Find all image elements
    images = soup.find_all('img')
    dir_p = url.split('/')[-1] 
    if len(dir_p) < 1:
        dir_p = url.split('/')[-2] 
    # Create a directory to store the tables
    os.makedirs(f'./{dir_p}/tables', exist_ok=True)
    os.makedirs(f'./{dir_p}/images', exist_ok=True)
    os.makedirs(f'./{dir_p}/summaries', exist_ok=True)
    df_image_filenames = []
    df_image_sums = []
    df_image_sources = []
     # Save each table as an HTML file
    for i, table in enumerate(tables, start=1):
        table_html = str(table)
        
        #Creat table summary
        with open(f'./{dir_p}/summaries/summary_table_{i}.txt', 'w') as f:
            table_sum = bedrock_textGen(model_id=model_id, 
                                        prompt='You are a perfect table reader and pay great attention to detail which makes you an expert at generating a comprehensive table summary in text based on this input:'+table_html, 
                                        max_tokens=max_token, 
                                        temperature=temperature, 
                                        top_p=top_p, 
                                        top_k=top_k, 
                                        stop_sequences='Human:',
                                       )  
            f.write(table_sum)

        with open(f'./{dir_p}/tables/table_{i}.html', 'w', encoding='utf-8') as f:
            f.write(table_html)

    # Save each image to a file
    for i, image in enumerate(images, start=1):
        image_src = image.get('src')
        if image_src.startswith('http'):
            image_url = image_src
        else:
            base_url = '/'.join(url.split('/')[:3])
            image_url = f'{base_url}/{image_src}'

        try:
            image_data = requests.get(image_url).content
            image_byteio = Image.open(io.BytesIO(image_data))
            width, height = image_byteio.size
            #print(f"IImage size:{width}x{height}, types: {type(image_data)} and {type(image_byteio)}")
            if width < 128 or height < 128:
                continue
            elif width > 2048 or height > 208:
                image_data = resize_bytes_image(image_data, int(width/2), int(height/2))
                image_byteio = Image.open(io.BytesIO(image_data))
                width, height = image_byteio.size
                #print(f"IImage new size:{width}x{height}, types: {type(image_data)} and {type(image_byteio)}")
            image_sum = bedrock_get_img_description(model_id, 
                                    prompt='You are an expert at analyzing images in great detail. Your task is to carefully examine the provided \
                                                image and generate a detailed, accurate textual description capturing all of key and supporting elements as well as \
                                                context present in the image. Pay close attention to any numbers, data, or quantitative information visible, \
                                                and be sure to include those numerical values along with their semantic meaning in your description. \
                                                Thoroughly read and interpret the entire image before providing your detailed caption describing the \
                                                image content in text format. Strive for a truthful and precise representation of what is depicted',
                                    image=image_byteio, 
                                    max_token=max_token, 
                                    temperature=temperature, 
                                    top_p=top_p, 
                                    top_k=top_k, 
                                    stop_sequences='Human:')
            #print(f'{type(image_byteio)} and {image_sum}')
            if len(image_sum) > 1:
                with open(f'./{dir_p}/summaries/summary_image_{i}.txt', 'w') as f:
                    f.write(image_sum)
                    image_base64 = base64.b64encode(image_data).decode('utf8')
                    image_sum_vectors = get_text_embedding(image_base64=image_base64, text_description=image_sum,  embd_model_id=embd_model_id)
                    df_image_sums.append(image_sum_vectors)
            else:
                df_image_sums.append("")
            f_n = f'./{dir_p}/images/image_{i}.png'
            with open(f_n, 'wb') as f:
                f.write(image_data)
                df_image_filenames.append(f_n)
                df_image_sources.append(url)
                #img_embedding = get_text_embedding(image_base64=resized_image, text_description=image_sum)
                #img_embedding = titan_multimodal_embedding(image_path=f_n, description=image_sum)
                #df_image_vectors.append(img_embedding)
            #print(f"image_file:{f_n}, image_source:{url}, image_sum:{image_sum}")
        except Exception as e:
            print(f'Error saving image: {e}')
            pass
    loader = DirectoryLoader(f'./{dir_p}/summaries', glob="**/*.txt")
    docs_sums = loader.load()
    # Save image df
    #df_image = pd.DataFrame({'image': df_image_filenames, 'vector': df_image_vectors, 'summary': df_image_sums})
    df_image = pd.DataFrame({'image': df_image_filenames, 'source':df_image_sources, 'summary': df_image_sums})
    existing_df = pd.read_csv(df_pers_file) if os.path.isfile(df_pers_file) else pd.DataFrame()
    combined_df = pd.concat([existing_df, df_image], ignore_index=True)
    combined_df.drop_duplicates(subset=['image'], inplace=True)
    combined_df.to_csv(df_pers_file, index=False)
    return docs_sums

def parse_images_tables_from_pdf(pdf_path:str, output_folder:str, model_id, max_token, temperature, top_k, top_p, df_pers_file, embd_model_id):
    os.makedirs(output_folder, exist_ok=True)
    # Load text content
    loader = PyPDFLoader(pdf_path)
    text_splitter = CharacterTextSplitter(chunk_size=100000, chunk_overlap=1000)
    pdf_texts = loader.load_and_split(text_splitter)
    df_image_filenames = []
    df_image_sums = []
    df_image_sources = []
    # Open the PDF file
    pdf_file = fitz.open(pdf_path)

    # Iterate through each page
    for page_index in range(len(pdf_file)):        
        # Select the page
        page = pdf_file[page_index]

        # Search for tables on the page
        tables = page.find_tables()

        for table_index, table in enumerate(tables):
            df = table.to_pandas()
            rows, columns = df.shape
            if rows < 2 or columns < 2:
                continue
            table_path = f"{output_folder}/table_{page_index}_{table_index}.csv"
            df.to_csv(table_path)
            '''
            # Save the table as a CSV file
            print(f"Table {table_index + 1} on Page {page_index + 1}:")
            table_path = f"{output_folder}/table_{page_index}_{table_index}.csv"
            with open(table_path, "w", encoding="utf-8") as csv_file:
                print(f"Here2: {table}")
                for row in table.rows:
                    csv_file.write(",".join([str(cell) for cell in row]) + "\n")
            print(f"Table saved: {table_path}")
            '''
        loader = DirectoryLoader(f"{output_folder}", glob='**/*.csv', loader_cls=CSVLoader)
        table_csvs = loader.load()
        
        # Search for images on the page
        images = page.get_images()
        for image_index, img in enumerate(images):
            # Get the image bounding box
            xref = img[0]
            image_info = pdf_file.extract_image(xref)
            print(f"Image {output_folder}_{image_index} res ({image_info['width']}, {image_info['height']}")
            if image_info['width'] < 128 or image_info['height'] < 128:
                continue
            image_data = image_info["image"]
            image_ext = image_info["ext"]

            # Save the image
            image_path = f"{output_folder}/image_{page_index}_{image_index}.{image_ext}"
            with open(image_path, "wb") as image_file:
                image_file.write(image_data)
            
            #print(f"Image saved: {image_path}")
            # Get image caption
            image_byteio = Image.open(io.BytesIO(image_data))

            try:
                image_sum = bedrock_get_img_description(model_id, 
                                        prompt='You are an expert at analyzing images in great detail. Your task is to carefully examine the provided \
                                                image and generate a detailed, accurate textual description capturing all of the important elements and \
                                                context present in the image. Pay close attention to any numbers, data, or quantitative information visible, \
                                                and be sure to include those numerical values along with their semantic meaning in your description. \
                                                Thoroughly read and interpret the entire image before providing your detailed caption describing the \
                                                image content in text format. Strive for a truthful and precise representation of what is depicted',
                                        image=image_byteio, 
                                        max_token=max_token, 
                                        temperature=temperature, 
                                        top_p=top_p, 
                                        top_k=top_k, 
                                        stop_sequences='Human:')
                #print(f'{type(image_byteio)} and {image_sum}')
                df_image_filenames.append(image_path)
                df_image_sources.append(pdf_path)
                if len(image_sum) > 1:
                    with open(f"{output_folder}/image_{page_index}_{image_index}.txt", 'w') as f:
                        f.write(image_sum)
                        image_base64 = base64.b64encode(image_data).decode('utf8')
                        image_sum_vectors = get_text_embedding(image_base64=image_base64, text_description=image_sum,  embd_model_id=embd_model_id)
                        df_image_sums.append(image_sum_vectors)
                else:
                        df_image_sums.append("")
            except Exception as e:
                print(f"Fail to process {image_path} with error {e}")
                pass

    # Close the PDF file
    pdf_file.close()
    loader = DirectoryLoader(f'./{output_folder}', glob="**/*.txt", loader_cls=TextLoader)
    text_splitter = CharacterTextSplitter(chunk_size=100000, chunk_overlap=1000)
    image_chart_sums = loader.load_and_split(text_splitter)
    print(f"filename len: {len(df_image_filenames)}, img_src len: {len(df_image_sources)} and img_sum len: {len(df_image_sums)}")
    df_image = pd.DataFrame({'image': df_image_filenames, 'source':df_image_sources, 'summary': df_image_sums})
    existing_df = pd.read_csv(df_pers_file) if os.path.isfile(df_pers_file) else pd.DataFrame()
    combined_df = pd.concat([existing_df, df_image], ignore_index=True)
    combined_df.drop_duplicates(subset=['image'], inplace=True)
    combined_df.to_csv(df_pers_file, index=False)
    pdf_texts.extend([*table_csvs, *image_chart_sums])
    return pdf_texts

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def combine_lists(nested_lists):
    return [element for sublist in nested_lists for element in sublist]

def extract_from_urls_or_pdf(urls: list, pdfs:list,  model_id, max_token, temperature, top_k, top_p, df_pers_file, embd_model_id):
    all_docs = []
    if len(urls) > 0:
        for url in urls:
            loader = RecursiveUrlLoader(
                url=url, max_depth=20, extractor=lambda x: Soup(x, "html.parser").text
            )
            docs = loader.load()
            sums = parse_tables_images(url, model_id, max_token, temperature, top_k, top_p, df_pers_file, embd_model_id)
            all_docs.append([*docs, *sums])
    elif len(pdfs) > 0:
        for pdf in pdfs:
            output_dir, ext = os.path.splitext(os.path.basename(pdf))
            sums_pdf = parse_images_tables_from_pdf(pdf, output_dir,  model_id, max_token, temperature, top_k, top_p, df_pers_file, embd_model_id)
            all_docs.append([*sums_pdf])
    else:
        return all_docs
    new_docs = combine_lists(all_docs)
    #docs_texts = [d.page_content for d in new_docs]
    return new_docs
    
# ---- Parse urls from questions ----
def parse_urls_by_question(query:str):
    response = requests.get(f"https://www.google.com/search?q={query}") # Make the request
    soup = Soup(response.text, "html.parser") # Parse the HTML
    links = soup.find_all("a") # Find all the links in the HTML
    urls = []
    for l in [link for link in links if link["href"].startswith("/url?q=")]:
        # get the url
        url = l["href"]
        # remove the "/url?q=" part
        url = url.replace("/url?q=", "")
        # remove the part after the "&sa=..."
        url = unquote(url.split("&sa=")[0])
        # special case for google scholar
        if url.startswith("https://scholar.google.com/scholar_url?url=http"):
            url = url.replace("https://scholar.google.com/scholar_url?url=", "").split("&")[0]
        elif 'google.com/' in url: # skip google links
            continue
        elif 'youtube.com/' in url:
            continue
        elif 'search?q=' in url:
            continue
        if url.endswith('.pdf'): # skip pdf links
            continue
        if '#' in url: # remove anchors (e.g. wikipedia.com/bob#history and wikipedia.com/bob#genetics are the same page)
            url = url.split('#')[0]
        # print the url
        urls.append(url)
    
    # Use numpy to dedupe the list of urls after removing anchors
    return list(np.unique(urls))


# ------ ETL ------#

def insert_into_chroma(urls:list, pdfs:list, persist_directory, embd_model_id, chunk_size_tok:int, chunk_overlap:int, model_id, max_token, temperature, top_k, top_p, df_pers_file):
    docs = extract_from_urls_or_pdf(urls, pdfs, model_id, max_token, temperature, top_k, top_p, df_pers_file, embd_model_id)
    d_sorted = sorted(docs, key=lambda x: x.metadata["source"])
    d_reversed = list(reversed(d_sorted))
    concatenated_content = "\n\n\n --- \n\n\n".join(
        [doc.page_content for doc in d_reversed]
    )
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size_tok, chunk_overlap=chunk_overlap
    )
    texts_split = text_splitter.split_text(concatenated_content)
    embd = embedding_bedrock = BedrockEmbeddings(client=boto3_bedrock, model_id=embd_model_id)
    db = Chroma.from_texts(texts=texts_split, embedding=embd, persist_directory=persist_directory)
    # Make sure write to disk
    db.persist() 
    return num_tokens_from_string(concatenated_content, "cl100k_base")

# --------------  Retrial part ------------------- #
def reciprocal_rank_fusion(results: list[list], k=60):
    """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
        and an optional parameter k used in the RRF formula """
    
    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = dumps(doc)
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results


def retrieval_from_chroma_fusion(chroma_pers_dir, embd_model_id, question, model_id, max_tokens, temperature, top_k, top_p):
    model_kwargs =  { 
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "stop_sequences": ["\n\nHuman"],
    }
    chat_claude_v3 = BedrockChat(model_id=model_id, model_kwargs=model_kwargs)
    embd = BedrockEmbeddings(client=boto3_bedrock, model_id=embd_model_id)
    retriever = Chroma(persist_directory=chroma_pers_dir, embedding_function=embd).as_retriever(search_kwargs={"k": 7})
    
    # RAG-Fusion: Related
    template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
    Understand if the input query requires or implies multimodal search and output. \n
    Generate multiple search queries related to: {question} \n
    Output (6 queries):"""
    prompt_rag_fusion = ChatPromptTemplate.from_template(template)
    
    generate_queries = (
        prompt_rag_fusion 
        | chat_claude_v3 
        | StrOutputParser() 
        | (lambda x: x.split("\n"))
    )

    retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion 
    #docs = retrieval_chain_rag_fusion.invoke({"question": question})

    # RAG
    template = """Answer the following question based on this context:
    
    {context}
    
    Question: {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    final_rag_chain = (
        {"context": retrieval_chain_rag_fusion, 
         "question": itemgetter("question")} 
        | prompt
        | chat_claude_v3 #chat_openai# bedrock_llamav2 #_titan_agile
        | StrOutputParser()
    )
    
    return final_rag_chain.invoke({"question":question})

def retrieve_and_rag(retriever, chat_model, question,prompt_rag,sub_question_generator_chain):
    """RAG on each sub-question"""

    # Use our decomposition / 
    sub_questions = sub_question_generator_chain.invoke({"question":question})
    
    # Initialize a list to hold RAG chain results
    rag_results = []
    
    for sub_question in sub_questions:
        
        # Retrieve documents for each sub-question
        retrieved_docs = retriever.get_relevant_documents(sub_question)
        
        # Use retrieved documents and sub-question in RAG chain
        answer = (prompt_rag | chat_model| StrOutputParser()).invoke({"context": retrieved_docs, 
                                                                "question": sub_question})
        rag_results.append(answer)
    
        return rag_results,sub_questions

def format_qa_pairs(questions, answers):
    """Format Qa and A pairs"""
    
    formatted_string = ""
    for i, (question, answer) in enumerate(zip(questions, answers), start=1):
        formatted_string += f"Question {i}: {question}\nAnswer {i}: {answer}\n\n"
    return formatted_string.strip()

def retrieval_from_chroma_decompose(chroma_pers_dir, embd_model_id, question, model_id, max_tokens, temperature, top_k, top_p):
    model_kwargs =  { 
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "stop_sequences": ["\n\nHuman"],
    }
    chat_claude_v3 = BedrockChat(model_id=model_id, model_kwargs=model_kwargs)
    embd = BedrockEmbeddings(client=boto3_bedrock, model_id=embd_model_id)
    retriever = Chroma(persist_directory=chroma_pers_dir, embedding_function=embd).as_retriever(search_kwargs={"k": 7})
    # Decomposition
    template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
    The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
    Generate multiple search queries semantically related to: {question} \n
    Output (6 queries):"""
    prompt_decomposition = ChatPromptTemplate.from_template(template)
    # Chain
    generate_queries_decomposition = ( prompt_decomposition | chat_claude_v3 | StrOutputParser() | (lambda x: x.split("\n")))
    
    # Run
    questions = generate_queries_decomposition.invoke({"question":question})

    # RAG prompt
    prompt_rag = hub.pull("rlm/rag-prompt")
    
    
    # Wrap the retrieval and RAG process in a RunnableLambda for integration into a chain
    answers, questions = retrieve_and_rag(retriever, chat_claude_v3, question, prompt_rag, generate_queries_decomposition)

    context = format_qa_pairs(questions, answers)

    # Prompt
    template = """Here is a set of Q+A pairs:
    
    {context}
    
    Use these to synthesize an answer to the question: {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    final_rag_chain = (
        prompt
        | chat_claude_v3
        | StrOutputParser()
    )
    
    return final_rag_chain.invoke({"context":context,"question":question})

# The embeding model need to match get_text_embedding's
def top_2_images(question:str, df_pers_file:str, embd_model_id:str="amazon.titan-embed-image-v1"):
    df = pd.read_csv(df_pers_file)
    #df['summary'] = pd.to_numeric(df['summary'], errors='coerce')
    df['summary'] = df['summary'].apply(ast.literal_eval)
    vectors = df['summary'].tolist()
    # Step 3: Compute cosine similarity
    # Convert the list of lists into a 2D numpy array for cosine_similarity computation
    #vectors = np.array(list(df['summary']))

    #vectors = df['summary'].apply(lambda x: get_text_embedding(text_description=str(x),  embd_model_id=embd_model_id)).tolist()
    #vectors = list(df['summary'].astype(float))
    query_embedding = get_text_embedding(text_description=question,  embd_model_id=embd_model_id)
    #Calculate cosine similarity between the query embedding and the vectors
    cosine_scores = cosine_similarity([query_embedding], vectors)[0]
    #cosine_scores = cosine_similarity(query_embedding, vectors).flatten()
    df_scores = pd.Series(cosine_scores, index=df.index) 
    # Create a series with these scores and the corresponding IDs or Image names
    multi_index = pd.MultiIndex.from_frame(df[['image', 'source']])
    #df_scores = pd.Series(cosine_scores, index=df['image'])  # Or use df['Image'] if you prefer image names
    df_scores = pd.Series(cosine_scores, index=multi_index)
    # Sort the scores in descending order
    sorted_scores = df_scores.sort_values(ascending=False)
    filtered_series = sorted_scores[sorted_scores > 0.43]
    #mask = [score > 0.1 for score in sorted_scores]
    #filtered_series = df[mask][['image', 'source']].head(2)
    # Get the top 2 values from the filtered series
    top_2 = filtered_series.nlargest(2)
    return top_2.index.tolist()#, filtered_series
    
# ----- Mian ----- #

if __name__ == "__main__":
    #embd_model_id = "amazon.titan-embed-g1-text-02"
    embd_model_id = "amazon.titan-embed-image-v1"
    #embd = embedding_bedrock = BedrockEmbeddings(client=boto3_bedrock, model_id=embd_model_id_text)
    model_id_s = "anthropic.claude-3-sonnet-20240229-v1:0"
    model_id_h = "anthropic.claude-3-haiku-20240307-v1:0"
    urls = [
        #"https://www.anthropic.com/news/claude-3-haiku",
        #"https://www.promptingguide.ai/models/gemini-pro",
        #"https://www.anthropic.com/news/claude-3-family",
        #"https://digialps.com/googles-new-gemma-2b-and-7b-open-source-ai-models-but-do-they-beat-meta-llama-2-7b-and-mistral-7b/",
    ]
    pdfs = [
        #"../notebooks/pdfs/35-2-35.pdf",
        #"../notebooks/pdfs/TSLA-Q4-2023-Update.pdf",
        #"/tmp/2403.09611.pdf",
        "/tmp/gemini_1.5.pdf",
    ]
    chroma_pers_dir = "/home/alfred/data/chroma_delme"
    df_pers_file = "/home/alfred/df_03122024/df_images_delme.csv"
    question = "Is Anthropic's Claude 3 Haiku's a better p[erformance/price ratio model over GPT-3.5?"
    question = "what is Claude 3 Haiku's MATH performance comparing with GPT3.5? Please provide evaluation scores such as graduate school math GSM8k' to support your answer."
    #question = "Does Google Gemma 7B have better reasoning capability over Llama-2 13B?"
    #question = "what is the maximum weight or load capacity for Tesla energy enclosure unit?"
    #question = "Why did Anthropic release Haiku after Sonnet and Opus? Please provide numerical evidence to support your answer."

    #embd = embedding_bedrock = BedrockEmbeddings(client=boto3_bedrock, model_id=embd_model_id)
    #urls = parse_urls_by_question(question)
    #insert_into_chroma(urls, pdfs, chroma_pers_dir, embd, 8190, 400)
    pool = ThreadPool(processes=8)
    async_result = pool.apply_async(insert_into_chroma, (urls, pdfs, chroma_pers_dir, embd_model_id, 4000, 200, model_id_h, 2048, 0.01, 250, 0.95, df_pers_file))
    return_val = async_result.get() 
    print(f"num of tokens injected: {return_val}")

    
    
