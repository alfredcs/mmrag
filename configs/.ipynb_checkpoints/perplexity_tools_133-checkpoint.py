import requests # Required to make HTTP requests
from bs4 import BeautifulSoup  # Required to parse HTML
import numpy as np # Required to dedupe sites
from urllib.parse import unquote # Required to unquote URLs
from xml.etree import ElementTree
from langchain_core.documents.base import Document
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.chat_models import BedrockChat
#from langchain.llms.bedrock import Bedrock
from langchain import hub
from operator import itemgetter
from langchain_community.llms.bedrock import Bedrock
from langchain.text_splitter import CharacterTextSplitter

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory, ConversationBufferMemory
from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.callbacks.base import BaseCallbackHandler

from langchain.chains import ConversationalRetrievalChain
#from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import sys, os, json, boto3, botocore, time, csv
from readabilipy import simple_json_from_html_string # Required to parse HTML to pure text
#from langchain.schema import Document # Required to create a Document object
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
#from langchain.vectorstores import OpenSearchVectorSearch
from langchain_community.vectorstores import OpenSearchVectorSearch
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from multiprocessing.pool import ThreadPool



#----------- Parse out web content -----------
def scrape_and_parse(url: str) -> Document:
    """Scrape a webpage and parse it into a Document object"""
    req = requests.get(url)
    article = simple_json_from_html_string(req.text, use_readability=False)
    # The following line seems to work with the package versions on my local machine, but not on Google Colab
    # return Document(page_content=article['plain_text'][0]['text'], metadata={'source': url, 'page_title': article['title']})
    return Document(page_content='\n\n'.join([a['text'] for a in article['plain_text']]), metadata={'source': url, 'page_title': article['title']})


# Saerch google and bing with a query and return urls
class newsSearcher:
    def __init__(self):
        self.google_url = "https://www.google.com/search?q="
        self.bing_url = "https://www.bing.com/search?q="
        #self.bing_url = "https://www.bing.com/search?q={query.replace(' ', '+')}"

    def search(self, query, count: int=10):
        google_urls = self.search_goog(query, count)
        bing_urls = self.search_bing(query, count)
        combined_urls = google_urls + bing_urls
        urls = list(set(combined_urls))  # Remove duplicates
        return [scrape_and_parse(f) for f in urls], urls # Scrape and parse all the url

    def search_goog(self, query, count):
        #response = requests.get(f"https://www.google.com/search?q={query}") # Make the request
        params = {
            "q": query,
            "num": count  # Number of results to retrieve
        }
        response = requests.get(self.google_url, params=params)
        soup = BeautifulSoup(response.text, "html.parser") # Parse the HTML
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
        return urls
        
    def search_google(self, query, count):
        params = {
            "q": query,
            "num": count  # Number of results to retrieve
        }
        response = requests.get(self.google_url, params=params)
        soup = BeautifulSoup(response.text, "html.parser")
        urls = [link.get("href") for link in soup.select(".yuRUbf a")]
        return urls

    def search_bing(self, query, count):
        params = {
            "q": query,
            "count": count # Number of results to retrieve
        }
        response = requests.get(self.bing_url, params=params)
        soup = BeautifulSoup(response.text, "html.parser")
        urls = [link.get("href") for link in soup.select(".b_algo h2 a")]
        return urls[:count]


class arxivSearcher:
    def __init__(self):
        self.base_url = "http://export.arxiv.org/api/query?"
        
    def search_and_convert(self, query, max_results=10):
        """Search arXiv, parse the results, and convert them to LangChain Document objects."""
        params = {"search_query": query, "start": 0, "max_results": max_results}
        response = requests.get(self.base_url, params=params)
        
        if response.status_code == 200:
            entries = self.parse_response(response.text)
            return [self.convert_to_document(entry) for entry in entries]
        else:
            print(f"Error fetching results from arXiv: {response.status_code}")
            return []
        
    def parse_response(self, xml_data):
        """Parse the XML response from arXiv and extract relevant data."""
        namespace = {'arxiv': 'http://www.w3.org/2005/Atom'}
        root = ElementTree.fromstring(xml_data)
        entries_data = []
        
        for entry in root.findall('arxiv:entry', namespace):
            entry_data = {
                'title': entry.find('arxiv:title', namespace).text,
                'summary': entry.find('arxiv:summary', namespace).text,
                'authors': [author.find('arxiv:name', namespace).text for author in entry.findall('arxiv:author', namespace)],
                'id': entry.find('arxiv:id', namespace).text,
                'link': entry.find('arxiv:link[@rel="alternate"]', namespace).attrib.get('href'),
                'published': entry.find('arxiv:published', namespace).text
            }
            entries_data.append(entry_data)
        
        return entries_data
    
    def convert_to_document(self, entry):
        """Convert an entry to a LangChain Document object."""
        # Adjust attributes according to the actual Document class definition
        document = Document(
            page_content=entry['summary'],
            metadata={
                'title': entry['title'],
                'authors': entry['authors'],
                'id': entry['id'],
                'link': entry['link'],
                'published': entry['published']
            }
        )
        return document

class healthSearcher:
    def __init__(self):
        self.pubmed_url = "https://pubmed.ncbi.nlm.nih.gov/?term="
        self.pmc_url = "https://www.ncbi.nlm.nih.gov/pmc/?term="

    def search_pubmed(self, query):
        """Search PubMed and return combined results as HTML."""
        response = requests.get(f"{self.pubmed_url}{query}")
        if response.status_code == 200:
            return response.text
        return ""

    def search_pmc(self, query):
        """Search PubMed Central (PMC) and return combined results as HTML."""
        response = requests.get(f"{self.pmc_url}{query}")
        if response.status_code == 200:
            return response.text
        return ""

    def combine_searches(self, query):
        """Combine searches from PubMed and PMC."""
        pubmed_results = self.search_pubmed(query)
        pmc_results = self.search_pmc(query)
        combined_results = pubmed_results + pmc_results  # Simplified combination for demonstration
        return combined_results

    def create_document(self, query):
        """Create a LangChain Document from the combined search results."""
        combined_results = self.combine_searches(query)
        # Simplification: In a real scenario, parse combined_results to extract structured data
        # For demonstration, we'll assume combined_results is a simple text representation of the content
        document = Document(
            page_content=combined_results,  # Same as above
            metadata={
                'query': query
                # Add other metadata as needed
            }
        )
        return document
    
#--- Configure Bedrock -----
def config_bedrock(embedding_model_id, model_id, max_tokens, temperature, top_p, top_k):
    bedrock_client = boto3.client('bedrock-runtime')
    embedding_bedrock = BedrockEmbeddings(client=bedrock_client, model_id=embedding_model_id)
    model_kwargs =  { 
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "stop_sequences": ["\n\nHuman"],
    }
    chat = BedrockChat(
        model_id=model_id, client=bedrock_client, model_kwargs=model_kwargs
    )
    #llm = Bedrock(
    #    model_id=model_id, client=bedrock_client, model_kwargs=model_kwargs
    #)

    return chat, embedding_bedrock


#------ Creat an AOSS index ----
def create_aoss(vector_store_name):
    #index_name = "bedrock-workshop-rag-index"
    encryption_policy_name = f"{vector_store_name}-sp"
    network_policy_name = f"{vector_store_name}-np"
    access_policy_name = f"{vector_store_name}-ap"
    identity = boto3.client('sts').get_caller_identity()['Arn']
    aoss_client = boto3.client('opensearchserverless')
    status =  aoss_client.list_collections(collectionFilters={'name':vector_store_name})
    if len(status['collectionSummaries']) > 0:
        return  status['collectionSummaries'][0]['id'] + '.' + os.environ.get("AWS_DEFAULT_REGION", None) + '.aoss.amazonaws.com:443'
    else:
        security_policy = aoss_client.create_security_policy(
            name = encryption_policy_name,
            policy = json.dumps(
                {
                    'Rules': [{'Resource': ['collection/' + vector_store_name],
                    'ResourceType': 'collection'}],
                    'AWSOwnedKey': True
                }),
            type = 'encryption'
        )
        
        network_policy = aoss_client.create_security_policy(
            name = network_policy_name,
            policy = json.dumps(
                [
                    {'Rules': [{'Resource': ['collection/' + vector_store_name],
                    'ResourceType': 'collection'}],
                    'AllowFromPublic': True}
                ]),
            type = 'network'
        )
        
        collection = aoss_client.create_collection(name=vector_store_name,type='VECTORSEARCH')
        
        while True:
            status = aoss_client.list_collections(collectionFilters={'name':vector_store_name})['collectionSummaries'][0]['status']
            if status in ('ACTIVE', 'FAILED'): break
            time.sleep(10)
        
        access_policy = aoss_client.create_access_policy(
            name = access_policy_name,
            policy = json.dumps(
                [
                    {
                        'Rules': [
                            {
                                'Resource': ['collection/' + vector_store_name],
                                'Permission': [
                                    'aoss:CreateCollectionItems',
                                    'aoss:DeleteCollectionItems',
                                    'aoss:UpdateCollectionItems',
                                    'aoss:DescribeCollectionItems'],
                                'ResourceType': 'collection'
                            },
                            {
                                'Resource': ['index/' + vector_store_name + '/*'],
                                'Permission': [
                                    'aoss:CreateIndex',
                                    'aoss:DeleteIndex',
                                    'aoss:UpdateIndex',
                                    'aoss:DescribeIndex',
                                    'aoss:ReadDocument',
                                    'aoss:WriteDocument'],
                                'ResourceType': 'index'
                            }],
                        'Principal': [identity],
                        'Description': 'Easy data policy'}
                ]),
            type = 'data'
        )
        
        host = collection['createCollectionDetail']['id'] + '.' + os.environ.get("AWS_DEFAULT_REGION", None) + '.aoss.amazonaws.com:443'
        return host


#=---- Upload an image to a S3 ----
def upload_to_s3(image_path:str, bucket_name:str, region_name:str ):
    s3_client = boto3.client("s3")
    def uploadDirectory(path,bucket_name):
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
    last_slash_pos = image_path.rfind('/')
    # Extract the substring from the beginning to the last '/' character
    directory_path = image_path[:last_slash_pos]
    #esponse = create_new_bucket(bucket_name, region_name)
    #uploadDirectory(directory_path, bucket_name)
    s3_client.upload_file(image_path,bucket_name,image_path)
    return f"s3://{bucket_name}/{image_path}"


#--- Insert into aoss ----
def insert_text_aoss(documents,bedrock_embeddings, host: str, index_name: str):
    credentials = boto3.Session().get_credentials()
    service = 'aoss'
    auth = AWSV4SignerAuth(credentials, os.environ.get("AWS_DEFAULT_REGION", None), service)

    # Splitter
    text_splitter = CharacterTextSplitter(separator=' ', chunk_size=8000, chunk_overlap=800)
    texts = text_splitter.split_documents(documents)

    
    docsearch = OpenSearchVectorSearch.from_documents(
        texts,
        bedrock_embeddings,
        opensearch_url=host,
        http_auth=auth,
        timeout = 100,
        use_ssl = True,
        verify_certs = True,
        connection_class = RequestsHttpConnection,
        index_name=index_name,
        engine="faiss",
    )
    return docsearch

#---- Saerch AOSS ----
def search_aoss(query: str, index_name: str, host:str, bedrock_embeddings, top_k: int):
    credentials = boto3.Session().get_credentials()
    auth = AWSV4SignerAuth(credentials, os.environ.get("AWS_DEFAULT_REGION", None), "aoss")
    new_docsearch = OpenSearchVectorSearch(
        index_name=index_name,  # TODO: use the same index-name used in the ingestion script
        embedding_function=bedrock_embeddings,
        opensearch_url=host,  # TODO: e.g. use the AWS OpenSearch domain instantiated previously
        http_auth=auth,
        timeout = 100,
        use_ssl = True,
        verify_certs = True,
        connection_class = RequestsHttpConnection,
        engine="faiss",
    )
    results = new_docsearch.similarity_search_with_score(query, k=top_k)  # our search query  # return 3 most relevant docs
    return results


#--- Basic chain ---
def bedrock_claude3_chain(query: str, docs:str, chat):
    '''
    messages = [
        ("system", "You are a great question and answer assistance."),
        ("human", "{question}"),
    ]
    
    prompt = ChatPromptTemplate.from_messages(messages)
    
    chain = prompt | chat | StrOutputParser()
    
    # Chain Invoke
    return chain.invoke({"question": query})
    '''
    
    # Decomposition
    template = """You are an AI assistant designed to provide comprehensive answers to the best of your abilities. \n
        Your responses should be based on the provided {context} information and address the specific {question}. \n
        Your answer should include all corresponding source urls in hyperlink format at the end. \n
        Please avoid repeating the question in your answer.\n
        Simply say I don't know if you can not find any evidence to match the question. \n\n 
        AI:"""
    prompt_decomposition = ChatPromptTemplate.from_template(template)
    # Chain
    generate_queries_decomposition = ( prompt_decomposition | chat | StrOutputParser() ) #| (lambda x: x.split("\n")))
    
    # Run
    answer = generate_queries_decomposition.invoke({"context": docs, "question":query})
    return answer
  

def basic_chain(query: str, index_name, aoss_host, llm_chat, bedrock_embeddings):
    prompt_template = hub.pull("rlm/rag-prompt")
    credentials = boto3.Session().get_credentials()
    auth = AWSV4SignerAuth(credentials, os.environ.get("AWS_DEFAULT_REGION", None), "aoss")
    
    docsearch = OpenSearchVectorSearch(
        index_name=index_name,  # TODO: use the same index-name used in the ingestion script
        embedding_function=bedrock_embeddings,
        opensearch_url=aoss_host,  # TODO: e.g. use the AWS OpenSearch domain instantiated previously
        http_auth=auth,
        timeout = 100,
        use_ssl = True,
        verify_certs = True,
        connection_class = RequestsHttpConnection,
        engine="faiss",
    )

    retriever = docsearch.as_retriever(search_kwargs={"k": 3})
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm_chat
        | StrOutputParser()
    )
    return rag_chain.invoke(query)


#---- Conversational Chain ----
def conversational_chain(query: str, index_name, host, llm_chat, bedrock_embeddings):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    #memory = ConversationBufferMemory(return_messages=True)
    #memory.load_memory_variables({})
    credentials = boto3.Session().get_credentials()
    auth = AWSV4SignerAuth(credentials, os.environ.get("AWS_DEFAULT_REGION", None), "aoss")
    
    docsearch = OpenSearchVectorSearch(
        index_name=index_name,  # TODO: use the same index-name used in the ingestion script
        embedding_function=bedrock_embeddings,
        opensearch_url=host,  # TODO: e.g. use the AWS OpenSearch domain instantiated previously
        http_auth=auth,
        timeout = 100,
        use_ssl = True,
        verify_certs = True,
        connection_class = RequestsHttpConnection,
        engine="faiss",
    )

    retriever = docsearch.as_retriever(search_kwargs={"k": 3})
    #bot = ConversationalRetrievalChain.from_llm(
    #    llm_chat, retriever, memory=memory, verbose=False
    #)
    #prompt_template = ChatPromptTemplate.from_template("Answer questions based on the context below: {context} / Question: {question}")
    #prompt_template = f"Answer questions based on the context below: {context} / Question: {question}. \n\n  AI:"
    messages = [
        ("system", """Your are a helpful assistant to provide omprehensive and truthful answers to questions, \n
                    drawing upon all relevant information contained within the specified in {context}. \n 
                    You add value by analyzing the situation and offering insights to enrich your answer. \n
                    Simply say I don't know if you can not find any evidence to match the question. \n
                    Extract the corresponding sources and add the clickable, relevant and unaltered URLs in hyperlink format to the end of your answer."""),
        #MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt | llm_chat | StrOutputParser())
    #chain = ({"context": retriever, "question": RunnablePassthrough(chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("chat_history"))} | prompt | llm_chat | StrOutputParser())
    return chain.invoke(query)



#---- Text classification ---
def classify_query(query, classes: str, modelId: str):
    """
    Classify a query into 'Tech', 'Health', or 'General' using an LLM.

    :param query: The query string to classify.
    :param openai_api_key: Your OpenAI API key.
    :return: A string classification: 'Tech', 'Health', or 'General'.
    """
    bedrock_client = boto3.client('bedrock-runtime')
    
    # Constructing the prompt for the LLM
    prompt = f"Human:Classify the following query into one of these categories: {classes}.\n\nQuery: {query}\n\n Please answer directly with the catergory name only. \n\n  AI:"
    payload = {
            "modelId": modelId,
            "contentType": "application/json",
            "accept": "application/json",
            "body": {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4096,
                "temperature": 0.01,
                "top_k": 250,
                "top_p": 0.95,
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
    try:
        # Convert the payload to bytes
        body_bytes = json.dumps(payload['body']).encode('utf-8')
        # Invoke the model
        response = bedrock_client.invoke_model(
            body=body_bytes,
            contentType=payload['contentType'],
            accept=payload['accept'],
            modelId=payload['modelId']
        )


        #response = bedrock_client.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
        response_body = json.loads(response.get('body').read())
        classification = ''.join([item['text'] for item in response_body['content'] if item.get('type') == 'text'])
        # Assuming the most likely category is returned directly as the output text
        #classification = response.choices[0].text.strip()
        return classification
    except Exception as e:
        print(f"Error classifying query: {e}")
        return "Error"

#--- Text semantic similarity check and cache ----
def get_text_embedding(image_base64=None, text_description=None,  embd_model_id:str="amazon.titan-embed-image-v1"):
    input_data = {}
    bedrock_client = boto3.client('bedrock-runtime')
    
    if image_base64 is not None:
        input_data["inputImage"] = image_base64
    if text_description is not None:
        input_data["inputText"] = text_description

    if not input_data:
        raise ValueError("At least one of image_base64 or text_description must be provided")

    body = json.dumps(input_data)

    response = bedrock_client.invoke_model(
        body=body,
        modelId=embd_model_id,
        accept="application/json",
        contentType="application/json"
    )

    response_body = json.loads(response.get("body").read())
    return response_body.get("embedding")
    
def check_and_cache_string(input_string:str, SIMILARITY_THRESHOLD:float, cache_file:str, embd_model_id:str="amazon.titan-embed-image-v1"):
    """
    Check if the input string is semantically similar to any string in the cached list.
    If not, append the input string to the cached list.
    """
    vectors = []
    cached_strings = []
    with open(cache_file, 'r') as file:
        # Create a CSV reader object
        reader = csv.reader(file)
        # Iterate over each row in the CSV file
        for row in reader:
            cached_strings.append(row)
            vector_str = row[1].strip('[]')  # Remove the square brackets
            vector_values = [float(x) for x in vector_str.split(',')]
            vectors.append((vector_values))
    
    #df = pd.read_csv(cache_file)
    # Encode the input string
    input_embedding = get_text_embedding(text_description=input_string,  embd_model_id=embd_model_id)
    if len(vectors) > 0 :
        cosine_scores = cosine_similarity([input_embedding], vectors)[0]
        df_scores = pd.Series(cosine_scores)
        sorted_scores = df_scores.sort_values(ascending=False)
        histories = sorted_scores[sorted_scores >= SIMILARITY_THRESHOLD]
        if len(histories) > 0 :
            print(f"'{input_string}' is semantically similar to a record in the cache history with cosine {cosine_scores}")
            return False

    # If no similar string found, append the input string to the cached list
    cached_strings.append((input_string, input_embedding))
    print(f"'{input_string}' added to the cached list")
    with open(cache_file, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write each row of the list to the CSV file
        for row in cached_strings:
            writer.writerow(row)

    return True


#--- Wrappler --
def bedrock_textGen_perplexity(option, prompt, max_token, temperature, top_p, top_k, stop_sequences, embd_model_id):
    aoss_collection_name = "mmrag-collection-032024"
    aoss_text_index_name = "mmrag-text-index"
    os.environ["AWS_DEFAULT_REGION"] = 'us-west-2'
    # Read cache search history
    cache_file = "/home/alfred/data/search_cache.csv"
    # Define the similarity threshold
    SIMILARITY_THRESHOLD = 0.75 
    # Initialize the cached list
    status = check_and_cache_string(prompt, SIMILARITY_THRESHOLD, cache_file, embd_model_id=embd_model_id)
    urls = []

    # Configure Bedrock LLM and Chat
    chat, embd = config_bedrock(embd_model_id, option, max_tokens=max_token, temperature=temperature, top_p=top_p, top_k=top_k)

    # Get AOSS host string
    aoss_host = create_aoss(aoss_collection_name)
    #print(aoss_host)

    # If new query or not data in AOSS then search and inert
    if status:
        # Text classification
        classes = "Technology, Health, News"
        classification = classify_query(prompt, classes, option)
        if '_technology' in classification.lower():
            searcher = arxivSearcher()
            documents = searcher.search_and_convert(option)
        elif "_health" in classification.lower():
            searcher = healthSearcher()
            documents = searcher.create_document(option)
        else:
            searcher = newsSearcher()
            documents, urls = searcher.search(option)
    
        # Insert into AOSS
        docsearcher = insert_text_aoss(documents,embd, aoss_host, aoss_text_index_name)

    # Query using conversational chain
    if 'naive' in stop_sequences.lower():
        docs = search_aoss(prompt, aoss_text_index_name, aoss_host, embd, top_k=5)
        results = bedrock_claude3_chain(prompt, docs, chat)
    else:
        results = conversational_chain(prompt, aoss_text_index_name, aoss_host, chat, embd)
    return results, urls

def bedrock_textGen_perplexity_memory(option, prompt, max_token, temperature, top_p, top_k, stop_sequences, embd_model_id):
    if "titan-embed-text" in embd_model_id:
        embedding_size = 1536 # Dimensions of the amazon.titan-embed-text-v1
    elif "titan-embed-image" in embd_model_id:
        embedding_size = 1024 #amazon.titan-embed-image-v1
    else: 
        embedding_size = 4096

    # Configure Bedrock LLM and Chat
    chat, embd = config_bedrock(embd_model_id, option, max_tokens=max_token, temperature=temperature, top_p=top_p, top_k=top_k)

    # Get FAISS host string
    index = faiss.IndexFlatL2(embedding_size)
    #embedding_fn = OpenAIEmbeddings().embed_query
    embedding_fn = embd.embed_query
    vectorstore = FAISS(embedding_fn, index, InMemoryDocstore({}), {})
    #print(aoss_host)


    # Insert into FAISS
    # Text classification
    classes = "Technology, Health, News"
    classification = classify_query(prompt, classes, option)
    if '_technology' in classification.lower():
        searcher = arxivSearcher()
        documents = searcher.search_and_convert(option)
    elif "_health" in classification.lower():
        searcher = healthSearcher()
        documents = searcher.create_document(option)
    else:
        searcher = newsSearcher()
        documents, urls = searcher.search(option)

    docsearcher = FAISS.from_documents(documents, embd)

    # Query using conversational chain
    if 'naive' in stop_sequences.lower():
        retriever = docsearcher.as_retriever(search_kwargs={"k": top_k})
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt_template
            | llm_chat
            | StrOutputParser()
        )
        results = rag_chain.invoke(query)
    else:
        results = conversational_chain(prompt, aoss_text_index_name, aoss_host, chat, embd)
        retriever = docsearch.as_retriever(search_kwargs={"k": 3})
    #bot = ConversationalRetrievalChain.from_llm(
    #    llm_chat, retriever, memory=memory, verbose=False
    #)
    #prompt_template = ChatPromptTemplate.from_template("Answer questions based on the context below: {context} / Question: {question}")
    #prompt_template = f"Answer questions based on the context below: {context} / Question: {question}. \n\n  AI:"
    messages = [
        ("system", """Your are a helpful assistant to provide omprehensive and truthful answers to questions, \n
                    drawing upon all relevant information contained within the specified in {context}. \n 
                    You add value by analyzing the situation and offering insights to enrich your answer. \n
                    Simply say I don't know if you can not find any evidence to match the question. \n
                    Extract the corresponding sources and add the clickable, relevant and unaltered URLs in hyperlink format to the end of your answer."""),
        #MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt | llm_chat | StrOutputParser())
    #chain = ({"context": retriever, "question": RunnablePassthrough(chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("chat_history"))} | prompt | llm_chat | StrOutputParser())
    return chain.invoke(query)
    return results, urls

#--- TBD Image ---#

def bedrock_imageGen_perplexity(option, prompt, max_token, temperature, top_p, top_k, stop_sequences, embd_model_id):
    aoss_collection_name = "mmrag-collection-032024"
    aoss_image_index_name = "mmrag-image-index"
    
    #get AOSS host address
    aoss_host = create_aoss(aoss_collection_name)

    # Get the text and urls from the query
    results, urls = bedrock_textGen_perplexity(option, prompt, max_token, temperature, top_p, top_k, 'naive', embd_model_id)
    pool = ThreadPool(processes=8)
    #async_result = pool.apply_async(insert_into_chroma, (urls, pdfs, chroma_pers_dir, embd_model_id, 4000, 200, model_id_h, 2048, 0.01, 250, 0.95, df_pers_file))
    #return_val = async_result.get() 

    return results
    
#--- Main ---

if __name__ == "__main__":
    aoss_collection_name = "mmrag-collection-032024"
    aoss_text_index_name = "mmrag-text-index"
    aoss_image_index_name = "mmrag-image-index"
    os.environ["AWS_DEFAULT_REGION"] = 'us-west-2'
    modelId = 'anthropic.claude-3-haiku-20240307-v1:0'
    #modelId = "anthropic.claude-3-sonnet-20240229-v1:0"
    #modelId = "anthropic.claude-v2:1"
    titan_image_embedding = "amazon.titan-embed-image-v1"
    titan_text_embedding = "amazon.titan-embed-g1-text-02"

    # Create document from query
    #query = "What is the difference beyween MoE and Mamba for LLM models?"
    #query = "Who own the ship which caused the Key bridage collapse accident in Baltimore? What is the company's safty track record?"
    #query = "Did NTSB conclude the root cause of the Key bridage collapse accident in Baltimore?"
    query = "COVID-19 vaccine side effects"
    query = "What happened in the musical hall attack in Moscow last week?"
    query = "what is the prize for the Powerball jackpot drawing on Saturday April 30, 2024?"

    # Read cache search history
    cache_file = "/home/alfred/data/search_cache.csv"
    # Define the similarity threshold
    SIMILARITY_THRESHOLD = 0.75  
    # Initialize the cached list
    status = check_and_cache_string(query, SIMILARITY_THRESHOLD, cache_file, embd_model_id=titan_text_embedding)

    
    # Configure Bedrock LLM and Chat
    chat, embd = config_bedrock(titan_text_embedding, modelId, max_tokens=100000, temperature=0.01, top_p=0.95, top_k=250)

    # Get AOSS host string
    aoss_host = create_aoss(aoss_collection_name)
    #print(aoss_host)

    # If new query or not data in AOSS then search and inert
    if status:
        # Text classification
        classes = "Technology, Health, News"
        classification = classify_query(query, classes, modelId)
        if 'technology' in classification.lower():
            searcher = arxivSearcher()
            documents = searcher.search_and_convert(query)
        elif "health" in classification.lower():
            searcher = healthSearcher()
            documents = searcher.create_document(query)
        else:
            searcher = newsSearcher()
            documents, urls = searcher.search(query)
            print(urls)
    
        # Insert into AOSS
        docsearcher = insert_text_aoss(documents,embd, aoss_host, aoss_text_index_name)

    # Query from AOSS
    docs = search_aoss(query, aoss_text_index_name, aoss_host, embd, top_k=3)
    #print(docs)

    # Query using conversational chain
    results = bedrock_claude3_chain(query, docs, chat)
    print(f"{results}\n ----conversational chain-----\n")
    results = conversational_chain(query, aoss_text_index_name, aoss_host, chat, embd)
    print(f"{results}\n ----basic chain-----\n")
    results = basic_chain(query, aoss_text_index_name, aoss_host, chat, embd)
    print(results)
    
    
    