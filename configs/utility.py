import json
import boto3
import random
import time
import os
import requests
import shutil

from typing import Optional
from botocore.config import Config
from langchain.llms.bedrock import Bedrock
from langchain import hub
from langchain_community.retrievers import AmazonKnowledgeBasesRetriever
from langchain_community.chat_models import BedrockChat
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import RunnableParallel
from langchain.prompts import ChatPromptTemplate


config_filename = '.aoss_config.txt'
suffix = random.randrange(200, 900)
boto3_session = boto3.session.Session()
region_name = boto3_session.region_name
iam_client = boto3_session.client('iam')
account_number = boto3.client('sts').get_caller_identity().get('Account')
identity = boto3.client('sts').get_caller_identity()['Arn']

# S3
sts_client = boto3.client('sts')
bedrock_agent_client = boto3_session.client('bedrock-agent', region_name=region_name)
bedrock_client = boto3.client("bedrock-runtime", region_name=region_name)
bedrock_agent_runtime_client = boto3.client("bedrock-agent-runtime", region_name=region_name)
s3_client = boto3.client('s3')
account_id = sts_client.get_caller_identity()["Account"]
s3_suffix = f"{region_name}-{account_id}"
bucket_name = f'bedrock-kb-{s3_suffix}' # replace it with your bucket name.


encryption_policy_name = f"bedrock-sample-rag-sp-{suffix}"
network_policy_name = f"bedrock-sample-rag-np-{suffix}"
access_policy_name = f'bedrock-sample-rag-ap-{suffix}'
bedrock_execution_role_name = f'AmazonBedrockExecutionRoleForKnowledgeBase_{suffix}'
fm_policy_name = f'AmazonBedrockFoundationModelPolicyForKnowledgeBase_{suffix}'
s3_policy_name = f'AmazonBedrockS3PolicyForKnowledgeBase_{suffix}'
oss_policy_name = f'AmazonBedrockOSSPolicyForKnowledgeBase_{suffix}'


def get_bedrock_client(
    assumed_role: Optional[str] = None,
    region: Optional[str] = None,
    runtime: Optional[bool] = True,
):
    """Create a boto3 client for Amazon Bedrock, with optional configuration overrides

    Parameters
    ----------
    assumed_role :
        Optional ARN of an AWS IAM role to assume for calling the Bedrock service. If not
        specified, the current active credentials will be used.
    region :
        Optional name of the AWS Region in which the service should be called (e.g. "us-east-1").
        If not specified, AWS_REGION or AWS_DEFAULT_REGION environment variable will be used.
    runtime :
        Optional choice of getting different client to perform operations with the Amazon Bedrock service.
    """
    if region is None:
        target_region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION"))
    else:
        target_region = region

    #print(f"Create new client\n  Using region: {target_region}")
    session_kwargs = {"region_name": target_region}
    client_kwargs = {**session_kwargs}

    profile_name = os.environ.get("AWS_PROFILE")
    if profile_name:
        #print(f"  Using profile: {profile_name}")
        session_kwargs["profile_name"] = profile_name

    retry_config = Config(
        region_name=target_region,
        retries={
            "max_attempts": 10,
            "mode": "standard",
        },
    )
    session = boto3.Session(**session_kwargs)

    if assumed_role:
        #print(f"  Using role: {assumed_role}", end='')
        sts = session.client("sts")
        response = sts.assume_role(
            RoleArn=str(assumed_role),
            RoleSessionName="langchain-llm-1"
        )
        #print(" ... successful!")
        client_kwargs["aws_access_key_id"] = response["Credentials"]["AccessKeyId"]
        client_kwargs["aws_secret_access_key"] = response["Credentials"]["SecretAccessKey"]
        client_kwargs["aws_session_token"] = response["Credentials"]["SessionToken"]

    if runtime:
        service_name='bedrock-runtime'
    else:
        service_name='bedrock'

    bedrock_client = session.client(
        service_name=service_name,
        config=retry_config,
        **client_kwargs
    )

    #print("boto3 Bedrock client successfully created!")
    #print(bedrock_client._endpoint)
    return bedrock_client
    
def create_bedrock_execution_role(bucket_name):
    foundation_model_policy_document = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "bedrock:InvokeModel",
                ],
                "Resource": [
                    f"arn:aws:bedrock:{region_name}::foundation-model/amazon.titan-embed-text-v1"
                ]
            }
        ]
    }

    s3_policy_document = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "s3:GetObject",
                    "s3:ListBucket"
                ],
                "Resource": [
                    f"arn:aws:s3:::{bucket_name}",
                    f"arn:aws:s3:::{bucket_name}/*"
                ],
                "Condition": {
                    "StringEquals": {
                        "aws:ResourceAccount": f"{account_number}"
                    }
                }
            }
        ]
    }

    assume_role_policy_document = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "bedrock.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }
    # create policies based on the policy documents
    fm_policy = iam_client.create_policy(
        PolicyName=fm_policy_name,
        PolicyDocument=json.dumps(foundation_model_policy_document),
        Description='Policy for accessing foundation model',
    )

    s3_policy = iam_client.create_policy(
        PolicyName=s3_policy_name,
        PolicyDocument=json.dumps(s3_policy_document),
        Description='Policy for reading documents from s3')

    # create bedrock execution role
    bedrock_kb_execution_role = iam_client.create_role(
        RoleName=bedrock_execution_role_name,
        AssumeRolePolicyDocument=json.dumps(assume_role_policy_document),
        Description='Amazon Bedrock Knowledge Base Execution Role for accessing OSS and S3',
        MaxSessionDuration=3600
    )

    # fetch arn of the policies and role created above
    bedrock_kb_execution_role_arn = bedrock_kb_execution_role['Role']['Arn']
    s3_policy_arn = s3_policy["Policy"]["Arn"]
    fm_policy_arn = fm_policy["Policy"]["Arn"]

    # attach policies to Amazon Bedrock execution role
    iam_client.attach_role_policy(
        RoleName=bedrock_kb_execution_role["Role"]["RoleName"],
        PolicyArn=fm_policy_arn
    )
    iam_client.attach_role_policy(
        RoleName=bedrock_kb_execution_role["Role"]["RoleName"],
        PolicyArn=s3_policy_arn
    )
    return bedrock_kb_execution_role


def create_oss_policy_attach_bedrock_execution_role(collection_id, bedrock_kb_execution_role):
    # define oss policy document
    oss_policy_document = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "aoss:APIAccessAll"
                ],
                "Resource": [
                    f"arn:aws:aoss:{region_name}:{account_number}:collection/{collection_id}"
                ]
            }
        ]
    }
    oss_policy = iam_client.create_policy(
        PolicyName=oss_policy_name,
        PolicyDocument=json.dumps(oss_policy_document),
        Description='Policy for accessing opensearch serverless',
    )
    oss_policy_arn = oss_policy["Policy"]["Arn"]
    print("Opensearch serverless arn: ", oss_policy_arn)

    iam_client.attach_role_policy(
        RoleName=bedrock_kb_execution_role["Role"]["RoleName"],
        PolicyArn=oss_policy_arn
    )
    return None


def create_policies_in_oss(vector_store_name, aoss_client, bedrock_kb_execution_role_arn):
    encryption_policy = aoss_client.create_security_policy(
        name=encryption_policy_name,
        policy=json.dumps(
            {
                'Rules': [{'Resource': ['collection/' + vector_store_name],
                           'ResourceType': 'collection'}],
                'AWSOwnedKey': True
            }),
        type='encryption'
    )

    network_policy = aoss_client.create_security_policy(
        name=network_policy_name,
        policy=json.dumps(
            [
                {'Rules': [{'Resource': ['collection/' + vector_store_name],
                            'ResourceType': 'collection'}],
                 'AllowFromPublic': True}
            ]),
        type='network'
    )
    access_policy = aoss_client.create_access_policy(
        name=access_policy_name,
        policy=json.dumps(
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
                    'Principal': [identity, bedrock_kb_execution_role_arn],
                    'Description': 'Easy data policy'}
            ]),
        type='data'
    )
    return encryption_policy, network_policy, access_policy


def delete_iam_role_and_policies():
    fm_policy_arn = f"arn:aws:iam::{account_number}:policy/{fm_policy_name}"
    s3_policy_arn = f"arn:aws:iam::{account_number}:policy/{s3_policy_name}"
    oss_policy_arn = f"arn:aws:iam::{account_number}:policy/{oss_policy_name}"
    iam_client.detach_role_policy(
        RoleName=bedrock_execution_role_name,
        PolicyArn=s3_policy_arn
    )
    iam_client.detach_role_policy(
        RoleName=bedrock_execution_role_name,
        PolicyArn=fm_policy_arn
    )
    iam_client.detach_role_policy(
        RoleName=bedrock_execution_role_name,
        PolicyArn=oss_policy_arn
    )
    iam_client.delete_role(RoleName=bedrock_execution_role_name)
    iam_client.delete_policy(PolicyArn=s3_policy_arn)
    iam_client.delete_policy(PolicyArn=fm_policy_arn)
    iam_client.delete_policy(PolicyArn=oss_policy_arn)
    return 0


def interactive_sleep(seconds: int):
    dots = ''
    for i in range(seconds):
        dots += '.'
        print(dots, end='\r')
        time.sleep(1)
    print('Done!')
    
# Get AOSS key and values
def read_key_value(file_path, key1):
    with open(file_path, 'r') as file:
        for line in file:
            key_value_pairs = line.strip().split(':')
            if key_value_pairs[0] == key1:
                return key_value_pairs[1].lstrip()
    return None

#Upload to S3
def uploadDirectory(path,bucket_name):
    for root,dirs,files in os.walk(path):
        for file in files:
            s3_client.upload_file(os.path.join(root,file),bucket_name,file)

def empty_directory(directory_path):
    if os.path.exists(directory_path):
        for item in os.scandir(directory_path):
            if item.is_file():
                os.remove(item.path)
            elif item.is_dir():
                #os.rmdir(item.path)
                shutil.rmtree(item.path)
        return True
    else:
        return False

def empty_versioned_s3_bucket(bucket_name):
    s3r = boto3.resource('s3')
    bucket = s3r.Bucket(bucket_name)
    bucket.object_versions.delete()
    return True

def bedrock_kb_injection(path):
    kb_id = read_key_value(config_filename, 'KB_id')
    ds_id = read_key_value(config_filename, 'DS_id')
    region_name = read_key_value(config_filename, 'Region')
    bucket_name = read_key_value(config_filename, 'S3_bucket_name')
    # Bedrock KB syncs with designated S3 bucket so be careful
    empty_versioned_s3_bucket(bucket_name)
    uploadDirectory(path,bucket_name)
    #ds = bedrock_agent_client.get_data_source(knowledgeBaseId = kb_id, dataSourceId = ds_id)
    start_job_response = bedrock_agent_client.start_ingestion_job(knowledgeBaseId = kb_id, dataSourceId = ds_id)
    job = start_job_response["ingestionJob"]
    while(job['status']!='COMPLETE' ):
      get_job_response = bedrock_agent_client.get_ingestion_job(
          knowledgeBaseId = kb_id,
            dataSourceId = ds_id,
            ingestionJobId = job["ingestionJobId"]
      )
      job = get_job_response["ingestionJob"]
    interactive_sleep(40)
    return job['statistics'], job['status'], kb_id

def bedrock_kb_retrieval(query: str, model_id: str) -> str:
    kb_id = read_key_value(config_filename, 'KB_id')
    model_arn = f'arn:aws:bedrock:{region_name}::foundation-model/{model_id}'
    prompt = f"""Your are a helpful assistant to provide comprehensive answers to the question by offering insights. \n
                    Please answer the following question:
                    <question>
                    {query}
                    </question>"""
    response = bedrock_agent_runtime_client.retrieve_and_generate(
        input={
            'text': prompt
        },
        retrieveAndGenerateConfiguration={
            'type': 'KNOWLEDGE_BASE',
            'knowledgeBaseConfiguration': {
                'knowledgeBaseId': kb_id,
                'modelArn': model_arn
            }
        },
    )

    generated_text = response['output']['text']
    return generated_text

# --- Decomposition with fusion----
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
    
def retrieve_and_rag(retriever, chat_model, question, prompt_rag,sub_question_generator_chain):
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
        
def bedrock_kb_retrieval_decomposition(query: str, model_id: str, max_tokens: int, temperature:float, top_p:float, top_k:int, stop_sequences:str) -> str:
    kb_id = read_key_value(config_filename, 'KB_id')
    prompt_template = hub.pull("rlm/rag-prompt")
    model_kwargs =  { 
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "stop_sequences":[stop_sequences],
    }
    chat_claude_v3 = BedrockChat(model_id=model_id, model_kwargs=model_kwargs)

    retriever = AmazonKnowledgeBasesRetriever(
        knowledge_base_id=kb_id,
        retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 8}},
    )

    # RAG-Fusion: Related
    # Decomposition
    template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
    The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
    Generate multiple search queries semantically related to: {question} \n
    Output (6 queries):"""
    prompt_decomposition = ChatPromptTemplate.from_template(template)
    generate_queries_decomposition = ( prompt_decomposition | chat_claude_v3 | StrOutputParser() | (lambda x: x.split("\n")))
    questions = generate_queries_decomposition.invoke({"question":query})
    # Wrap the retrieval and RAG process in a RunnableLambda for integration into a chain
    answers, questions = retrieve_and_rag(retriever, chat_claude_v3, query, prompt_template, generate_queries_decomposition)
    context = format_qa_pairs(questions, answers)

    # Now retrieval
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
    
    return final_rag_chain.invoke({"context":context,"question":query})


# ----- Retrieval -----
def bedrock_kb_retrieval_advanced(query: str, model_id: str, max_tokens: int, temperature:float, top_p:float, top_k:int, stop_sequences:str) -> str:
    kb_id = read_key_value(config_filename, 'KB_id')
    prompt_template = hub.pull("rlm/rag-prompt")

    prompt = f"""Your are a helpful assistant to provide comprehensive answers to the question by offering insights. \n
                    Your answers should include all insights and analysis related to the question and context. \n
                    Please answer the following question:
                    <question>
                    {query}
                    </question>"""
    model_kwargs =  { 
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "stop_sequences":[stop_sequences],
        }
    
    chat_claude_v3 = BedrockChat(model_id=model_id, model_kwargs=model_kwargs)
        
    retriever = AmazonKnowledgeBasesRetriever(
        knowledge_base_id=kb_id,
        retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 8}},
    )
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    
    rag_chain = (
        #{"context": retriever | format_docs, "question": RunnablePassthrough()}
        RunnableParallel(context = retriever | format_docs, question = RunnablePassthrough() )
        | prompt_template
        | chat_claude_v3
        | StrOutputParser()
    )
    
    return rag_chain.invoke(prompt)
    
def bedrock_textGen(model_id, prompt, max_tokens, temperature, top_p, top_k, stop_sequences):
    stop_sequence = [stop_sequences]
    if  "anthropic.claude-v2" in model_id.lower() or "anthropic.claude-instant" in model_id.lower():
        inference_modifier = {
            "max_tokens_to_sample": max_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "stop_sequences": stop_sequence,
        }
    
        textgen_llm = Bedrock(
            model_id=model_id,
            client=bedrock_client,
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
                "stop_sequences": stop_sequence,
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
        response = bedrock_client.invoke_model(
            body=body_bytes,
            contentType=payload['contentType'],
            accept=payload['accept'],
            modelId=payload['modelId']
        )
        
        # Process the response
        response_body = response['body'].read().decode('utf-8')
        data = json.loads(response_body)
        return data['content'][0]['text']
    elif 'meta' in model_id.lower():
        # Embed the message in Llama 3's prompt format.
        meta_prompt = f"""
        <|begin_of_text|>
        <|start_header_id|>user<|end_header_id|>
        {prompt}
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        """
        
        # Format the request payload using the model's native structure.
        request = {
            "prompt": meta_prompt,
            # Optional inference parameters:
            "max_gen_len": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        # Encode and send the request.
        response = bedrock_client.invoke_model(body=json.dumps(request), modelId=model_id)
        
        # Decode the native response body.
        model_response = json.loads(response["body"].read())
        
        # Extract and print the generated text.
        return model_response["generation"]


        '''
        inference_modifier = {
            "max_gen_len": max_tokens,
            "temperature": temperature,
            #"top_k": top_k,
            "top_p": top_p,
            #"stop_sequences": stop_sequence,
        }
    
        textgen_llm = Bedrock(
            model_id=model_id,
            client=bedrock_client,
            model_kwargs=inference_modifier,
        )     
        return textgen_llm(prompt)
        '''
    else:
        return f"Incorrect Bedrock model ID {model_id.lower()} selected!"
