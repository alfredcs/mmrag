# ⚓️ Multi-modal Assistant With Advanced RAG And Amazon Bedrock Claude 3

Retrieval-Augmented Generation (RAG) models have emerged as a promising approach to enhancing the capabilities of language models by incorporating external knowledge from large text corpora. However, despite their impressive performance in various natural language processing tasks, RAG models still face several limitations that need to be addressed.

Naive RAG models face limitations such as missing content, reasoning mismatch, and challenges in handling multimodal data. While they can retrieve relevant information, they may struggle to generate complete and coherent responses when required information is absent, leading to incomplete or inaccurate outputs. Additionally, even with relevant information retrieved, the models may have difficulty correctly interpreting and reasoning over the content, resulting in inconsistencies or logical errors. Furthermore, effectively understanding and reasoning over multimodal data remains a significant challenge for these primarily text-based models.

In this blog post, we present a new approach named Multi-modal RAG (mmRAG) to tackle those existing limitations in greater detail. The solution intends to address these limitations for the practical generative AIassistant use cases. Additionally, we will examine potential solutions to enhance the capabilities of large language and visual language models with advanced Langchain capabilities, enabling them to generate more comprehensive, coherent, and accurate outputs while effectively handling multimodal data. Amazon Bedrock is a fully managed service that offers a choice of high-performing foundation models from leading AI companies, providing a broad set of capabilities to build generative AI applications with security, privacy, and responsible AI. 


## Release
- [3/21/24] Initial draft

**Usage and License Notices**: This project utilizes certain datasets and checkpoints that are subject to their respective original licenses. Users must comply with all terms and conditions of these original licenses, including but not limited to the [Amazon Terms of Use](https://aws.amazon.com/s) for the specific licenses for base vidual language models for tools (e.g. [Langchain community license](https://github.com/langchain-ai/langchain/blob/master/LICENSE) for Langchain. This project does not impose any additional constraints beyond those stipulated in the original licenses. Furthermore, users are reminded to ensure that their use of the dataset and checkpoints is in compliance with all applicable laws and regulations.  

## Content
- [Install](#install)
- [Models](#models)
- [Tools](#tools)
- [Demo](#demo)
- [Evaluation](#evaluation)
- [Blog](#blog)

## Install

1. Install the required Python packages
   ```
   pip install -t requirements.txt
   ```
3. Request an AWS account to provision Bedrock model acess via [Amazon Bedrock console](https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/modelaccess)

## Tools

## Demo

## Evaluation

## Blog
