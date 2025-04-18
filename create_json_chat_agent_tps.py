from langchain.prompts.chat import ChatPromptTemplate  # Updated import for compatibility
from langchain_community.agent_toolkits.json.toolkit import JsonToolkit  # Corrected import
from langchain_community.tools.json.tool import JsonSpec  # Corrected import
from langchain_ollama import ChatOllama  # Updated import to avoid deprecation
from langchain_community.agent_toolkits.json.base import create_json_agent  # Import create_json_agent
import json  # Use Python's built-in JSON module


from pathlib import Path
import pandas as pd
import torch
import numpy
from numpy import random

from torch.profiler import profile, record_function, ProfilerActivity

import sys
sys.path.append('/home/ailab/files/llama_index')
import llama_index
import llama_index.core
import llama_index.core.base.response.schema as response_schema
import llama_index.readers.file as file_readers
from llama_index.core import Document


# Load the JSON file
file = "/home/ailab/files/llama_index/data/llama2_paper/rag_dataset.json"  # Adjust the path to your JSON file
with open(file, "r", encoding="utf-8") as f:
    data = json.load(f)  # Use json.load to parse the JSON file

# Debugging information
print(f"\n[INFO] Loaded JSON data from file: {file}")
print(f"\n[INFO] JSON data loaded successfully. Number of examples: {len(data.get('examples', []))}")

# Define the JSON schema and toolkit
json_spec = JsonSpec(dict_=data, max_value_length=4000)  # Create a JSON schema
json_toolkit = JsonToolkit(spec=json_spec)  # Create a LangChain-processable JSON object

# Debugging information for JSON schema
print(f"\n[INFO] JSON schema keys: {list(data.keys())}")
if "examples" in data:
    print(f"[INFO] Number of queries in examples: {len(data['examples'])}")
    for i, example in enumerate(data['examples'][:5]):  # Print first 5 examples for debugging
        print(f"\n[INFO] Example {i+1}: Query: {example.get('query', 'N/A')}, Answer: {example.get('reference_answer', 'N/A')}")


# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Free up memory 
import gc
# Run garbage collection to free system memory
gc.collect()
import torch.distributed as dist
def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()
    return "\n[INFO] torch.distributed.destroy_process_group() cleanup SUCCESSFUL."


cleanup()
torch.cuda.empty_cache()

# Define the prompt
query_str = "Describe the first computers people used for programming, the languages used, and the challenges faced?"
# final_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", """
#          You are a knowledgeable JSON wizard who knows everything about deployments. The below are the JSON descriptions:
#          Json Key descriptions in knowledge base:
#              - query: The question or query being asked.
#              - reference_contexts: Contextual information related to the query.
#              - reference_answer: The answer to the query based on the context.
#          """),
#         ("human", query_str),
#     ]
# )
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
         You are a knowledgeable JSON wizard who knows everything about deployments. The below are the JSON descriptions:
         Json Key descriptions in knowledge base:
             - query: The question or query being asked.
             - reference_contexts: Contextual information related to the query.
             - reference_answer: The answer to the query based on the context.
         Please provide a valid JSON response in the following format:
         {
             "query": "<query>",
             "reference_answer": "<answer>"
         }
         """),
        ("human", "{query_str}"),
    ]
)

# # Create the JSON agent
# json_agent_executor = create_json_agent(
#     llm=ChatOllama(model="llama3.2", temperature=0.5),  # Updated to use ChatOllama
#     toolkit=json_toolkit,
#     prompt=final_prompt,
# )

# Create the JSON agent & Enable error handling for parsing failures
json_agent_executor = create_json_agent(
    llm=ChatOllama(model="llama3.2", temperature=0.5),  # Pass temperature dynamically if needed
    toolkit=json_toolkit,
    prompt=final_prompt,
    handle_parsing_errors=True,  # Enable error handling
)

# Use CUDA profiler
import accelerate
tokens_per_second_results = []
for i in range(10):
    with torch.no_grad(): # for inferencing: Memory Efficiency, Speed Improvement, Safety
        temp = random.rand()
        print(f"\n[INFO] for the {i}-th run, set randomized temperature = ", temp)
        # Execute the agent
        query_str = "Describe the first computers people used for programming, the languages used, and the challenges faced?"
        
        # Create CUDA events to measure time
        # Measure the performance using PyTorch's built-in functions
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True) 
        # Start CUDA profiler
        torch.cuda.profiler.start()
        # Record the start event
        start_event.record()
        # Adjust sampling parameters for efficiency
        output = json_agent_executor.invoke({"input": {query_str}}) # [TODO] find out how to pass `temp` into json_agent_executor
        print('\n[OUTPUT] for query: ', query_str, " the model invoked output = ", output)
        print('\n[INFO] for query: ', query_str, " the model invoked output is of type: ", type(output)) # <class 'dict'>
        print('\n[INFO] model invoked output contains keys: ', output.keys()) # dict_keys(['input', 'output'])
        print('\n[INFO] model invoked output is of dim: ', len(output['output'])) # dim(output['output']) = 265
        
        # Record the end event
        end_event.record()
    
        # Stop CUDA profiler
        torch.cuda.profiler.stop()
        
        # Synchronize events
        torch.cuda.synchronize()
        
        # Calculate the time taken in milliseconds
        time_taken_ms = start_event.elapsed_time(end_event)
        time_taken_s = time_taken_ms / 1000 # Convert milliseconds to seconds
        # Calculate tokens per second
        num_tokens = len(output['output'])
        tokens_per_second = num_tokens / time_taken_s
        tokens_per_second_results.append(tokens_per_second)
        
        # Print the results
        print(f"\n[INFO] Tokens generated: {num_tokens}")
        print(f"\n[INFO] Time taken: {time_taken_s:.4f} seconds")
        print(f"\n[INFO] Tokens per second: {tokens_per_second:.2f}")
        
avg_tokens_per_second_results = sum(tokens_per_second_results) / len(tokens_per_second_results)
print(f"\n[INFO][FINAL] Average tokens per second: {avg_tokens_per_second_results:.2f}")

cleanup()


# Add a retriever for RAG functionality
from langchain.schema import BaseRetriever  # Ensure BaseRetriever is imported
from langchain.chains import RetrievalQA  # Ensure RetrievalQA is imported

class SimpleRetriever(BaseRetriever):
    def get_relevant_documents(self, query):
        # Retrieve documents based on the query
        return [
            {"query": query, "reference_answer": "This is a mock answer for testing."}
        ]

retriever = SimpleRetriever()

# Create a RetrievalQA chain for RAG
rag_chain = RetrievalQA.from_chain_type(
    llm=ChatOllama(model="llama3.2", temperature=0.1),
    retriever=retriever,
    return_source_documents=True,
)

# Create the JSON agent
json_agent_executor = create_json_agent(
    llm=ChatOllama(model="llama3.2", temperature=0.1),  # Updated to use ChatOllama
    toolkit=json_toolkit,
    prompt=final_prompt,
)

# Execute the agent
output = json_agent_executor.invoke(
    {"input": "Describe the first computers people used for programming, the language was used, and the challenges faced?"}
)
print(output)