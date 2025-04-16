from pathlib import Path
import pandas as pd
import torch
import numpy
from numpy import random

from torch.profiler import profile, record_function, ProfilerActivity

import json
import sys
sys.path.append('/home/ailab/files/llama_index')
import llama_index
import llama_index.core
import llama_index.core.base.response.schema as response_schema
import llama_index.readers.file as file_readers
from llama_index.core import Document

from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    get_leaf_nodes,
    get_root_nodes,
)
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import AutoMergingRetriever

from transformers import AutoTokenizer, AutoModelForCausalLM

from vllm import LLM  # Use VLLM instead of ChatOllama
from vllm import SamplingParams # for offline batch inference

from llama_index.core.indices.list import ListIndex

import langchain
from langchain_core.runnables.base import RunnableSerializable
# from llama_index.langchain import chat_models
from langchain import chat_models
from typing import Union


# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load TXT document using file_readers
file_loader = file_readers.RTFReader()
txt_path = Path("/home/ailab/files/llama_index/data/llama2_paper/source_files/source.txt")
docs0 = file_loader.load_data(input_file=txt_path)
doc_text = "\n\n".join([d.get_content() for d in docs0])
docs = [Document(text=doc_text)]
print(f"\n[INFO] var docs is of type: ", type(docs))
print(f"\n[INFO] txt file read INTO: docs, where 'doc_text' containing content = \n", doc_text)

jsn_path = Path("/home/ailab/files/llama_index/data/yjoonjang-squad_v2_ragsoluted.json")
with open(jsn_path, 'r') as file:
    jsn_file = json.load(file)
# Normalize JSON data
df = pd.json_normalize(jsn_file)
print(f"\n[DEUBG] var df is of type: ", type(df))

# Save DataFrame to CSV
csv_path = jsn_path.with_suffix('.csv')
df.to_csv(csv_path, index=False)
print(f"\n[INFO] JSON file converted to CSV and saved at {csv_path}")

pd_csv_file_loader = file_readers.PandasCSVReader()
print(f"\n[INFO] Pandas CSV read into DataFrame {df}")
print(f"\n[INFO] df var currently of type: ", type(df))

docs1 = pd_csv_file_loader.load_data(file=csv_path)
docs1_text = "\n\n".join([d.get_content() for d in docs1]) # type: str
docs_1 = [Document(text=docs1_text)]
docs.append(docs_1) # docs = [docs[0], docs[1]] = [Document(text=doc_text), Document(text=doc1_text)]
docs_attr_dct = {"_id",""}

# Convert DataFrame to Document
doc1_text = df.to_string(index=False)
# Concatenate doc_text and doc1_text
txt_csv_text = doc_text + "\n\n" + doc1_text
docss = [Document(text=txt_csv_text)]

# Parse nodes from documents
node_parser = HierarchicalNodeParser.from_defaults()
nodes = node_parser.get_nodes_from_documents(docss)
print("\n[INFO] Text+CSV HierarchicalNodeParser: 'nodes' length =", len(nodes))

# Get leaf and root nodes
leaf_nodes = get_leaf_nodes(nodes)
print("\n[INFO] No. of 'leaf' nodes that don't have children of their own =", len(leaf_nodes))
root_nodes = get_root_nodes(nodes)

# LLM batch offline inference
generation_config="/home/ailab/files/Phi-4-mini-instruct-bnb-4bit/config.json" 
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.5, top_p=0.95)
print("\n[INFO] From VLLM: sampling_params is of type ", type(sampling_params))
print("\n[INFO] From VLLM: sampling_params will be used for batch offline inference, where temperature=0.8, top_p=0.95")

# Free up memory after adding nodes to the docstore
import gc

# Define storage context
docstore = SimpleDocumentStore()
docstore.add_documents(nodes)
storage_context = StorageContext.from_defaults(docstore=docstore)
# Build ListIndex
base_index = ListIndex.from_documents(documents=docss)
print("[INFO] ListIndex `base_index` built successfully.")
# Delete unused variables
del nodes, docss, doc_text, doc1_text, txt_csv_text, leaf_nodes, root_nodes

# Run garbage collection to free system memory
gc.collect()
import torch.distributed as dist
def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()
    return "\n[INFO] torch.distributed.destroy_process_group() cleanup SUCCESSFUL."


cleanup()
torch.cuda.empty_cache()


import accelerate
# Load the tokenizer and model
model_path = "/home/ailab/files/Phi-4-mini-instruct-bnb-4bit"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)

# Move the model to the GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
 
# Step 1: Tokenize the input & Encode the tokens
query_str = "Who's Mark Zuckerberg? Describe his role in Meta in 500 words."
# print(f"\n[DEBUG] for query: {query_str}, inputs = ", tokenizer.encode(query_str, skip_special_tokens=True))
print(f"\n[INFO] for query: {query_str}, tokenized inputs = ", tokenizer.encode(query_str))

inputs = tokenizer.encode(query_str, return_tensors="pt").to(device)
print(f"\n[INFO] for {query_str}, tokenized inputs = ", inputs) # [NOTE] type(inputs) = tensor

# Adjust sampling parameters for efficiency
outputs = model.generate(inputs, max_length=1512, temperature=0.3, top_p=0.9, do_sample=True)
model_resp = tokenizer.decode(outputs[0])
print(f"\n[INFO] for query: {query_str}, the model {model.type} response with decoded outputs: ", model_resp) # type(outputs) = tensor

encoded_input_tensors = tokenizer.encode(text=query_str, return_tensors='pt').to(device)
"""
tensor([[128000,  15546,    596,   4488,  67034,     30,  61885,    813,   3560,
            304,  16197,    304,    220,   1135,   4339,     13]],
       device='cuda:0')
"""
print(f"\n[INFO] for query: {query_str}, the encoded_input_tensors = ", encoded_input_tensors)

cleanup()
torch.cuda.empty_cache()


tokens_per_second_results = []
for i in range(10):
    with torch.no_grad(): # for inferencing: Memory Efficiency, Speed Improvement, Safety
        temp = random.rand()
        print(f"\n[INFO] for the {i}-th run, set randomized temperature = ", temp)
        # Define your chat template
        chat_template = {
            "role": "system",
            "content": "You are a helpful, conversational, and professional assistant."
        }
        query_str = "Who's Manny Pacquio? Describe his career achievements in boxing in 500 words."
        chat_messages = [
            {"role": "user", "content": query_str}
        ]
        inputs = tokenizer(text=query_str, return_tensors="pt").to(device)
        print(f"\n[INFO] for {query_str}, tokenized inputs tensors = ", inputs) # [NOTE] type(inputs) = tensor    
        # Create CUDA events to measure time
        # Measure the performance using PyTorch's built-in functions
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True) 
        # Start CUDA profiler
        torch.cuda.profiler.start()
        # Record the start event
        start_event.record()
        # Adjust sampling parameters for efficiency
        outputs = model.generate(**inputs, max_length=5436, temperature=temp, top_p=0.9, do_sample=True)
        print(f"\n[INFO] for query: {query_str}, the model {model.type} response with outputs: (tensor)", outputs) # type(outputs) = tensor
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
        num_tokens = outputs.shape[1]
        tokens_per_second = num_tokens / time_taken_s
        tokens_per_second_results.append(tokens_per_second)
        
        # Print the results
        print(f"\n[INFO] Tokens generated: {num_tokens}")
        print(f"\n[INFO] Time taken: {time_taken_s:.4f} seconds")
        print(f"\n[INFO] Tokens per second: {tokens_per_second:.2f}")
            
        # decoded_outputs = tokenizer.decode(outputs[0], skip_special_tokens=True) # type(decoded_outputs) = <class 'str'>
        # print(f"\n[INFO] tokenizer decoded outputs: (text)", decoded_outputs)
        # inputs_0 = tokenizer(text=decoded_outputs, return_tensors="pt").to(device)
        # outputs_0 = model.generate(**inputs_0, max_new_tokens=8872, temperature=0.7, top_p=0.9, do_sample=True)
        # print(f"\n[INFO] for query: {inputs_0}, the llm response was: ", tokenizer.decode(outputs_0[0], skip_special_tokens=True))
        # Ensure cleanup is called at the end
        torch.cuda.empty_cache()
        cleanup()


avg_tokens_per_second_results = sum(tokens_per_second_results) / len(tokens_per_second_results)
print(f"\n[INFO][FINAL] Average tokens per second: {avg_tokens_per_second_results:.2f}")
# Clear unused GPU memory
torch.cuda.empty_cache()
print("[INFO] System and GPU memory cleared after answering queries.")


# # Use AutoMergingRetriever
# base_retriever = base_index.as_retriever(similarity_top_k=6, embed_model=llm.embed)
# response = model.add_request(prompt=query_str, request_id='678910', params=sampling_params)
# # response_chat = llm.chat({"input": query_str})
# chat_messages = [
#     {"role": "user", "content": query_str}
# ]
# response_chat = llm.generate(query_str)
# print("\n[INFO] LLM chat response:", response_chat)

# from llama_index.core.response.notebook_utils import display_source_node
# from IPython.core.display import Markdown
# from IPython.core import display

# # Query the retriever
# query_str = (
#     "What could be the potential outcomes of adjusting the amount of safety"
#     " data used in the RLHF stage?"
# )
# nodes = base_retriever.retrieve(query_str)
# base_nodes = base_retriever.retrieve(query_str)
# print("\n[INFO] length of retriever nodes =", len(nodes))
# print("\n[INFO] length of base_retriever nodes =", len(base_nodes))
# for node in base_nodes:
#     print("\n[INFO] Retrieved node content:")
#     print(node.node.get_content())
#     print(display_source_node(node, source_length=10000))
#     print("\n[INFO] Node model card:")
#     version = node.node.metadata.get("version", "Not specified")
#     model_card_text = (
#         f"Model Card Details:\n"
#         f"-------------------\n"
#         f"Model Schema: {node.node.metadata}\n"
#         f"VLLM Config: {model.llm_engine.vllm_config}\n"
#         f"Other Metadata Info: {node.node.metadata.keys()}\n"
#     )
#     print(model_card_text)


# cleanup()

# # import langchain_core.runnables.base as langchain_base

# # from langchain_core.runnables.base import RunnableSerializable, RunnableParallel
# # import langchain_core.tools.simple as simple_tools
# # from langchain_core.tools.base import (
# #     ArgsSchema,
# #     BaseTool,
# #     ToolException,
# #     _get_runnable_config_param,
# # )
# # from langchain_core.tools import Tool
# # from langchain_core.messages import ToolCall
# # query_eng_tool = llama_index.core.tools.QueryEngineTool(query_engine=llm_engine, metadata=base_nodes[:])

# # from typing import Any, Dict, List, Union

# # from pydantic import BaseModel, Field
# # from langchain_core.runnables import RunnableLambda
# # def f(x: Dict[str, Any]) -> str:
# #     return str(x["a"] * max(x["b"]))
# # class FSchema(BaseModel):
# #     """Apply a function to an integer and list of integers."""
# #     a: int = Field(..., description="Integer")
# #     b: List[int] = Field(..., description="List of ints")
    
# # runnable = RunnableLambda(f)
# # as_tool = runnable.as_tool(FSchema)
# # as_tool.invoke({"a": 3, "b": [1, 2]})
# # run_dct_key = runnable.__dict__.keys
# # print(f"\n[DEBUG] runnable instance keys = ", {run_dct_key})

# # # Define the input and output types
# # InputType = Union[str, Dict[str, Any]]
# # OutputType = Dict[str, Any]

# # MyRunnable = RunnableSerializable[InputType, OutputType]
# # print("\n[INFO] MyRunnable model class: ", MyRunnable.get_name(MyRunnable)) # 'ModelMetaclass'

# # # Define a simple tool function
# # def simple_tool_function(data: Dict[str, Any]) -> Dict[str, Any]:
# #     if isinstance(data, dict) and "input" in data:
# #         return {"result": data["input"] * 2}
# #     else:
# #         raise ValueError("Input data must be a dictionary with an 'input' key.")

# # # Convert the tool function into a Tool using BaseTool
# # class SimpleMathTool(BaseTool):
# #     name: str = "SimpleMathTool"
# #     description: str = "A tool that doubles the input."

# #     def _run(self, input: Dict[str, Any]) -> Dict[str, Any]:
# #        return simple_tool_function(input)
# # simple_tool = SimpleMathTool()


# # # Create a subclass of RunnableSerializable and implement the invoke method
# # class MyRunnable(RunnableSerializable):
# #     def invoke(self, input: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
# #         print(f"\n[DEBUG] simple_tool.invoke(input) = ", simple_tool.invoke(input))
# #         return simple_tool.invoke(input)

# # # Create an instance of MyRunnable
# # runnable_instance = MyRunnable()
# # print(f"\n[DEBUG] RunnableSerializable instance = {runnable_instance.__dict__.keys}")

# # # Manually chain tools
# # def chain_tools(input_data: Dict[str, Any]) -> Dict[str, Any]:
# #     if not isinstance(input_data, dict) or "input" not in input_data:
# #         raise ValueError("Input data must be a dictionary with an 'input' key.")
    
# #     intermediate_result = simple_tool.invoke(input_data)
    
# #     if not isinstance(intermediate_result, dict) or "result" not in intermediate_result:
# #         raise ValueError("Intermediate result must be a dictionary with a 'result' key.")
        
# #     final_result = simple_tool.invoke({"input": intermediate_result["result"]})
# #     return final_result
# # # TODO: if above '.invoke(...)' fn failed, TRY: '.ainvoke' ('a'='async'?)

# # # Example usage of manually chaining tools
# # input_data = {"input": 5}
# # result = chain_tools(input_data)
# # print(f"\n[DEBUG] Tool Chain result: ", result)

# # # Call cleanup at the end of your script
# # cleanup()
