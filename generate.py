import os
import torch
import onnx
import onnxruntime
import json
import time

from tqdm import tqdm
from onnxconverter_common.float16 import convert_float_to_float16
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Pooling
from transformers import (AutoConfig, AutoModel, AutoTokenizer)

# generate st model
hf_model_path = "amu/tao-8k"
local_model_path = "./models/tao-8k"
model = SentenceTransformer(hf_model_path)
model.half().save(local_model_path)

# get onnx model
## onnxf32
output_dir = os.path.join(".", "onnx_models")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
export_model_path = os.path.join(output_dir, 'tao_8k_f32.onnx')

config = AutoConfig.from_pretrained(local_model_path)
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModel.from_pretrained(local_model_path, config=config)

st = ['如果你把智力看得比其他人类品质更重要，那么你会过得很糟糕']
inputs = tokenizer(
    st,
    padding=True,
    truncation=True,
    max_length=1024,
    return_tensors="pt"
)

device = torch.device("cpu")
model.eval()
model.to(device)

with torch.no_grad():
    symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
    torch.onnx.export(model,                                            
                    args=tuple(inputs.values()),                      
                    f=export_model_path,                              
                    opset_version=11,                                
                    do_constant_folding=True,                        
                    input_names=['input_ids',                        
                                'attention_mask',
                                'token_type_ids'],
                    output_names=['start', 'end'],                   
                    dynamic_axes={'input_ids': symbolic_names,
                                'attention_mask' : symbolic_names,
                                'token_type_ids' : symbolic_names,
                                'start' : symbolic_names,
                                'end' : symbolic_names})
    print("Model exported at ", export_model_path)

## convert onnxf32 to onnxf16
model = onnx.load("./onnx_models/tao_8k_f32.onnx")
model_fp16 = convert_float_to_float16(model)
onnx.save(model_fp16, "./models/tao-8k/tao_8k_f16.onnx")

## get optimized_model_gpu.onnx
device_name = 'gpu'
sess_options = onnxruntime.SessionOptions()
sess_options.optimized_model_filepath = os.path.join(output_dir, "optimized_model_{}.onnx".format(device_name))
sess_options.intra_op_num_threads = 1

session = onnxruntime.InferenceSession("./models/tao-8k/tao_8k_f16.onnx",
                                       sess_options,
                                       providers=['CUDAExecutionProvider'],
                                       provider_options=[{
                                           'device_id': '0'
                                       }])
pooling_model = Pooling.load("./models/tao-8k/1_Pooling")

# sentence transformer vs. onnx

## onnx
latency = []
for i in tqdm(range(100)):
    start = time.perf_counter()
    inputs = tokenizer(
        st,
        padding=True,
        truncation=True,
        max_length=1024,
        return_tensors="pt"
    )
    ort_inputs = {k:v.cpu().numpy() for k, v in inputs.items()}
    ort_outputs = session.run(None, ort_inputs)
    ort_result = pooling_model.forward(
        features={
            'token_embeddings': torch.Tensor(ort_outputs[0]),
            'attention_mask': inputs.get('attention_mask')
        })
    result = ort_result.get('sentence_embedding')
    latency.append(time.perf_counter() - start)
    
print("OnnxModel Runtime {} Inference time = {} ms".format(device_name, format(sum(latency) * 1000 / len(latency), '.2f')))

## sentence transformer
latency = []
model = SentenceTransformer(local_model_path)
for i in tqdm(range(100)):
    start = time.perf_counter()
    result = model.encode(st)
    latency.append(time.perf_counter() - start)
print("Sentence Transformer {} Inference time = {} ms".format(device_name, format(sum(latency) * 1000 / len(latency), '.2f')))


