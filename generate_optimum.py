import time
import torch 

from tqdm import tqdm
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
from pathlib import Path
from optimum.onnxruntime import ORTOptimizer
from optimum.onnxruntime.configuration import OptimizationConfig
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Pooling

st = ['如果你把智力看得比其他人类品质更重要，那么你会过得很糟糕']
device_name = 'gpu'

model_path= "amu/tao-8k"
onnx_path = Path("./onnx_models")
local_model_path = "./models/tao-8k"
model = SentenceTransformer(model_path)
model.half().save(local_model_path)

# load vanilla transformers and convert to onnx
model = ORTModelForFeatureExtraction.from_pretrained(model_path, export=True, provider='CUDAExecutionProvider', provider_options={'device_id': 0})
tokenizer = AutoTokenizer.from_pretrained(model_path)

# save onnx checkpoint and tokenizer
model.save_pretrained(onnx_path)
tokenizer.save_pretrained(onnx_path)

# optimize
optimizer = ORTOptimizer.from_pretrained(model)
optimization_config = OptimizationConfig(optimization_level=99,
                                         optimize_for_gpu=True,
                                         fp16=True
                                         )

# apply the optimization configuration to the model
optimizer.optimize(
    save_dir=onnx_path,
    optimization_config=optimization_config,
)

# load optimized model
onnx_model = ORTModelForFeatureExtraction.from_pretrained(onnx_path, file_name="model_optimized.onnx", provider='CUDAExecutionProvider', provider_options={'device_id': 0})

# pooling_layer
pool_layer = Pooling(word_embedding_dimension=1024)

# onnx
latency = []
for i in tqdm(range(1000)):
    start = time.perf_counter()
    inputs = tokenizer(
        st,
        padding=True,
        truncation=True,
        max_length=1024,
        return_tensors="pt"
    )
    ort_inputs = {k:v.cpu().numpy() for k, v in inputs.items()}
    # if has token_type_ids delete
    if 'token_type_ids' in ort_inputs:
        del ort_inputs['token_type_ids']
    ort_outputs = onnx_model.model.run(None, ort_inputs)
    ort_result = pool_layer(
        features={
            'token_embeddings': torch.Tensor(ort_outputs[0]),
            'attention_mask': inputs.get('attention_mask')
        })
    result = ort_result.get('sentence_embedding')
    latency.append(time.perf_counter() - start)
    
print("[Optimum] OnnxModel Runtime {} Inference time = {} ms".format(device_name, format(sum(latency) * 1000 / len(latency), '.2f')))

## sentence transformer
latency = []
model = SentenceTransformer(model_path)
for i in tqdm(range(1000)):
    start = time.perf_counter()
    result = model.encode(st)
    latency.append(time.perf_counter() - start)
print("Sentence Transformer {} Inference time = {} ms".format(device_name, format(sum(latency) * 1000 / len(latency), '.2f')))
