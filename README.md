# vector_by_onnxmodel
accelerate generating vector by using onnx model

# install
```python
conda create -n vo python=3.10
pip install -r requirements.txt
```

# how to use
```python
python generate.py
```

# result(~4x faster)
```python
# you can see the inference time of onnx model is much faster than using sentence_transformers
# used model: https://huggingface.co/amu/tao-8k
OnnxModel Runtime gpu Inference time = 4.52 ms
Sentence Transformer gpu Inference time = 22.19 ms
```
