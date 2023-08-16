'''
do install necessary libraries for this script to run. 
In this script, I show how I made use of ONNXRuntime (ORT) to generate an inference of our onnx model and run it. 
necessary comments are inserted for better understanding '''

from optimum.exporters import TasksManager
from io import BytesIO
import requests
from optimum.exporters.onnx import export
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor, AutoTokenizer
import onnxruntime
import time
from PIL import Image
import numpy as np
from datasets import load_dataset
from pathlib import Path

model_id = "google/pix2struct-docvqa-base"
model = Pix2StructForConditionalGeneration.from_pretrained(model_id)

onnx_model_path = Path("model.onnx")
onnx_config_const = TasksManager.get_exporter_config_constructor("onnx", model, task='visual-question-answering') #importing onnx config constructor from TM of HF
onnx_config = onnx_config_const(model.config)

processor = Pix2StructProcessor.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

image_url= "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.jpg"
# Download and open the image
image_temp = requests.get(image_url)
image_np = Image.open(BytesIO(image_temp.content))

# Giving the question here.
question = "What does the label 15 represent? (1) lava (2) core (3) tunnel (4) ash cloud"

#using pre-existing information to run this configuration and export the necessary onnx model

onnx_inputs, onnx_outputs = export(model, onnx_config, onnx_model_path, onnx_config.DEFAULT_ONNX_OPSET)   #default opset value is 12

ort_sess = onnxruntime.InferenceSession('model.onnx') #initiating an inference session, ort is the most optimum way to infer an onnx model. ort stands for onnxruntime

onnx_time = time.time()

encoded_input = processor(images=image_np, text=question, return_tensors="np")
encoded_text = tokenizer(question, return_tensors="np")

onnx_results = ort_sess.run(output_names=['logits'],
             input_feed={
    'flattened_patches':encoded_input["flattened_patches"],
    'attention_mask':encoded_input["attention_mask"].astype(np.int64),
    'decoder_input_ids': encoded_text["input_ids"].astype(np.int64)
})

print(processor.decode(onnx_results[0][0].argmax(axis=1), skip_special_tokens=True).strip())
print("ONNX Inference Time:", round(time.time() - onnx_time, 3))


