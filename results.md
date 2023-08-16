size of huggingface model : 1103.275 MB
size of ONNX model : 1103.275 MB

ONNX Inference Time: 19.797

Transformers Inference Time: 24.115


**I tried to optimize my onnx model which would have reduced the model size to ~500MB but my college id crashed resulted in me losing the colab notebook in which I was working on. Due to less time, I couldn't redo it, but that's something I'll have in mind as a future scope.

I am also attaching the colab notebook in which I worked. It has the code in it's entirety. 

Necessary Libraries:
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

before installing these, run the following commands,
!pip install optimum[exporters]

!pip install optimum[onnx]

!pip install optimum[onnxruntime]