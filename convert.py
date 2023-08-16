'''
for this script to run do install necessary and relevant libraries first
I will be attaching another file containing all the necessary libraries to be imported, along with the enitre code.
'''
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

# Load the exported ONNX configuration
onnx_config = onnx.load(onnx_model_path)

# Save the ONNX configuration as an ONNX model file
onnx.save(onnx_config, "pix2struct_model.onnx")