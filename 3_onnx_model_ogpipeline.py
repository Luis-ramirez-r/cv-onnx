import time
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import onnxruntime as ort



filename = './images/dog.jpg'

with open('synset.txt', 'r') as f:
    labels = [l.rstrip() for l in f]

input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model


ort_session = ort.InferenceSession("./models/googlenet.onnx",providers=['CPUExecutionProvider'])


input_batch = input_batch.detach().cpu().numpy()

outputs = ort_session.run(
    None,
    {"input": input_batch.astype(np.float32)},
)

a = np.argsort(-outputs[0].flatten())
results = {}
for i in a[0:5]:
    results[labels[i]]=float(outputs[0][0][i])

print (results)

