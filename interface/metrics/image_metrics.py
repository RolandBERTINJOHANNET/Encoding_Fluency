import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../core/")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../core/model/")))
import metrics
import torch
import model
import json
import pandas as pd

# Load model parameters
model_name = input("Enter a model name : ")
params = json.load(open(f"../train/{model_name}/parameters/{model_name}.json"))

# Create model instance
model = model.Model(model_name=model_name,**params['model_params']).to(torch.device(params['model_params']["device"]))

model.load_state_dict(torch.load(f"../train/{model_name}/weights/{model_name}.pth"))# Load weights

image_path = input("Enter an image path: ")# Ask for image path

image_metrics = metrics.get_metrics_image(image_path, model)# Get the metrics

df = pd.DataFrame(image_metrics, index=[image_path])# Convert the metrics to a pandas DataFrame

file_path = input("Enter an output path: ")# Ask for output location
df.to_csv(file_path)# Save to file