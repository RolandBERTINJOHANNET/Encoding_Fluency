import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../core/")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../core/model/")))
import metrics_short
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

# Create MetricExtractor instance and get the metrics
metric_extractor = metrics_short.MetricExtractor(model)
image_metrics = metric_extractor(image_path)

df = pd.DataFrame(image_metrics, index=[image_path])# Convert the metrics to a pandas DataFrame

file_path = input("Enter an output path: ")# Ask for output location
df.to_csv(file_path)# Save to file