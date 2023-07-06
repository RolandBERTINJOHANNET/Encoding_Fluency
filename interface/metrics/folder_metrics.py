import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../core/")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../core/model/")))
import metrics
import data
import torch
import model
import json
import pandas as pd
from torch.utils.data import DataLoader

# Load model parameters
model_name = input("Enter a model name : ")
params = json.load(open(f"../train/{model_name}/parameters/{model_name}.json"))

# Create model instance
model = model.Model(model_name=model_name,**params['model_params']).to(torch.device(params['model_params']["device"]))

model.load_state_dict(torch.load(f"../train/{model_name}/weights/{model_name}.pth"))# Load weights

# Ask for the root directory of the images
root_dir = input("Enter the root directory of the images: ")
# Create dataset and dataloader
dataset = data.OptionalSplitDataset(root_dir, split="none", device=torch.device("cpu"))
dataloader = DataLoader(dataset, batch_size=1)

# Process each image in the dataset
all_metrics = {}
for image_path in dataset.file_paths:
    image_metrics = metrics.get_metrics_image(image_path, model)
    all_metrics[image_path] = image_metrics

# Convert the dictionary to a pandas DataFrame
df = pd.DataFrame.from_dict(all_metrics, orient='index').apply(lambda x : x.apply(lambda x : float(x)))

file_path = input("Enter an output path: ")# Ask for output location
df.to_csv(file_path)# Save to file