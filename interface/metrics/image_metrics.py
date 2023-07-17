import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../core/")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../core/model/")))
import metrics
import torch
import model
import json
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)#lpips is old and has deprecation warnings

# Load model parameters
model_name = input("Enter a model name : ")
params = json.load(open(f"../train/{model_name}/parameters/{model_name}.json"))

# Create model instance
model = model.Model(model_name=model_name,**params['model_params']).to(torch.device(params['model_params']["device"]))

model.load_state_dict(torch.load(f"../train/{model_name}/weights/{model_name}.pth"))# Load weights

image_path = input("Enter an image path: ")# Ask for image path

# Create MetricExtractor instance and get the metrics
metric_extractor = metrics.MetricExtractor(model)
image_metrics = metric_extractor(image_path)

# Convert the metrics to a pandas DataFrame
df = pd.DataFrame(image_metrics, index=[image_path]).apply(lambda x : x.apply(lambda x : float(x)))

dir_path = input("Enter an output directory: ")# Ask for output directory
os.makedirs(dir_path,exist_ok=True)

# the remainder of the code splits the metrics by type (L1, reconstruction...) and saves them separately
metric_types = ["Kurtosis", "L1", "Gini", "Mean", "Std", "Attention", "SAM", "Reco"]

# Loop over metric types
for metric_type in metric_types:
    df_metric = df.filter(like=metric_type)# Filter columns related to the current metric type
    
    file_path = f"{dir_path}/{model_name}_{metric_type}.csv"# Define the output filename
    
    df_metric.to_csv(file_path)# write the metrics file
