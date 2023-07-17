import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../core/")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../core/model/")))
import data
import plotting
import torch
import model
import json
import pandas as pd
from torch.utils.data import DataLoader
from metrics import MetricExtractor
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)#lpips is old and has deprecation warnings

# Load model parameters
model_name = input("Enter a model name : ")
params = json.load(open(f"../train/{model_name}/parameters/{model_name}.json"))

# Create model instance
model = model.Model(model_name=model_name,**params['model_params']).to(torch.device(params['model_params']["device"]))

model.load_state_dict(torch.load(f"../train/{model_name}/weights/{model_name}.pth"))# Load weights

# Create MetricExtractor instance
metric_extractor = MetricExtractor(model)

# Ask for the root directory of the images
root_dir = input("Enter the root directory of the images: ")
# Create dataset and dataloader
dataset = data.OptionalSplitDataset(root_dir, split="none", device=model.device)
dataloader = DataLoader(dataset, batch_size=10)

# Process each image in the dataset
all_metrics = {}
i=0
for image_path in tqdm(dataset.file_paths, total=len(dataset.file_paths), desc="extracting metrics",unit="image"):
    image_metrics = metric_extractor(image_path)
    all_metrics[image_path] = image_metrics
    if i>=5:
        break

# Convert the dictionary to a pandas DataFrame
df = pd.DataFrame.from_dict(all_metrics, orient='index').apply(lambda x : x.apply(lambda x : float(x)))

dir_path = input("Enter an output directory: ")# Ask for output directory
os.makedirs(dir_path,exist_ok=True)

plotting.plot_inNout(dataloader, model, dir_path, model_name)# Plot in-and-out images

#--------------now make a histograms plot
# Call the function to plot activation histograms
plotting.plot_activation_histograms(metric_extractor, dataset, dir_path)

# the remainder of the code splits the metrics by type (L1, reconstruction...) and saves them separately
metric_types = ["Kurtosis", "L1", "Gini", "Mean", "Std", "Attention", "SAM", "Reco"]

# Loop over metric types
for metric_type in metric_types:
    df_metric = df.filter(like=metric_type)# Filter columns related to the current metric type
    
    file_path = f"{dir_path}/{model_name}_{metric_type}.csv"# Define the output filename
    
    df_metric.to_csv(file_path)# write the metrics file
