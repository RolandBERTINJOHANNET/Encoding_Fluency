import sys
sys.path.append("../../core/","../../core/model/")
import metrics
import data
import torch
import model
import json
import pandas as pd
from torch.utils.data import DataLoader

# Load model parameters
model_name = input("Enter a model name : ")
params = json.load(open("../models/parameters/"+model_name))

# Create model instance
model_instance = model.Model(
    torch.device("cpu"),
    layer_sparsity_cstraint=params["constraint"],
    attention=params["attention"],
    sparsity_coeff=0.,sparsity_param=0.0
)
# Load weights
model_instance.load_state_dict(torch.load("../models/weights/"+model_name))

# Ask for the root directory of the images
root_dir = input("Enter the root directory of the images: ")
# Create dataset and dataloader
dataset = data.CustomDataset(root_dir, split="test", device=torch.device("cpu"))
dataloader = DataLoader(dataset, batch_size=1)

# Process each image in the dataset
all_metrics = {}
for i, (image, image_path) in enumerate(zip(dataloader, dataset.file_paths)):
    image_metrics = metrics.get_metrics_image(image, model_instance)
    all_metrics[image_path] = image_metrics

# Convert the dictionary to a pandas DataFrame
df = pd.DataFrame.from_dict(all_metrics, orient='index')

file_path = input("Enter an output path: ")# Ask for output location
df.to_csv(file_path)# Save to file