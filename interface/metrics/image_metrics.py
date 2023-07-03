import sys
sys.path.append("../../core/","../../core/model/")
import metrics
import torch
import model
import json
import pandas as pd

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
model_instance.load_state_dict(torch.load("../models/weights/"+model_name))# Load weights

image_path = input("Enter an image path: ")# Ask for image path

image_metrics = metrics.get_metrics_image(image_path, model_instance)# Get the metrics

df = pd.DataFrame(image_metrics, index=[image_path])# Convert the metrics to a pandas DataFrame

file_path = input("Enter an output path: ")# Ask for output location
df.to_csv(file_path)# Save to file