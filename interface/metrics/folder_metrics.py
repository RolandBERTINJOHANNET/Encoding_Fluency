import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../core/")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../core/model/")))
import data
import torch
import model
import json
from torch.utils.data import DataLoader
from metrics import MetricExtractor
import matplotlib.pyplot as plt

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

"""# Process each image in the dataset
all_metrics = {}
for image_path in dataset.file_paths:
    image_metrics = metric_extractor(image_path)
    all_metrics[image_path] = image_metrics

# Convert the dictionary to a pandas DataFrame
df = pd.DataFrame.from_dict(all_metrics, orient='index').apply(lambda x : x.apply(lambda x : float(x)))
"""
file_path = input("Enter an output folder : ")# Ask for output location
"""
os.makedirs(file_path,exist_ok=True)
df.to_csv(f"{file_path}/metrics.csv")# Save csv to file
training.plot_inNout(dataloader, model, file_path, model_name)# Plot in-and-out images
"""
#--------------now make a histograms plot
# Create a figure for the plot
num_layers = len(model.layers + model.attention_layers)
num_images = 5
fig, axs = plt.subplots(num_images, num_layers+1, figsize=(5*num_layers, 5*num_images))
# Process each image in the dataset
for i, image_path in enumerate(dataset.file_paths[:num_images]):
    image_metrics = metric_extractor(image_path)#setup the metrics extractor
    print(metric_extractor.image.shape,metric_extractor.image.device)
    # Plot the original image
    image = data.OptionalSplitDataset.process_image(image_path, model.device)[None,:].cpu().numpy()
    axs[i, 0].imshow(image[0].transpose(1, 2, 0))
    axs[i, 0].axis('off')

    # Plot the activation histograms for each layer
    for j, layer in enumerate(model.all_layers):
        activations = metric_extractor.get_activations(layer).cpu().numpy()
        axs[i, j+1].hist(activations.flatten(), bins=50)
        axs[i, j+1].set_title(layer)

# Save the plot to a file
plt.savefig(f"{file_path}/activation_histograms.png")
plt.close(fig)