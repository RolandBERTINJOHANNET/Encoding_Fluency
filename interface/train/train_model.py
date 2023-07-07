import json
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch
import torch.optim as optim
import lpips
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from core.data import OptionalSplitDataset
from core.model.model import Model
from core.training import train_the_model
import datetime
import warnings
warnings.filterwarnings("ignore", category=UserWarning)#lpips is old and has deprecation warnings

# Prompt for path to JSON file containing model parameters
params_path = input('Enter the path to the JSON file containing the model parameters: ')
# Load parameters from JSON file
with open(params_path, 'r') as f:
    params = json.load(f)

# Create a new directory with the model name if it doesn't exist
model_name = params['model_name']
for path in [f'{model_name}/plots',f'{model_name}/parameters',f'{model_name}/weights']:
    os.makedirs(path, exist_ok=True)


# Get device object based on user requirement
device = torch.device("cuda:0") if params['model_params']["device"]=="cuda" and torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Create the dataset & dataloader
dataset = OptionalSplitDataset(params['dataset_folder'], params['split'], device)
data_loader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True)

# Create the model, optimizer and loss function and send to device
model = Model(model_name=model_name,**params['model_params']).to(device)
optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
loss_fn = lpips.LPIPS(net='alex').to(device)

# Train the model
reconstruction_loss, kl_divergence, attention = train_the_model(model, data_loader, loss_fn, optimizer, params['num_epochs'])

# Save the model parameters and weights
with open(f'{model_name}/parameters/{model_name}.json', 'w') as f:
    json.dump(params, f)
torch.save(model.state_dict(), f'{model_name}/weights/{model_name}.pth')

# Plot and save the losses
plt.figure()
plt.plot(reconstruction_loss)
plt.title('Reconstruction Loss')
plt.savefig(f'{model_name}/plots/reconstructions_{model_name}.png')

plt.figure()
plt.plot(kl_divergence)
plt.title('KL Divergence')
plt.savefig(f'{model_name}/plots/kl_div_{model_name}.png')

plt.figure()
plt.plot(attention)
plt.title('Attention map L1 norm')
plt.savefig(f'{model_name}/plots/attention_{model_name}.png')

# Create a log file with some info
date_time = datetime.datetime.now().strftime("%Y_%B_%d_%H:%M:%S")
with open(f'{model_name}/{date_time}.log', 'w') as f:
    f.write(f'Saved model parameters to: {model_name}/parameters/{model_name}.json\n')
    f.write(f'Saved model weights to: {model_name}/weights/{model_name}.pth\n')
    f.write(f'Saved reconstruction loss plot to: {model_name}/plots/reconstructions_{model_name}.png\n')
    f.write(f'Saved KL divergence plot to: {model_name}/plots/kl_div_{model_name}.png\n')
    f.write(f'Saved attention plot to: {model_name}/plots/attention_{model_name}.png\n')