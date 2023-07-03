import sys
sys.path.append("../../core/","../../core/model/")
import metrics
import torch
import model
import json

#ask image path : 
image_path = input("Enter an image path: ")
#ask model path
model_name = input("Enter a model name : ")
#load model parameters
params = json.load(open("../models/parameters/"+model_name))
#create model instance
model = model.Model(
    torch.device("cpu"),
    layer_sparsity_cstraint=params["constraint"],
    attention=params["attention"],
    sparsity_coeff=0.,sparsity_param=0.0
    )
#load weights
model.load_state_dict(torch.load("../models/weights/"+model_name))
#get the metrics
metrics.get_metrics_image(image_path,model)
#ask output location
file_path = input("Enter an output path: ")
#save to file
with open(file_path, 'w') as file:
    json.dump(metrics, file, indent=4)