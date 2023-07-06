#this file contains all the functions relevant to metrics extraction and computation
#note that the code is far from optimized. It is ridiculusly inefficient, and could be improved at the cost of clarity,
#but the objective was to maintain readability and simplicity as metrics extraction is usually done on
#a couple hundred images and thus is not very expensive.
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../core/")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../core/model/")))
import data
import torch
import training
import lpips
import copy
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

def get_metrics_image(image_path,model):
    """Extract and return all the metrics from the processing of an image at a certain path"""
    #this way of organizing code makes us re-load the models for every image, but it makes simpler code
    #and we won't be extracting metrics on thousands upon thousands of images (which would be pretty long)

    metrics = {}

    #read and normalize image + add batch dimension
    image = data.OptionalSplitDataset.process_image(image_path, model.device)[None,:]
    print(f"image.shape : {image.shape}")

    #accumulate all metrics dictionnaries into metrics_dico_list -- at the end, put them all together into a single dico
    #get gini, l1, kurtosis and the attention featuremaps L1 norms
    metrics_dico_list = [get_gini(image, model),
                         get_L1(image, model),
                         get_kurtosis(image, model),
                         get_attention(image, model)]

    
    #get the 2 lpips reconstruction errors, the L2 and SSIM errors
    loss_functions = {"LPIPS_notune":lpips.LPIPS(net='alex',lpips=False).to(model.device),
              "LPIPS_tuned":lpips.LPIPS(net='alex',lpips=True).to(model.device),
              "SSIM":SSIM().to(model.device),
              "L2":torch.nn.MSELoss()}
    #for each loss function:
    for loss_name,loss in loss_functions.items():
        metrics_dico_list.append(get_reco_error(image, model, loss, loss_name))
        #get the sharpness metric
        metrics_dico_list.append(get_SAM_delta(image, model,loss))
    
    #put all the metrics in a single dir
    for metric in metrics_dico_list:
        metrics.update(metric)
    
    return metrics


def get_L1(image,model):
    """for a given model, computes the L1 norm of activations on all layers"""
    result = {}
    for layer in model.layers:
        with torch.no_grad():
            #get the activations for given layer
            activations = model.get_activations(image,layer)

            activations = activations.flatten(start_dim=1)#reorder activations so that you have a list of activations for each image
            
            result["L1_"+layer]=torch.sum(torch.abs(activations)).cpu()
    return result

def get_gini(image,model):
    result = {}
    for layer in model.layers:
        with torch.no_grad():
            activations = model.get_activations(image,layer)
            #reorder activations into a 1D tensor
            activations = activations.flatten(start_dim=1)
            #extract for the given layer and take away the min to have no negative values
            activations = activations - (torch.min(activations,dim=1)[0].reshape(len(activations),1))

            
            #compute gini
            n = activations.shape[1]#length for each image
            activations = torch.sort(activations.cpu(),dim=1)[0].to(torch.device("cuda:0"))#sort the list for each image (on the cpu to prevent oom)
            activations = torch.cumsum(activations,dim=1)
            #(the last gini step is in this line too)
            result["Gini_"+layer]= ((n + 1 - 2 * activations.sum(dim=1) / activations[:,-1]).cpu() / n)
    return result

def get_kurtosis(image,model):
    result = {}
    for layer in model.layers:
        with torch.no_grad():
            #extract for the given layer and take away the min to have no negative values
            activations = model.get_activations(image,layer)
            #reorder activations into a 1D tensor
            activations = activations.flatten(start_dim=1)
            #compute kurtosis
            result["kurtosis_"+layer]= (((activations - activations.mean())**4).mean() / (activations.std()**4)).cpu()
    return result

def get_attention(image,model):
    result = {}
    for layer in model.attention_layers:
        with torch.no_grad():
            #extract for the given layer
            activations = model.get_activations(image,layer)
            #reorder activations into a 1D tensor
            activations = activations.flatten(start_dim=1)
            #compute kurtosis
            result["attention_"+layer]=torch.sum(torch.abs(activations)).cpu()
    return result

def get_reco_error(image,model,loss_fun,loss_fun_name):
    #predict
    with torch.no_grad():
        prediction,_,_ = model(image)
        loss = loss_fun(prediction,image).flatten()#compute loss
    return {f"{loss_fun_name}":loss.detach().cpu()}

def get_SAM_delta(image,model,loss_fun):
    #model_copy = copy.deepcopy(model)#keep copy to revert to former version at the end

    #define the SAM optimizer
    opt = training.SAM(model.parameters(), torch.optim.Adam)
    #predict
    prediction,_,_ = model(image)
    loss = loss_fun(prediction,image)
    loss.backward()
    opt.first_step(zero_grad=True)#step towards local maximum
    prediction_max,_,_ = model(image)#predict from local max
    loss_max = loss_fun(prediction_max,image)#compute loss from local max

    #revert to former version
    #model = model_copy
    return {"SAM":(loss_max-loss).detach().cpu()}