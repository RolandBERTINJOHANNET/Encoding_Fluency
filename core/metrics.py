#this file contains all the functions relevant to metrics extraction and computation
#note that the code is far from optimized. It is ridiculusly inefficient, and could be improved at the cost of clarity,
#but the objective was to maintain readability and simplicity as metrics extraction is usually done on
#a couple hundred images and thus is not very expensive.

def get_metrics_image(image_path):
    """Extract and return all the metrics from the processing of an image at a certain path"""
    #this way of organizing code makes us re-load the models for every image, but it makes simpler code
    #and we won't be extracting metrics on thousands upon thousands of images (which would be pretty long)

    metrics = {}

    #read and preprocess image

    #get gini, l1, kurtosis
    #metrics["gini_L1"]=gini ......

    #get the 2 lpips reconstruction errors
    #get the L2 and SSIM errors
    #metrics["L2"]=L2 ......

    #get the attention fm L1 norms
    #metrics["attention"]=attention

    #get the sharpness metric
    #metrics["sharpness"]=sharpness

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
            #extract for the given layer and take away the min to have no negative values
            activations = model.get_activations(image,layer)
            activations = activations - (torch.min(activations,dim=1)[0].reshape(len(activations),1))

            #reorder activations into a 1D tensor
            activations = activations.flatten(start_dim=1)
            
            #compute gini
            n = activations.shape[1]#length for each image
            activations = torch.sort(activations.cpu(),dim=1)[0].to(torch.device("cuda:0"))#sort the list for each image (on the cpu to prevent oom)
            activations = torch.cumsum(activations,dim=1)
            #(the last gini step is in this line too)
            result["Gini_"+layer]= torch.sum(torch.abs(activations))((n + 1 - 2 * activations.sum(dim=1) / activations[:,-1]).cpu() / n).cpu()
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
            result["kurtosis_"+layer]=torch.kurtosis(activations).cpu()
    return result

def get_attention(image,model):
    result = {}
    for layer in model.attention_layers:
        with torch.no_grad():
            #extract for the given layer and take away the min to have no negative values
            activations = model.get_activations(image,layer)
            #reorder activations into a 1D tensor
            activations = activations.flatten(start_dim=1)
            #compute kurtosis
            result["attention_"+layer]=torch.sum(torch.abs(activations)).cpu()
    return result

def get_reco_error(image,model,loss_fun,loss_fun_name):
    #predict
    with torch.no_grad():
        prediction,_ = model(image)
        loss = loss_fun(prediction,batch[idx:idx+batch_size])#compute loss
    return {"loss_fun_name":loss.detach().cpu()}

def get_SAM_delta(image,model,loss_fun):
    model_copy = model.clone()#keep copy to revert to former version at the end

    #define the SAM optimizer
    opt = SAM_opt.SAM(model.parameters(), torch.optim.Adam)
    #predict
    prediction,_ = model(image)
    loss = loss_fun(prediction,batch[idx:idx+batch_size])
    loss.backward()
    opt.first_step(zero_grad=True)#step towards local maximum
    prediction,_ = model(batch[idx:idx+batch_size])#predict from local max
    loss_max = loss_fun(prediction,batch[idx:idx+batch_size]).mean()#compute loss from local max

    #revert to former version
    model = model_copy
    return {"SAM":(loss_max-loss).detach().cpu()}