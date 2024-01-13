from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

import autograd.numpy as np

from autograd.extend import primitive, defvjp

from seldonian.models.pytorch_model import SupervisedPytorchBaseModel

@primitive
def predict_primitive(theta, X, model, **kwargs):
    """Do a forward pass through the PyTorch model.
    Must convert back to numpy array before returning.

    :param theta: model weights
    :type theta: numpy ndarray
    :param X: model features
    :type X: numpy ndarray

    :param model: An instance of a class inheriting from
            SupervisedPytorchBaseModel

    :return pred_numpy: model predictions (logits for the positive class)
    :rtype pred_numpy: numpy ndarray same shape as labels
    """
    # First update model weights
    if not model.params_updated:
        model.update_model_params(theta, **kwargs)
        model.params_updated = True
    # Do the forward pass, need to get a tensor back
    # if X is a list of lists, get indices, flatten, pass through
    # then reconstruct afterward
    nested = False
    if isinstance(X[0],list):
        nested=True
        sublist_lengths = [len(x) for x in X]
        X = [item for sublist in X for item in sublist]
    pred = model.forward_pass(X, **kwargs) # will be flat
    # set the predictions attribute of the model

    model.predictions = pred # needs to be a tensor

    # convert to numpy array
    pred_numpy = pred.cpu().detach().numpy()
    if not nested: 
        return pred_numpy
    
    # Need to reconstruct to original nestedness 
    start = 0
    nested_pred_numpy = []
    for n in sublist_lengths:
        end = start+n
        nested_pred_numpy.append(pred_numpy[start:end])
        start = end
    return nested_pred_numpy


def predict_vjp_primitive(ans, theta, X, model):
    """Do a backward pass through the PyTorch model,
    obtaining the Jacobian d pred / dtheta.
    Must convert back to numpy array before returning

    :param ans: The result from the forward pass
    :type ans: numpy ndarray
    :param theta: model weights
    :type theta: numpy ndarray
    :param X: model features
    :type X: numpy ndarray

    :param model: An instance of a class inheriting from
            SupervisedPytorchBaseModel

    :return fn: A function representing the vector Jacobian operator
    """
    local_predictions = model.predictions

    def fn(v):
        # For the primary objective: 
        # v is a vector with length primary_batch_size


        # For the constraint: 
        # v is a list of vectors with length batch_size, where the size of each vector
        # is given by the number of sentences in that set.
        # Return a 2D array [dF_i/dtheta[0],dF_i/dtheta[1],dF_i/dtheta[2]],[]
        # where i is the data row index
        if isinstance(v[0],float):
            external_grad = torch.from_numpy(v).float().to(model.device)
        else:
            # flatten external grad
            #[item for sublist in nest_list for item in sublist]
            # v_i = sublist
            # v = 
            flattened_v = np.array([item for v_i in v for item in v_i])
            external_grad = torch.from_numpy(flattened_v).float().to(model.device)
        dpred_dtheta = model.backward_pass(local_predictions, external_grad)
        model.params_updated = False  # resets for the next forward pass
        return dpred_dtheta

    return fn


# Link the predict function with its gradient,
# telling autograd not to look inside either of these functions
defvjp(predict_primitive, predict_vjp_primitive)

class RobertaHateSpeechModel(SupervisedPytorchBaseModel):
    def __init__(self, device):
        """Wrapper to Facebook's "facebook/roberta-hate-speech-dynabench-r4-target" 
        hate speech model on HuggingFace.

        :param device: The torch device, e.g.,
                "cuda" (NVIDIA GPU), "cpu" for CPU only,
                "mps" (Mac M1 GPU)
        """
        self.name = "facebook/roberta-hate-speech-dynabench-r4-target" 
        super().__init__(device)
        self.pretrained_weights = self.get_model_params() # used in the primary objective
    
    def get_initial_solution(self,*args):
        # return self.pretrained_weights + np.random.normal(
        #     loc=0,scale=0.1,size=len(self.pretrained_weights)
        # )
        return self.pretrained_weights 

    def create_model(self, **kwargs):
        """Create an instance of the model and tokenizer
        """
        model, self.tokenizer = AutoModelForSequenceClassification.from_pretrained(self.name), AutoTokenizer.from_pretrained(self.name)
        return model
    
    def predict(self, theta, X, **kwargs):
        """Do a forward pass through the PyTorch model.
        Must convert back to numpy array before returning

        :param theta: model weights
        :type theta: numpy ndarray

        :param X: model features
        :type X: numpy ndarray

        :return pred_numpy: model predictions
        :rtype pred_numpy: numpy ndarray same shape as labels
        """
        return predict_primitive(theta, X, self)

    def forward_pass(self, X, **kwargs):
        """Do a forward pass through the PyTorch model and return the
        model outputs (logits). 

        :param X: A list of lists (or arrays) of identical sentences with only pronouns permutated.
            Each sublist can have a different length
        :type X: ragged List(array) 

        :return: List of predicted logits
        :rtype: torch.Tensor
        """
        # predictions = []

        
        inputs = self.tokenizer(X, return_tensors='pt', padding=True, truncation=True).to(self.device)
        logits = self.pytorch_model(**inputs).logits[:,1] 
        return logits

    def backward_pass(self, predictions, external_grad):
        """Do a backward pass through the PyTorch model and return the
        (vector) gradient of the model with respect to theta as a numpy ndarray

        :param predictions: The result of running the forward pass
        :type: ragged List(array)
        :param external_grad: List of gradients of the model with respect to itself, one item for each sentence set
                see: https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#differentiation-in-autograd
                for more details
        :type external_grad: torch.Tensor
        """
        self.zero_gradients()
        predictions.backward(gradient=external_grad, retain_graph=True)

        grad_params_list = []
        for param in self.pytorch_model.parameters():
            if param.requires_grad:
                grad_numpy = param.grad.cpu().numpy()
                grad_params_list.append(grad_numpy.flatten())
        return np.concatenate(grad_params_list)

