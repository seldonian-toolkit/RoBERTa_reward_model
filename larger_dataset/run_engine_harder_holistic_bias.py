# createSpec.py
import autograd.numpy as np
import os
import pandas as pd
from seldonian.parse_tree.parse_tree import ParseTree
from seldonian.dataset import CustomDataSet, CustomMetaData
from seldonian.utils.io_utils import (load_json,load_pickle,save_pickle)
from seldonian.utils.stats_utils import stability_const,stddev
from seldonian.spec import Spec
from seldonian.seldonian_algorithm import SeldonianAlgorithm

from pytorch_toxicity import RobertaHateSpeechModel

from torch import device

# Our primary objective
def cross_entropy(model,theta,data,**kwargs):
    """Calculate cross entropy from a batch of data

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param data: List of [text,label] pairs

    :return: vector of mean squared error values
    :rtype: numpy ndarray(float)
    """
    # Get text from data
    X = [el[0] for el in data]
    Y = np.array([el[1] for el in data])
    logits = model.predict(theta,X) # logits for positive_class
    # Sigmoid to get probabilities for positive class
    Y_pred = 1/(1+np.exp(-logits))
    
    # Add stability constant. This guards against
    # predictions that are 0 or 1, which cause log(Y_pred) or
    # log(1.0-Y_pred) to be nan. If Y==0 and Y_pred == 1,
    # cost will be np.log(1e-15) ~ -34.
    # Similarly if Y==1 and Y_pred == 0.
    # It's a ceiling in the cost function, essentially.
    loss = np.mean(
        -Y * np.log(Y_pred + stability_const)
        - (1.0 - Y) * np.log(1.0 - Y_pred + stability_const)
    )
    return loss

# Our custom measure function 

def vector_probs_variance(model, theta, data, **kwargs):
    """Calculate sample variance of logits. 

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param data: List of string sentences

    :return: vector of mean squared error values
    :rtype: numpy ndarray(float)
    """
    predictions = model.predict(theta, data) # list of lists, each sublist is logits for positive class of each sentence in group
    # for each sublist calculate the sample variance
    # Get the probabilities 
    probs_list = [1/(1+np.exp(-1*x)) for x in predictions]
    sample_vars = np.array([stddev(probs)**2 for probs in probs_list])
    return sample_vars


if __name__ == '__main__':
    regime='custom'
    sub_regime=None
    np.random.seed(42)
    # Load the primary dataset - the dynabench R4 dataset with labels
    data_savename="../dynabench_dataset_4seldo.csv"
    metadata_savename="../dynabench_metadata_4seldo.json"
    metadata_dict = load_json(metadata_savename)
    all_col_names = metadata_dict['all_col_names']

    df_primary = pd.read_csv(data_savename)
    # Only take the training data (there are also dev and test splits)
    df_primary = df_primary.loc[df_primary['split'] == 'train']
    # Shuffle it
    # Now build Seldonian dataset object using only the columns we'll need to evaluate the primary objective
    primary_data = df_primary[['text','label']].values.tolist()
    print(f"Have {len(primary_data)} samples in primary dataset")
    np.random.shuffle(primary_data)
    primary_meta = CustomMetaData(all_col_names=all_col_names)
    primary_dataset = CustomDataSet(data=primary_data, sensitive_attrs=[], num_datapoints=len(primary_data), meta=primary_meta)

    # Use harder variance dataset for addl dataset
    addl_dataset_filename = './holistic_bias_grouped_texts_90percentile_variance_chopped.pkl'
    addl_data  = load_pickle(addl_dataset_filename)
    #rand_indices = np.random.choice(range(len(addl_data)),replace=False,size=len(addl_data))
    print(f"Have {len(addl_data)} sentence groups in addl dataset")
    #print([len(x) for x in chopped_grouped_texts])
    addl_meta = CustomMetaData(
        all_col_names=["text_groups"],
        sensitive_col_names=[]
    )

    addl_dataset = CustomDataSet(
        data=addl_data, 
        sensitive_attrs=[], 
        num_datapoints=len(addl_data), 
        meta=addl_meta)

    # Use Roberta hate speech text model
    torch_device = device("cuda") # update to "cuda" if not on Mac M1
    model = RobertaHateSpeechModel(torch_device)
    primary_batch_size = 12 # this is how many sentences in the dynabench hatespeech dataset we pass through 
    # in a batch

    # Define behavioral constraint
    constraint_str = 'VPROB <= 0.05'
    delta = 0.05
    
    custom_measure_functions = {
        "VPROB": vector_probs_variance
    }
    
    # Create parse tree object
    pt = ParseTree(
        delta=delta, regime=regime, sub_regime=sub_regime, columns=[],
        custom_measure_functions=custom_measure_functions
    )

    # Fill out tree
    pt.build_tree(
        constraint_str=constraint_str
    )

    parse_trees = [pt]

    # Make additional datasets dict
    # a batch is a bunch of sentences, so batch size should be small
    addl_batch_size = 5
    # addl_batch_size = 5
    additional_datasets = {
        pt.constraint_str: {
            "VPROB": {
                "dataset":addl_dataset,
                "batch_size": addl_batch_size
            }
        }
    }

    # Start with pre-trained weights, no perturbation to start
    initial_solution_fn = model.get_initial_solution

    frac_data_in_safety = 0.6
    # Use vanilla Spec object for custom datasets.
    spec = Spec(
        dataset=primary_dataset,
        additional_datasets=additional_datasets,
        model=model,
        parse_trees=parse_trees,
        frac_data_in_safety=frac_data_in_safety,
        primary_objective=cross_entropy,
        initial_solution_fn=initial_solution_fn,
        use_builtin_primary_gradient_fn=False,
        optimization_technique='gradient_descent',
        optimizer='adam',
        optimization_hyperparams={
            'lambda_init'   : np.array([0.01]),
            'alpha_theta'   : 0.00001,
            'alpha_lamb'    : 0.00005,
            'beta_velocity' : 0.9,
            'beta_rmsprop'  : 0.95,
            'use_batches'   : True,
            'batch_size'    : primary_batch_size,
            'n_epochs'      : 3,
            'gradient_library': "autograd",
            'hyper_search'  : None,
            'verbose'       : True,
        },
        batch_size_safety=8
    )
    SA = SeldonianAlgorithm(spec)
    passed_safety,solution = SA.run(debug=True,write_cs_logfile=True)
