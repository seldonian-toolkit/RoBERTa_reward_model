# createSpec.py
import autograd.numpy as np
import os
import pandas as pd
from seldonian.parse_tree.parse_tree import ParseTree
from seldonian.dataset import CustomDataSet, CustomMetaData
from seldonian.utils.io_utils import (load_json,save_pickle)
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

def vector_logits_variance(model, theta, data, **kwargs):
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
    sample_vars = np.array([stddev(sublist)**2 for sublist in predictions])
    return sample_vars


if __name__ == '__main__':
    regime='custom'
    sub_regime=None

    # Load the primary dataset - the dynabench R4 dataset with labels
    data_savename="./dynabench_dataset_4seldo.csv"
    metadata_savename="./dynabench_metadata_4seldo.json"
    metadata_dict = load_json(metadata_savename)
    all_col_names = metadata_dict['all_col_names']

    df_primary = pd.read_csv(data_savename)
    # Only take the training data (there are also dev and test splits)
    df_primary = df_primary.loc[df_primary['split'] == 'train']
    # Shuffle it
    # Now build Seldonian dataset object using only the columns we'll need to evaluate the primary objective
    primary_data = df_primary[['text','label']].values.tolist()
    np.random.shuffle(primary_data)
    N_limit_primary = 100
    primary_data = primary_data[:N_limit_primary]
    primary_meta = CustomMetaData(all_col_names=all_col_names)
    primary_dataset = CustomDataSet(data=primary_data, sensitive_attrs=[], num_datapoints=len(primary_data), meta=primary_meta)

    # Create addl dataset for the constraint
    addl_data_path = "sentences__small_set.csv"    

    df_addl = pd.read_csv(addl_data_path)
    grouped_texts = df_addl.groupby(["template", "descriptor"])['text'].apply(list)


    # limit to first N_limit sentence sets for testing
    # N_limit = len(grouped_texts)
    N_limit = 40
    addl_data = [x for x in grouped_texts[0:N_limit]]

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
    torch_device = device("mps") # update to "cuda" if not on Mac M1
    model = RobertaHateSpeechModel(torch_device)
    # primary_batch_size = 5 # this is how many sentences in the dynabench hatespeech dataset we pass through 
    primary_batch_size = 5 # this is how many sentences in the dynabench hatespeech dataset we pass through 
    # in a batch

    # Define behavioral constraint
    constraint_str = 'VLOG <= 0.05'
    delta = 0.05
    
    custom_measure_functions = {
        "VLOG": vector_logits_variance
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
            "VLOG": {
                "dataset":addl_dataset,
                "batch_size": addl_batch_size
            }
        }
    }

    # Start with pre-trained weights, no perturbation to start
    initial_solution_fn = model.get_initial_solution

    frac_data_in_safety = 0.75
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
            'lambda_init'   : np.array([0.5]),
            'alpha_theta'   : 0.0005,
            'alpha_lamb'    : 0.0005,
            'beta_velocity' : 0.9,
            'beta_rmsprop'  : 0.95,
            'use_batches'   : True,
            'batch_size'    : primary_batch_size,
            'n_epochs'      : 1,
            'gradient_library': "autograd",
            'hyper_search'  : None,
            'verbose'       : True,
        },
        batch_size_safety=5
    )
    SA = SeldonianAlgorithm(spec)
    passed_safety,solution = SA.run(debug=True,write_cs_logfile=True)
