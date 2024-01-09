import autograd.numpy as np
import os
import math
from seldonian.utils.io_utils import (load_json,load_pickle,save_pickle)
from seldonian.utils.stats_utils import stability_const,stddev
from experiments.generate_plots import CustomPlotGenerator
from pytorch_toxicity import RobertaHateSpeechModel
from experiments.experiment_utils import make_batch_epoch_dict_fixedniter
import torch
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

def perf_eval_fn(theta,model,data,**kwargs):
    texts = [el[0] for el in data]
    Y = np.array([el[1] for el in data])
    batch_size = kwargs["eval_batch_size"]
    N_tot = len(Y)
    Y_pred = np.zeros(N_tot)
    n_batches = math.ceil(N_tot / batch_size)
    for i in range(n_batches):
        start = i*batch_size
        end = start + batch_size
        batch_data = texts[start:end]
        logits = model.predict(theta,batch_data)
        probs = 1/(1+np.exp(-logits))
        Y_pred[start:end] = probs

    loss = np.mean(
        -Y * np.log(Y_pred + stability_const)
        - (1.0 - Y) * np.log(1.0 - Y_pred + stability_const)
    )
    return loss

if __name__ == '__main__':
    np.random.seed(42)
    run_experiments=True
    make_plots=False
    save_plot=False
    n_trials = 10
    # data_fracs = np.logspace(-4,0,10)
    data_fracs = np.logspace(-2.5,0,10)[0:1]
    n_workers = 1
    results_dir = f'results/roberta_hate_speech_and_holistic_bias_90thpercentile_10trials'
    plot_savename = os.path.join(results_dir,f'threeplots.png')

    verbose=True

    # Load spec
    specfile="../roberta_hate_speech_and_holistic_bias_90thpercentile_spec.pkl"
    spec = load_pickle(specfile)
    # Make new model instantiation to ensure that model gets put on proper GPU device
    del spec.model
    torch_device = torch.device("cuda:0") # update to "cuda" if not on Mac M1
    model = RobertaHateSpeechModel(torch_device)
    # Remove model from original device and empty cache to free up GPU memory 
    torch.cuda.empty_cache()
    spec.model = model
    os.makedirs(results_dir,exist_ok=True)

    # Define primary ground truth dataset for evaluating performance plot
    test_dataset = spec.dataset

    test_data = test_dataset.data
    # Setup performance evaluation function and kwargs 
    perf_eval_kwargs = {
        'test_data':test_data,
        'eval_batch_size':10
    }

    # Use original additional_datasets as ground truth for evaluating safety plot
    constraint_eval_kwargs = {}
    constraint_eval_kwargs["additional_datasets"] = spec.additional_datasets
    constraint_eval_kwargs["eval_batch_size"] = 10

    # Use a batch epoch dict to ensure small data fracs have enough total iterations
    N_max = int(round(test_dataset.num_datapoints*(1.0-spec.frac_data_in_safety)))
    batch_epoch_dict = make_batch_epoch_dict_fixedniter(
        niter=1000, 
        data_fracs=data_fracs, 
        N_max=N_max, 
        batch_size=spec.optimization_hyperparams['batch_size']
    )
    print("batch_epoch_dict:")
    print(batch_epoch_dict)
    plot_generator = CustomPlotGenerator(
        spec=spec,
        n_trials=n_trials,
        data_fracs=data_fracs,
        n_workers=n_workers,
        datagen_method='resample',
        perf_eval_fn=perf_eval_fn,
        constraint_eval_fns=[],
        constraint_eval_kwargs=constraint_eval_kwargs,
        results_dir=results_dir,
        perf_eval_kwargs=perf_eval_kwargs,
        batch_epoch_dict=batch_epoch_dict
    )

    if run_experiments:
        plot_generator.run_seldonian_experiment(verbose=verbose)

    if make_plots:
        plot_generator.make_plots(
            tot_data_size=test_dataset.num_datapoints,
            fontsize=14,
            legend_fontsize=12,
            performance_label="Cross entropy loss",
            include_legend=False,
            save_format="png",
            savename=plot_savename if save_plot else None)
