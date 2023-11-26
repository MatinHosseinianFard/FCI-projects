from reader import MP5Dataset
import numpy as np 
def get_dataset_from_arrays(X,y):
    """This function returns a valid pytorch dataset from feature and label vectors

    Args:
        X ([np.array]): The feature vectors 
        y ([np.array]): The label vectors of the dataset

    Returns:
        [Dataset]: a valid pytorch dataset which you can use with the pytorch dataloaders
    """
    return MP5Dataset(X,y)

def compute_accuracies(predicted_labels, dev_set, dev_labels):
    yhats = predicted_labels
    assert predicted_labels.dtype == np.int64, "Your predicted labels have type {}, but they should have type np.int (consider using .astype(int) on your output)".format(predicted_labels.dtype)

    if len(yhats) != len(dev_labels):
        print("Lengths of predicted labels don't match length of actual labels", len(yhats), len(dev_labels))
        return 0., 0., 0., 0.
    accuracy = np.mean(yhats == dev_labels)
    conf_m = np.zeros((len(np.unique(dev_labels)),len(np.unique(dev_labels))))
    for i,j in zip(dev_labels,predicted_labels):
        conf_m[i,j] +=1

    return accuracy, conf_m

def get_parameter_counts(net):
    """ Get the parameters of your network
    @return params: a list of tensors containing all parameters of the network
            num_params: count of the total number of parameters
    """
    params = net.parameters()
    num_parameters = sum([ np.prod(w.shape) for w  in params])

    return num_parameters,params
