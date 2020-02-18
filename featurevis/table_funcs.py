import torch


def load_ensemble_model(member_table, trained_model_table, key=None):
    """Loads an ensemble model.

    Args:
        member_table: A Datajoint table containing a subset of the trained models in the trained model table.
        trained_model_table: A Datajoint table containing trained models. Must have a method called "load_model" which
            must itself return a PyTorch module.
        key: A dictionary used to restrict the member table.

    Returns:
        A function that has the model's input as parameters and returns the mean output across the individual models
        in the ensemble.
    """

    def ensemble_model(x, *args, **kwargs):
        outputs = [m(x, *args, **kwargs) for m in models]
        mean_output = torch.stack(outputs, dim=0).mean(dim=0)
        return mean_output

    if key:
        query = member_table() & key
    else:
        query = member_table()
    model_keys = query.fetch(as_dict=True)
    dataloaders, models = tuple(list(x) for x in zip(*[trained_model_table().load_model(key=k) for k in model_keys]))
    return dataloaders[0], ensemble_model


def get_output_selected_model(csrf_v1_selector, model, key):
    """Creates a version of the model that has its output selected down to a single uniquely identified neuron.

    Args:
        csrf_v1_selector: A Datajoint table containing information that can be used to map biological neurons to
            artificial neurons in the model's output.
        model: A PyTorch module that can be called with a keyword argument called "data_key". The output of the
            module is expected to be a two dimensional Torch tensor where the first dimension corresponds to the
            batch size and the second to the number of neurons.
        key: A dictionary used to restrict the selector table to one entry.

    Returns:
        A function that takes the model input(s) as parameter(s) and returns the model output corresponding to the
        selected neuron.
    """
    neuron_pos, session_id = (csrf_v1_selector & key).fetch1("neuron_position", "session_id")

    def output_selected_model(x, *args, **kwargs):
        output = model(x, *args, data_key=session_id, **kwargs)
        return output[:, neuron_pos]

    return output_selected_model
