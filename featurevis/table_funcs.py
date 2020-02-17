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
