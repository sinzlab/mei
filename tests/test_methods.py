from unittest.mock import Mock

from featurevis import methods


def test_gradient_ascent():
    dataloaders = dict(train=dict(session1=None))
    model = "model"
    config = dict(
        optim_kwargs=None, transform=None, regularization="module.function", gradient_f=None, post_update=None
    )
    import_object = Mock(return_value="imported_function")
    get_dims = Mock(return_value=dict(session1=dict(inputs=(100, 10, 24, 24))))
    get_initial_guess = Mock(return_value="initial_guess")
    ascend = Mock(return_value=("mei", "evaluations", "_"))

    returned = methods.gradient_ascent(
        dataloaders,
        model,
        config,
        import_object=import_object,
        get_dims=get_dims,
        get_initial_guess=get_initial_guess,
        ascend=ascend,
    )

    assert returned == ("mei", "evaluations")
    import_object.assert_called_once_with("module.function")
    get_dims.assert_called_once_with(dict(session1=None))
    get_initial_guess.assert_called_once_with(1, 10, 24, 24)
    ascend.assert_called_once_with(
        "model",
        "initial_guess",
        optim_kwargs=dict(),
        transform=None,
        regularization="imported_function",
        gradient_f=None,
        post_update=None,
    )
