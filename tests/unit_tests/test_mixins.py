from unittest.mock import MagicMock, call
from contextlib import contextmanager
from functools import partial

import pytest

from featurevis import mixins


@contextmanager
def does_not_raise():
    yield


@pytest.fixture
def key():
    return MagicMock(name="key")


class TestTrainedEnsembleModelTemplateMixin:
    @pytest.fixture
    def trained_ensemble_model_template(
        self, dataset_table, trained_model_table, ensemble_model_class, insert1, insert
    ):
        trained_ensemble_model_template = mixins.TrainedEnsembleModelTemplateMixin
        trained_ensemble_model_template.dataset_table = dataset_table
        trained_ensemble_model_template.trained_model_table = trained_model_table
        trained_ensemble_model_template.ensemble_model_class = ensemble_model_class
        trained_ensemble_model_template.insert1 = insert1
        trained_ensemble_model_template.Member.insert = insert
        return trained_ensemble_model_template

    @pytest.fixture
    def dataset_table(self):
        dataset_table = MagicMock()
        dataset_table.return_value.__and__.return_value.__len__.return_value = 1
        dataset_table.return_value.proj.return_value.__and__.return_value.fetch1.return_value = dict(ds=0)
        return dataset_table

    @pytest.fixture
    def trained_model_table(self):
        trained_model_table = MagicMock()
        trained_model_table.return_value.proj.return_value.__and__.return_value.fetch.return_value = [
            dict(m=0),
            dict(m=1),
        ]
        trained_model_table.return_value.__and__.return_value.fetch.return_value = [dict(m=0, a=0), dict(m=1, a=1)]
        trained_model_table.return_value.load_model = MagicMock(
            side_effect=[("dataloaders1", "model1"), ("dataloaders2", "model2")]
        )
        return trained_model_table

    @pytest.fixture()
    def ensemble_model_class(self):
        return MagicMock()

    @pytest.fixture
    def insert1(self):
        return MagicMock()

    @pytest.fixture
    def insert(self):
        return MagicMock()

    @pytest.mark.parametrize(
        "n_datasets,expectation",
        [(0, pytest.raises(ValueError)), (1, does_not_raise()), (2, pytest.raises(ValueError))],
    )
    def test_if_key_correctness_is_checked(
        self, key, trained_ensemble_model_template, dataset_table, n_datasets, expectation
    ):
        dataset_table.return_value.__and__.return_value.__len__.return_value = n_datasets
        with expectation:
            trained_ensemble_model_template().create_ensemble(key)

    def test_if_dataset_key_is_correctly_fetched(self, key, trained_ensemble_model_template, dataset_table):
        trained_ensemble_model_template().create_ensemble(key)
        dataset_table.return_value.proj.return_value.__and__.assert_called_once_with(key)
        dataset_table.return_value.proj.return_value.__and__.return_value.fetch1.assert_called_once_with()

    def test_if_primary_model_keys_are_correctly_fetched(
        self, key, trained_ensemble_model_template, trained_model_table
    ):
        trained_ensemble_model_template().create_ensemble(key)
        trained_model_table.return_value.proj.return_value.__and__.assert_called_once_with(key)
        trained_model_table.return_value.proj.return_value.__and__.return_value.fetch.assert_called_once_with(
            as_dict=True
        )

    def test_if_ensemble_key_is_correctly_inserted(self, key, trained_ensemble_model_template, insert1):
        trained_ensemble_model_template().create_ensemble(key)
        insert1.assert_called_once_with(
            dict(ds=0, ensemble_hash="536072017a2a3501ea8f09fffa51ee61", ensemble_comment="")
        )

    def test_if_member_models_are_correctly_inserted(self, key, trained_ensemble_model_template, insert):
        trained_ensemble_model_template().create_ensemble(key)
        insert.assert_called_once_with(
            [
                dict(ds=0, ensemble_hash="536072017a2a3501ea8f09fffa51ee61", m=0),
                dict(ds=0, ensemble_hash="536072017a2a3501ea8f09fffa51ee61", m=1),
            ]
        )

    def test_if_model_keys_are_correctly_fetched(self, key, trained_ensemble_model_template, trained_model_table):
        trained_ensemble_model_template().load_model(key)
        trained_model_table.return_value.__and__.assert_called_once_with(key)
        trained_model_table.return_value.__and__.return_value.fetch.assert_called_once_with(as_dict=True)

    def test_if_models_are_correctly_loaded(self, key, trained_ensemble_model_template, trained_model_table):
        trained_ensemble_model_template().load_model(key)
        trained_model_table.return_value.load_model.assert_has_calls(
            [call(key=dict(m=0, a=0)), call(key=dict(m=1, a=1))]
        )

    def test_if_ensemble_model_is_correctly_initialized(
        self, key, trained_ensemble_model_template, ensemble_model_class
    ):
        trained_ensemble_model_template().load_model(key)
        ensemble_model_class.assert_called_once_with("model1", "model2")

    def test_if_only_first_dataloader_is_returned(self, key, trained_ensemble_model_template):
        dataloaders, _ = trained_ensemble_model_template().load_model(key)
        assert dataloaders == "dataloaders1"


@pytest.fixture
def model():
    return MagicMock(name="Model")


class TestCSRFV1SelectorTemplateMixin:
    @pytest.fixture
    def selector_template(self, dataset_table, constrained_output_model, insert, magic_and):
        selector_template = mixins.CSRFV1SelectorTemplateMixin
        selector_template.dataset_table = dataset_table
        selector_template.constrained_output_model = constrained_output_model
        selector_template.dataset_fn = "dataset_fn"
        selector_template.insert = insert
        selector_template.__and__ = magic_and
        return selector_template

    @pytest.fixture
    def dataset_table(self):
        dataset_table = MagicMock()
        dataset_table.return_value.__and__.return_value.fetch1.return_value = "dataset_config"
        return dataset_table

    @pytest.fixture
    def constrained_output_model(self):
        return MagicMock(return_value="constrained_output_model")

    @pytest.fixture
    def insert(self):
        return MagicMock()

    @pytest.fixture
    def magic_and(self):
        magic_and = MagicMock()
        magic_and.return_value.fetch1.return_value = "neuron_pos", "session_id"
        return magic_and

    @pytest.fixture
    def get_mappings(self):
        return MagicMock(return_value="mappings")

    def test_if_key_source_is_correct(self, selector_template, dataset_table):
        dataset_table.return_value.__and__.return_value = "key_source"
        assert selector_template()._key_source == "key_source"
        dataset_table.return_value.__and__.assert_called_once_with(dict(dataset_fn="dataset_fn"))

    def test_if_dataset_config_is_correctly_fetched(self, key, selector_template, dataset_table, get_mappings):
        selector_template().make(key, get_mappings=get_mappings)
        dataset_table.return_value.__and__.assert_called_once_with(key)
        dataset_table.return_value.__and__.return_value.fetch1.assert_called_once_with("dataset_config")

    def test_if_get_mappings_is_correctly_called(self, key, selector_template, get_mappings):
        selector_template().make(key, get_mappings=get_mappings)
        get_mappings.assert_called_once_with("dataset_config", key)

    def test_if_mappings_are_correctly_inserted(self, key, selector_template, insert, get_mappings):
        selector_template().make(key, get_mappings=get_mappings)
        insert.assert_called_once_with("mappings")

    def test_if_neuron_position_and_session_id_are_correctly_fetched(self, key, model, selector_template, magic_and):
        selector_template().get_output_selected_model(model, key)
        magic_and.assert_called_once_with(key)
        magic_and.return_value.fetch1.assert_called_once_with("neuron_position", "session_id")

    def test_if_constrained_output_model_is_correctly_initialized(
        self, key, model, selector_template, constrained_output_model
    ):
        selector_template().get_output_selected_model(model, key)
        constrained_output_model.assert_called_once_with(
            model, "neuron_pos", forward_kwargs=dict(data_key="session_id")
        )

    def test_if_output_selected_model_is_correctly_returned(self, key, model, selector_template):
        output_selected_model = selector_template().get_output_selected_model(model, key)
        assert output_selected_model == "constrained_output_model"


class TestMEIMethodMixin:
    @pytest.fixture
    def generate_mei(self, mei_method, dataloaders, model, seed):
        return partial(mei_method().generate_mei, dataloaders, model, dict(key="key"), seed)

    @pytest.fixture
    def mei_method(self, insert1, magic_and, import_func):
        mei_method = mixins.MEIMethodMixin
        mei_method.insert1 = insert1
        mei_method.__and__ = magic_and
        mei_method.import_func = import_func
        return mei_method

    @pytest.fixture
    def dataloaders(self):
        return MagicMock(name="dataloaders")

    @pytest.fixture
    def seed(self):
        return 42

    @pytest.fixture
    def insert1(self):
        return MagicMock()

    @pytest.fixture
    def magic_and(self):
        magic_and = MagicMock()
        magic_and.return_value.fetch1.return_value = "method_fn", "method_config"
        return magic_and

    @pytest.fixture
    def import_func(self, method_fn):
        return MagicMock(return_value=method_fn)

    @pytest.fixture
    def method_fn(self):
        return MagicMock(return_value=("mei", "score", "output"))

    def test_that_method_is_correctly_inserted(self, mei_method, insert1):
        method_config = MagicMock(name="method_config")
        mei_method().add_method("method_fn", method_config)
        insert1.assert_called_once_with(
            dict(method_fn="method_fn", method_hash="d41d8cd98f00b204e9800998ecf8427e", method_config=method_config)
        )

    def test_that_method_is_correctly_fetched(self, generate_mei, magic_and):
        generate_mei()
        magic_and.assert_called_once_with(dict(key="key"))
        magic_and.return_value.fetch1.assert_called_once_with("method_fn", "method_config")

    def test_if_method_function_is_correctly_imported(self, generate_mei, import_func):
        generate_mei()
        import_func.assert_called_once_with("method_fn")

    def test_if_method_function_is_correctly_called(self, generate_mei, model, dataloaders, seed, method_fn):
        generate_mei()
        method_fn.assert_called_once_with(dataloaders, model, "method_config", seed)

    def test_if_returned_mei_entity_is_correct(self, generate_mei):
        mei_entity = generate_mei()
        assert mei_entity == dict(key="key", mei="mei", score="score", output="output")


class TestMEITemplateMixin:
    @pytest.fixture
    def mei_template(
        self, trained_model_table, selector_table, method_table, seed_table, insert1, save, model_loader_class
    ):
        mei_template = mixins.MEITemplateMixin
        mei_template.trained_model_table = trained_model_table
        mei_template.selector_table = selector_table
        mei_template.method_table = method_table
        mei_template.seed_table = seed_table
        mei_template.insert1 = insert1
        mei_template.save = save
        mei_template.model_loader_class = model_loader_class
        get_temp_dir = MagicMock()
        get_temp_dir.return_value.__enter__.return_value = "/temp_dir"
        mei_template.get_temp_dir = get_temp_dir
        mei_template._create_random_filename = MagicMock(side_effect=["filename1", "filename2"])
        return mei_template

    @pytest.fixture
    def trained_model_table(self):
        return MagicMock()

    @pytest.fixture
    def selector_table(self):
        selector_table = MagicMock()
        selector_table.return_value.get_output_selected_model.return_value = "output_selected_model"
        return selector_table

    @pytest.fixture
    def method_table(self):
        method_table = MagicMock()
        method_table.return_value.generate_mei.return_value = dict(mei="mei", output="output")
        return method_table

    @pytest.fixture
    def seed_table(self):
        seed_table = MagicMock()
        seed_table.return_value.__and__.return_value.fetch1.return_value = "seed"
        return seed_table

    @pytest.fixture
    def insert1(self):
        return MagicMock()

    @pytest.fixture
    def save(self):
        return MagicMock()

    @pytest.fixture
    def model_loader_class(self, model_loader):
        return MagicMock(return_value=model_loader)

    @pytest.fixture
    def model_loader(self):
        model_loader = MagicMock()
        model_loader.load.return_value = "dataloaders", "model"
        return model_loader

    def test_if_model_loader_is_correctly_initialized(self, mei_template, trained_model_table, model_loader_class):
        mei_template(cache_size_limit=5)
        model_loader_class.assert_called_once_with(trained_model_table, cache_size_limit=5)

    def test_if_model_is_correctly_loaded(self, key, mei_template, model_loader):
        mei_template().make(key)
        model_loader.load.assert_called_once_with(key=key)

    def test_if_correct_model_output_is_selected(self, key, mei_template, selector_table):
        mei_template().make(key)
        selector_table.return_value.get_output_selected_model.assert_called_once_with("model", key)

    def test_if_seed_is_correctly_fetched(self, key, mei_template, seed_table):
        mei_template().make(key)
        seed_table.return_value.__and__.assert_called_once_with(key)
        seed_table.return_value.__and__.return_value.fetch1.assert_called_once_with("mei_seed")

    def test_if_mei_is_correctly_generated(self, key, mei_template, method_table):
        mei_template().make(key)
        method_table.return_value.generate_mei.assert_called_once_with(
            "dataloaders", "output_selected_model", key, "seed"
        )

    def test_if_mei_is_correctly_saved(self, key, mei_template, save):
        mei_template().make(key)
        assert save.call_count == 2
        save.has_calls(
            call("mei", "/temp_dir/mei_filename1.pth.tar"), call("output", "/temp_dir/output_filename2.pth.tar")
        )

    def test_if_mei_entity_is_correctly_saved(self, key, mei_template, insert1):
        mei_template().make(key)
        insert1.assert_called_once_with(
            dict(mei="/temp_dir/mei_filename1.pth.tar", output="/temp_dir/output_filename2.pth.tar")
        )
