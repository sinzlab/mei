"""This module contains the main tables and table templates used in the MEI generation process."""

import datajoint as dj

from nnfabrik.main import Dataset, schema
from . import mixins


class TrainedEnsembleModelTemplate(mixins.TrainedEnsembleModelTemplateMixin, dj.Manual):
    """TrainedEnsembleModel table template.

    To create a functional "TrainedEnsembleModel" table, create a new class that inherits from this template and
    decorate it with your preferred Datajoint schema. Next assign the trained model table of your choosing to the class
    variable called "trained_model_table". By default the created table will point to the "Dataset" table in the
    Datajoint schema called "nnfabrik.main". This behaviour can be changed by overwriting the class attribute called
    "dataset_table".
    """

    dataset_table = Dataset

    class Member(mixins.TrainedEnsembleModelTemplateMixin.Member, dj.Part):
        """Member table template."""


class CSRFV1SelectorTemplate(mixins.CSRFV1SelectorTemplateMixin, dj.Computed):
    """CSRF V1 selector table template.

    To create a functional "CSRFV1Selector" table, create a new class that inherits from this template and decorate it
    with your preferred Datajoint schema. By default, the created table will point to the "Dataset" table in the
    Datajoint schema called "nnfabrik.main". This behavior can be changed by overwriting the class attribute called
    "dataset_table".
    """

    dataset_table = Dataset


@schema
class MEISeed(mixins.MEISeedMixin, dj.Lookup):
    """Seed table for MEI method."""


@schema
class MEIMethod(mixins.MEIMethodMixin, dj.Lookup):
    """Table that contains MEI methods and their configurations."""


class MEITemplate(mixins.MEITemplateMixin, dj.Computed):
    """MEI table template.

    To create a functional "MEI" table, create a new class that inherits from this template and decorate it with your
    preferred Datajoint schema. Next assign your trained model (or trained ensemble model) and your selector table to
    the class variables called "trained_model_table" and "selector_table". By default, the created table will point to
    the "MEIMethod" table in the Datajoint schema called "nnfabrik.main". This behavior can be changed by overwriting
    the class attribute called "method_table".
    """

    method_table = MEIMethod
    seed_table = MEISeed
