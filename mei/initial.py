from abc import ABC, abstractmethod

from torch import Tensor, randn

import os
fetch_download_path = os.environ.get('FETCH_DOWNLOAD_PATH', '/data/fetched_from_attach')


class InitialGuessCreator(ABC):
    """Implements the interface used to create an initial guess."""

    @abstractmethod
    def __call__(self, *shape) -> Tensor:
        """Creates an initial guess from which to start the MEI optimization process given a shape."""


class RandomNormal(InitialGuessCreator):
    """Used to create an initial guess tensor filled with values distributed according to a normal distribution."""

    _create_random_tensor = randn

    def __call__(self, *shape):
        """Creates a random initial guess from which to start the MEI optimization process given a shape."""
        return self._create_random_tensor(*shape)


    def __repr__(self):
        return f"{self.__class__.__qualname__}()"


class RandomNormalNullChannel(InitialGuessCreator):
    """Used to create an initial guess tensor filled with values distributed according to a normal distribution."""

    _create_random_tensor = randn

    def __init__(self, null_channel, null_value=0):
        self.null_channel = null_channel
        self.null_value = null_value

    def __call__(self, *shape):
        """Creates a random initial guess from which to start the MEI optimization process given a shape."""
        inital = self._create_random_tensor(*shape)
        inital[:, self.null_channel, ...] = self.null_value
        return inital

    def __repr__(self):
        return f"{self.__class__.__qualname__}()"

    
class RandomNormalCenterRing(InitialGuessCreator):
    """Used to create an initial guess tensor filled with values distributed according to a normal distribution."""

    _create_random_tensor = randn

    def __init__(self, key, mask_thres_for_center=0.3):
        inner_ensemble_hash = key["inner_ensemble_hash"]
        src_method_hash = key["src_method_hash"]
        unit_id = key["unit_id"]

        inner_mei_path = (MEI & dict(ensemble_hash=inner_ensemble_hash) & dict(method_hash=src_method_hash) & dict(unit_id=unit_id)).fetch1('mei', download_path=fetch_download_path)
        centermei = torch.load(inner_mei_path)
        self.centermask = (self.centermei[0][1] > mask_thres_for_center) * 1
        self.centerimg = self.centermei[0][0]
        
    def __call__(self, *shape):
        """Creates a random initial guess from which to start the MEI optimization process given a shape."""
        randinitial = self._create_random_tensor(*shape)
        initial = randinitial * (1.0 - self.centermask) + self.centermask * self.centerimg
        return initial

    def __repr__(self):
        return f"{self.__class__.__qualname__}()"