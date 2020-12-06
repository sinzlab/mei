import torch
from torch import nn
import numpy as np


class FixedInputGenerator(nn.Module):
    def __init__(self, input_shape, mu=0, sigma=1):
        self.fixed_input = nn.Parameter(torch.randn(input_shape) * sigma + mu)

    def forward(self):
        return self.fixed_input


class FactorizedInputGenerator(nn.Module):
    def __init__(self, input_shape, n_factors=3, split=3, mu=0, sigma=1, n_inputs=1):
        super().__init__()
        target_shape = tuple(input_shape[1:])
        split -= 1
        factor_one_shape = target_shape[:split] + (1,) * (len(target_shape) - split)
        factor_two_shape = (1,) * split + target_shape[split:]
        self.register_buffer("mu", torch.tensor(mu, dtype=torch.float32))

        base_factors = []
        for _ in range(n_factors):
            parameter_list = []
            parameter_list.append(nn.Parameter(torch.randn(factor_one_shape)))
            parameter_list.append(nn.Parameter(torch.randn(factor_two_shape) * sigma))
            base_factors.append(nn.ParameterList(parameter_list))

        self.base_factors = nn.ModuleList(base_factors)
        self.factor_weights = nn.Parameter(torch.randn(n_inputs, n_factors))

    def forward(self):
        bases = []
        for factor_part in self.base_factors:
            bases.append(factor_part[0] * factor_part[1])
        bases = torch.stack(bases)
        orig_shape = bases.shape
        return (self.factor_weights @ bases.view(orig_shape[0], -1)).view(
            -1, *orig_shape[1:]
        ) + self.mu


class CPPN(nn.Module):
    def __init__(
        self, ndim=3, hiddens=(10, 10, 10), output_range=(0, 255), activation=None
    ):
        super().__init__()
        if activation is None:
            activation = nn.Tanh
        self.activation = activation
        components = []
        out = ndim
        for hidden in hiddens:
            components.append(nn.Linear(out, hidden))
            components.append(self.activation())
            out = hidden
        # final layer
        components.append(nn.Linear(out, 1))
        components.append(nn.Sigmoid())
        self.layers = nn.Sequential(*components)
        self.delta = output_range[1] - output_range[0]
        self.base = output_range[0]

    def forward(self, x):
        return self.layers(x) * self.delta + self.base


class CPPNGenerator(nn.Module):
    def __init__(
        self, input_shape, hiddens=(10, 10, 10), output_range=(0, 255), activation=None
    ):
        super().__init__()

        self.target_shape = (1,) + tuple(input_shape[1:])  # don't consider batch dim
        shape_info = np.array(self.target_shape)
        shape_info = shape_info[
            shape_info > 1
        ]  # only work with non singleton dimensions
        coordinates = np.meshgrid(*[np.arange(e) for e in shape_info])
        self.register_buffer(
            "coordinates",
            torch.tensor(
                np.stack([e.ravel() for e in coordinates]).T, dtype=torch.float32
            ),
        )
        self.cppn = CPPN(
            ndim=len(shape_info),
            hiddens=hiddens,
            output_range=output_range,
            activation=activation,
        )

    def forward(self):
        return self.cppn(self.coordinates).view(self.target_shape)
