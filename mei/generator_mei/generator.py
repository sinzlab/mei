import torch
from torch import nn
import numpy as np
from collections import OrderedDict
from torch.nn import functional as F


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


class BankFactorGenerator(nn.Module):
    def __init__(
        self, input_shape, n_factors=3, n_banks=5, split=3, mu=0, sigma=1, n_inputs=1
    ):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_banks = n_banks
        self.n_factors = n_factors

        target_shape = tuple(input_shape[1:])
        split -= 1
        factor1_shape = (
            target_shape[:split] + (1,) * (len(target_shape) - split) + (n_banks,)
        )
        factor2_shape = (1,) * split + target_shape[split:] + (n_banks,)
        self.register_buffer("mu", torch.tensor(mu, dtype=torch.float32))

        self.factor1_bank = nn.Parameter(torch.randn(factor1_shape))
        self.factor2_bank = nn.Parameter(torch.randn(factor2_shape) * sigma)
        self.factor1_weights = nn.Parameter(torch.randn(n_inputs * n_factors, n_banks))
        self.factor2_weights = nn.Parameter(torch.randn(n_inputs * n_factors, n_banks))

        target_shape = tuple(input_shape[1:])

    def forward(self):
        f1 = F.linear(self.factor1_bank, self.factor1_weights)
        f2 = F.linear(self.factor2_bank, self.factor2_weights)
        comb = f1 * f2
        comb = comb.view(*comb.shape[:-1], self.n_inputs, self.n_factors).sum(dim=-1)
        return comb.permute(-1, *range(len(comb.shape) - 1)) + self.mu


class CPPN(nn.Module):
    def __init__(
        self,
        ndim=3,
        nout=1,
        hiddens=(10, 10, 10),
        output_range=(0, 255),
        activation=None,
        bias=1.0,
        std=1.0,
    ):
        super().__init__()
        self.activation = activation
        self.bias = bias
        self.std = std
        components = []
        out = ndim
        for hidden in hiddens:
            components.append(nn.Linear(out, hidden))
            if activation is not None:
                components.append(self.activation())
            out = hidden
        # final layer
        components.append(nn.Linear(out, nout))
        components.append(nn.Sigmoid())
        self.layers = nn.Sequential(*components)
        self.delta = output_range[1] - output_range[0]
        self.base = output_range[0]
        self.initialize()

    def initialize(self):
        def val(x):
            if isinstance(x, nn.Linear):
                nn.init.normal_(x.weight, std=self.std)
                nn.init.constant_(x.bias, self.bias)

        self.apply(val)

    def forward(self, x):
        return self.layers(x) * self.delta + self.base


class CPPNGenerator(nn.Module):
    def __init__(
        self,
        input_shape,
        hiddens=(10, 10, 10),
        output_range=(0, 255),
        activation=nn.Tanh,
        bias=1.0,
        std=1.0,
    ):
        super().__init__()

        self.target_shape = (1,) + tuple(input_shape[1:])  # don't consider batch dim
        shape_info = np.array(self.target_shape)
        shape_info = shape_info[
            shape_info > 1
        ]  # only work with non singleton dimensions
        coordinates = np.meshgrid(*[np.arange(e) for e in shape_info], indexing="ij")
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
            bias=bias,
            std=std,
        )

    def forward(self):
        return self.cppn(self.coordinates).view(self.target_shape)


class ExpandedCPPN(nn.Module):
    def __init__(
        self,
        ndim=2,
        layers=8,
        width=15,
        aux_dim=1,
        out_channels=1,
        expand_sinusoids=True,
        output_range=(0, 255),
        nonlinearity=nn.Tanh,
        final_nonlinearity=nn.Sigmoid,
        bias=False,
    ):
        """
        CPPN class producing images from coordinates (and some function of coordinates) + auxiliary inputs

        Args:
            ndim (int, optinal): Defaults to 2.
            layers (int, optional): [description]. Defaults to 8.
            width (int, optional): [description]. Defaults to 20.
            aux_dim (int, optional): [description]. Defaults to 1.
            out_channels (int, optional): [description]. Defaults to 1.
            nonlinearity ([type], optional): [description]. Defaults to Cos.
            final_nonlinearity ([type], optional): [description]. Defaults to nn.Sigmoid.
            bias (bool, optional): whether to use bias term for the linear layers. Defaults to False.
        """

        super().__init__()

        # TODO: make this flexible
        self.ndim = ndim  # number of deterministic input dimensions

        self.aux_dim = aux_dim  # number of auxiliary dimensions
        self.expand_sinusoids = expand_sinusoids
        dim_expansion = 3 if expand_sinusoids else 1
        self.in_dim = self.ndim * dim_expansion + self.aux_dim
        self.out_channels = out_channels

        # scaling of extra coordinate dims in sinusoidal space
        if self.expand_sinusoids:
            self.scale = nn.Parameter(torch.rand(self.ndim * 2) + 1)
        else:
            self.scale = None

        # add layers
        n_input = self.in_dim
        elements = []
        for i in range(layers - 1):
            elements.append((f"layer{i}", nn.Linear(n_input, width, bias=bias)))
            if nonlinearity is not None:
                elements.append((f"nonlinearity{i}", nonlinearity()))
            n_input = width

        # last layer
        elements.append(
            (f"layer{layers-1}", nn.Linear(n_input, out_channels, bias=bias))
        )
        elements.append((f"nonlinearity{layers-1}", final_nonlinearity()))

        self.func = nn.Sequential(OrderedDict(elements))

        self.delta = output_range[1] - output_range[0]
        self.base = output_range[0]

        self.apply(self.weights_init)

    @staticmethod
    def weights_init(m):
        if "Linear" in m.__class__.__name__:
            m.weight.data.normal_(0.0, 1.0)
            if m.bias is not None:
                m.bias.data.fill_(0)

    def forward(self, aux, out_shape):
        """
        Generates images.
        Args:
            aux (torch.tensor): auxiliary inputs which has a shape of (images_n, aux_dim).
                Therefore, if you want 10 images, for instance, pass a tensor with dimensions (10, aux_dim).
            out_shape (tuple): specify the size (height and width) of the output image(s)
        Returns:
            torch.tensor: images with shape (images_n, channels, height, width)
        """

        device = aux.device

        # number of images to be produced
        n = aux.shape[0]

        # get the coordinate values
        coords = torch.meshgrid(
            [torch.linspace(-1, 1, shape, device=device) for shape in out_shape]
        )

        # add sinusoids
        if self.expand_sinusoids:
            sin_coords = [
                torch.sin(sin_scale * coord)
                for sin_scale, coord in zip(self.scale[::2], coords)
            ]
            cos_coords = [
                torch.cos(cos_scale * coord)
                for cos_scale, coord in zip(self.scale[1::2], coords)
            ]
            coords = list(coords) + sin_coords + cos_coords

        fixed_inputs = torch.stack(coords, dim=-1)
        fixed_inputs = fixed_inputs.unsqueeze(0).expand(n, *fixed_inputs.shape)

        aux_inputs = aux.view(n, *([1] * len(out_shape)), self.aux_dim).expand(
            -1, *out_shape, -1
        )  # n_images x h x w x aux_dim

        # concatentate the inputs and pass through the network
        x = torch.cat(
            (fixed_inputs, aux_inputs), dim=-1
        )  # n_images x h x w x (fixed_dim + aux_dim)
        y = self.func(x).permute(0, -1, *tuple(range(1, self.ndim + 1)))

        return y * self.delta + self.base


class ExpandedCPPNGenerator(nn.Module):
    def __init__(
        self,
        input_shape,
        aux_dim,
        n_neurons=None,
        out_channels=None,
        # cppn-specific arguments
        output_range=(0, 255),
        layers=8,
        width=15,
        expand_sinusoids=True,
        nonlinearity=nn.Tanh,
        final_nonlinearity=nn.Sigmoid,
    ):

        super().__init__()
        self.aux_dim = aux_dim
        batch, channel, *out_shape = input_shape
        if n_neurons is None:
            n_neurons = batch
        self.n_neurons = n_neurons
        if out_channels is None:
            out_channels = channel
        self.out_shape = out_shape
        ndim = len(out_shape)
        self._neuron_embeddings = nn.Parameter(torch.rand(n_neurons, aux_dim) - 0.5)
        self.cppn = ExpandedCPPN(
            ndim=ndim,
            aux_dim=aux_dim,
            layers=layers,
            width=width,
            output_range=output_range,
            expand_sinusoids=expand_sinusoids,
            out_channels=out_channels,
            nonlinearity=nonlinearity,
            final_nonlinearity=final_nonlinearity,
            bias=False,
        )

    @property
    def neuron_embeddings(self):
        #         return torch.sigmoid(self._neuron_embeddings)
        return self._neuron_embeddings

    def forward(self, out_shape=None):
        out_shape = out_shape if out_shape is not None else self.out_shape
        return self.cppn(self.neuron_embeddings, out_shape)