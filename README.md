[![Test](https://github.com/sinzlab/mei/actions/workflows/test.yml/badge.svg)](https://github.com/sinzlab/mei/actions/workflows/test.yml)
[![Black](https://github.com/sinzlab/mei/actions/workflows/black.yml/badge.svg)](https://github.com/sinzlab/mei/actions/workflows/black.yml)
[![Isort](https://github.com/sinzlab/mei/actions/workflows/isort.yml/badge.svg)](https://github.com/sinzlab/mei/actions/workflows/isort.yml)

Generate most exciting inputs (MEIs).

[Documentation](docs/README.md)

## Installation

The easiest way to install the package is via pip:

```python
pip install neuro-mei
```

If you want to install from source from a version, for example version `v0.1.0`, you can run the following command:

```python
pip install git+https://github.com/sinzlab/mei.git@v0.1.0
```

## Quickstart

Check out the demo notebooks in the `demo` folder to get started.
- [Quickstart](demo/quickstart.ipynb)
- [DataJoint Integration](demo/dj_integration.ipynb)

Simple usage with a pre-trained model:
```python
from mei.methods import gradient_ascent

# Create an instance of the model
model = ... # your model here

method_config = {
    "initial": {"path": "mei.initial.RandomNormal"},
    "optimizer": {"path": "torch.optim.SGD", "kwargs": {"lr": 0.1}},
    "stopper": {"path": "mei.stoppers.NumIterations", "kwargs": {"num_iterations": 1000}},
    "objectives": [
        {"path": "mei.objectives.EvaluationObjective", "kwargs": {"interval": 10}}
    ],
    "device": "cuda",
}

mei, _, _ = gradient_ascent(model, config=method_config, seed=0, shape=(1, 1, 100, 100))
```

## How to run the tests :test_tube:

Clone this repository and run the following command from within the cloned repository to run all tests:

```bash
docker-compose run pytest
```

## How to contribute :fire:

Pull requests (and issues) are always welcome. This section describes some
preconditions that pull requests need to fulfill.

### Tests

Please make sure your changes pass the tests. Take a look at the [test running
section](#how-to-run-the-tests-test_tube) for instructions on how to run them. Adding tests
for new code is highly recommended.

### Code Style

#### black

This project uses the [black](https://github.com/psf/black) code formatter. You
can check whether your changes comply with its style by running the following
command:

```bash
docker-compose run black
```

Furthermore you can pass a path to the service to have black fix any errors in
the Python modules it finds in the given path.

#### isort

[isort](https://github.com/PyCQA/isort) is used to sort Python imports. You can check the order of imports by running the following command:

```bash
docker-compose run isort
```

The imports can be sorted by passing a path to the service.
