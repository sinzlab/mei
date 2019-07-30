# featurevis
Visualize features that activate neurons via gradient ascent.

## Installation
After [installing PyTorch](https://pytorch.org/get-started/locally/), run:
```
pip3 install git+https://github.com/cajal/featurevis.git
```

## Usage
`feature_vis.gradient_ascent` receives a function $f(x)$ to optimize, an initial estimate $x$ and some optimization parameters like step size and number of iterations.

Optionally, it can receive any of: a differentiable `transform` $t(x)$ to apply to $x$ at each iteration before evaluating $f$, a differentiable `regularization` $r(x)$ to be minimized, i.e., optimization becomes:
$$\arg\max_{x} f(t(x)) - r(t(x))\text{ ,}$$
a `gradient_f` function $g(x)$ to apply to the gradient before applying the update and a `post_update` function $p(x)$ to apply to the updated $x$ after each iteration:
$$
    x_{t+1} = p\left(x_t + \alpha g\left(\frac{\delta f}{\delta x_t}\right)\right)\text{.}
$$
These functions ($t$, $r$, $g$ and $p$) should cover the most common scenarios when creating [feature visualizations](https://distill.pub/2017/feature-visualization/) for neural network models. We provide implementations for many of these commonly used functions in `feature_vis.ops`. 

You can check the `Examples.ipynb` notebook to see how to visualize features from a VGG network or real neurons[1] under different configurations. 

[1]: Models for real neurons come from a [private repo](https://github.com/cajal/static-networks) but the examples should still be a useful starting point.
