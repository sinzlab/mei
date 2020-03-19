class MEI:
    """Wrapper around the function and the MEI tensor."""

    def __init__(self, func, initial_guess):
        """Initializes MEI.

        Args:
            func: A callable that will receive the to be optimized MEI tensor of floats as its only argument and that
                must return a tensor containing a single float.
            initial_guess: A tensor containing floats representing the initial guess to start the optimization process
                from.
        """
        self.func = func
        self.initial_guess = initial_guess
        self._mei = self._initialize_mei()

    def _initialize_mei(self):
        """Detaches and clones the initial guess and enables the gradient on the cloned tensor."""
        mei = self.initial_guess.detach().clone()
        mei.requires_grad_()
        return mei

    def evaluate(self):
        """Evaluates the current MEI on the callable and returns the result."""
        return self.func(self._mei)

    def __call__(self):
        """Detaches and clones the current MEI and returns it."""
        return self._mei.detach().clone()

    def __repr__(self):
        return f"{self.__class__.__qualname__}({self.func}, {self.initial_guess})"


def optimize(mei, optimizer, optimized):
    """Optimizes the input to a given function such that it maximizes said function using gradient ascent.

    Args:
        mei: An instance of the to be optimized MEI.
        optimizer: A PyTorch-style optimizer class.
        optimized: A callable that receives the current optimal input as its argument before each optimization step
            and returns a boolean. The optimization process will be stopped if it returns True.

    Returns:
        A tensor of floats having the same shape as "initial_guess" representing the input that maximizes the function.
    """
    evaluation = mei.evaluate()
    while not optimized(mei, evaluation):
        optimizer.zero_grad()
        evaluation = mei.evaluate()
        (-evaluation).backward()
        optimizer.step()
    return mei()
