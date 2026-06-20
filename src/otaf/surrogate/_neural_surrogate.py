# -*- coding: utf-8
__author__ = "Kramer84"
__all__ = ["NeuralRegressorNetwork", "initialize_model_weights"]

import copy
from typing import Any, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import openturns as ot

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torcheval.metrics import R2Score

import otaf

DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)


class NeuralRegressorNetwork(nn.Module):
    """
    Construct a regression neural network with built-in data pipeline scaling.

    Orchestrate forward prediction transitions, data standardization buffers, 
    on-the-fly regularization noise injection, training execution monitors, 
    and multi-backend integrations (such as OpenTURNS wrappers).

    Parameters
    ----------
    input_dim : int
        The number of expected input features.
    output_dim : int
        The number of target regression prediction properties.
    X : Any
        Training input source matrix used to initialize normalization stats.
    y : Any
        Training target labels matrix used to initialize normalization stats.
    clamping : bool, default=False
        Flag forcing network parameter weights to bound inside `[-1.0, 1.0]`.
    finish_critertion_epoch : int, default=20
        The evaluation epoch count threshold required before evaluating early stopping.
    loss_finish : float, default=1e-16
        The absolute loss evaluation target minimum defining a perfect optimization termination.
    metric_finish : float, default=0.999
        The target validation R2 score metric triggering an optimized performance cutoff.
    max_epochs : int, default=100
        The upper restriction limit tracking total validation epoch runs.
    batch_size : int, default=100
        The structural size constraints partitioning batch iteration runs.
    compile_model : bool, default=True
        Flag enabling structural compilation graphs optimization wrappers.
    train_size : float, default=0.7
        The linear fraction determining the initial dataset splits partition layout.
    save_path : Optional[str], default=None
        Target operational file path pointing to tracking parameter serialization destinations.
    input_description : Optional[List[str]], default=None
        Descriptive string documentation defining features label elements properties.
    display_progress_disable : bool, default=True
        Flag disabling internal console progress indicator displays.
    input_normalization : bool, default=True
        Flag determining if input vectors undergo standard scale transformations.
    output_normalization : bool, default=True
        Flag determining if targets undergo standardization during optimization updates.
    noise_dims : List[int], default=[]
        Target dimensions collection selected to receive Gaussian noise augmentation.
    noise_level : float, default=0.1
        The structural coefficient multiplier applied to standard deviation noise scaling.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        X: Any,
        y: Any,
        clamping: bool = False,
        finish_critertion_epoch: int = 20,
        loss_finish: float = 1e-16,
        metric_finish: float = 0.999,
        max_epochs: int = 100,
        batch_size: int = 100,
        compile_model: bool = True,
        train_size: float = 0.7,
        save_path: Optional[str] = None,
        input_description: Optional[List[str]] = None,
        display_progress_disable: bool = True,
        input_normalization: bool = True,
        output_normalization: bool = True,
        noise_dims: List[int] = [],
        noise_level: float = 0.1,
    ) -> None:
        super().__init__()

        self.register_buffer("input_dim", torch.tensor(input_dim, dtype=int, requires_grad=False))
        self.register_buffer("output_dim", torch.tensor(output_dim, dtype=int, requires_grad=False))

        if input_dim == 1:
            X = X.reshape(-1, 1)
        if output_dim == 1:
            y = y.reshape(-1, 1)

        self.X_raw = copy.deepcopy(X)
        self.y_raw = copy.deepcopy(y)

        self.register_buffer("X_mean", torch.tensor(X.mean(axis=0), requires_grad=False))
        self.register_buffer("y_mean", torch.tensor(y.mean(axis=0), requires_grad=False))
        self.register_buffer(
            "X_std",
            torch.tensor(np.where(X.std(axis=0) == 0.0, 1, X.std(axis=0)), requires_grad=False),
        )
        self.register_buffer(
            "y_std",
            torch.tensor(np.where(y.std(axis=0) == 0.0, 1, y.std(axis=0)), requires_grad=False),
        )

        if input_normalization:
            self.X = (torch.tensor(self.X_raw, dtype=torch.float32) - self.X_mean) / self.X_std
        else:
            self.X = torch.tensor(self.X_raw, dtype=torch.float32)

        if output_normalization:
            self.y = (torch.tensor(self.y_raw, dtype=torch.float32) - self.y_mean) / self.y_std
        else:
            self.y = torch.tensor(self.y_raw, dtype=torch.float32)

        self.clamping = clamping

        # Finish criterions
        self.finish_critertion_epoch = finish_critertion_epoch
        self.loss_finish = loss_finish
        self.metric_finish = metric_finish

        self.compile_model = compile_model
        self.train_size = train_size

        self.save_path = save_path
        self.input_description = input_description
        self.display_progress_disable = display_progress_disable

        self.register_buffer(
            "input_normalization",
            torch.tensor(input_normalization, dtype=bool, requires_grad=False),
        )
        self.register_buffer(
            "output_normalization",
            torch.tensor(output_normalization, dtype=bool, requires_grad=False),
        )

        self.get_train_test_data()

        # Initialize the base model, output of d 1
        self.model = otaf.surrogate.get_base_relu_mlp_model(input_dim, output_dim)

        # Performance metric
        self.metric = R2Score()

        # loss function and optimizer
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = None
        self.scheduler = None

        # training parameters
        self.n_epochs = max_epochs  # number of epochs to run
        self.batch_size = batch_size  # size of each batch
        self.batch_start = torch.arange(0, len(self.X_train), self.batch_size)

        # Hold the best model
        self.best_metric = 0
        self.best_loss = np.inf  # init to infinity
        self.best_weights = copy.deepcopy(self.state_dict())
        self.history_loss = []
        self.history_metric = []

        self.noise_dims = noise_dims
        self.noise_level = noise_level

    @classmethod
    def from_checkpoint(cls, filepath: str) -> NeuralRegressorNetwork:
        """
        Instantiate the network directly from a saved checkpoint, bypassing the need for training data.

        Extract normalization statistics, tracking buffers metadata, and architectural state dictionaries 
        to rebuild the complete operational execution module instance.

        Parameters
        ----------
        filepath : str
            The path pointing to the serialized target checkpoint storage dictionary.

        Returns
        -------
        NeuralRegressorNetwork
            The fully restored, evaluation-ready model instance context.
        """
        checkpoint = torch.load(filepath, map_location=DEVICE, weights_only=False)

        # Create an empty instance bypassing __init__
        instance = cls.__new__(cls)
        super(NeuralRegressorNetwork, instance).__init__()

        input_dim = checkpoint['input_dim']
        output_dim = 1 # Assuming 1 for this specific implementation

        instance.register_buffer("input_dim", torch.tensor(input_dim, dtype=torch.int, requires_grad=False))
        instance.register_buffer("output_dim", torch.tensor(output_dim, dtype=torch.int, requires_grad=False))

        # Restore normalization buffers explicitly
        norm = checkpoint.get('normalization_metadata', {})
        instance.register_buffer("X_mean", norm.get('X_mean', torch.zeros(input_dim)))
        instance.register_buffer("y_mean", norm.get('y_mean', torch.zeros(output_dim)))
        instance.register_buffer("X_std", norm.get('X_std', torch.ones(input_dim)))
        instance.register_buffer("y_std", norm.get('y_std', torch.ones(output_dim)))

        instance.register_buffer("input_normalization", torch.tensor(checkpoint.get('input_normalization', True), dtype=torch.bool))
        instance.register_buffer("output_normalization", torch.tensor(checkpoint.get('output_normalization', True), dtype=torch.bool))

        # Rebuild architecture
        parsed_arch = [input_dim if str(a).lower() == 'dim' else int(a) for a in checkpoint['architecture']]
        instance.model = torch.nn.Sequential(
            *otaf.surrogate.get_custom_mlp_layers(parsed_arch, activation_class=torch.nn.GELU)
        )

        # Load weights
        instance.model.load_state_dict(checkpoint['model_state_dict'])
        instance.eval()

        return instance

    def evaluate_model_non_standard_space(
        self, 
        x: Any, 
        batch_size: int = 50000, 
        return_on_gpu: bool = False
    ) -> torch.Tensor:
        """
        Evaluate the model in non-standard space, processing the input in batches.

        Apply input preprocessing standardization steps, stream vectors through network layers via memory-safe 
        batch windows, restore structural target scaling properties, and gather predictions.

        Parameters
        ----------
        x : Any
            The input coordinates collection requiring evaluation inference.
        batch_size : int, default=50000
            The linear size constraints bounding separate batch passes.
        return_on_gpu : bool, default=False
            Flag forcing the final aggregated matrix allocations to remain tracking on active device pools.

        Returns
        -------
        torch.Tensor
            The consolidated prediction outputs translated back into original target scaling.
        """
        # Standardize the input
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)

        if self.input_normalization:
            x = (x - self.X_mean.cpu()) / self.X_std.cpu()

        # Ensure the model is in eval mode and sent to the right device
        self.eval()
        self.to(DEVICE)

        # Prepare to collect the results
        results = []

        try:
            # Evaluate in batches
            with torch.no_grad():
                for i in range(0, len(x), batch_size):
                    batch = x[i : i + batch_size].to(DEVICE)
                    output = self(batch)
                    # Convert the output to the original space and move it to CPU if necessary
                    if self.output_normalization:
                        output = output * self.y_std + self.y_mean

                    if not return_on_gpu:
                        output = output.cpu()

                    results.append(output)

            # Concatenate all batch results into a single tensor
            final_result = torch.cat(results, dim=0)
        finally:
            # Ensure to free up the GPU memory
            del batch, output
            torch.cuda.empty_cache()

        return final_result

    def pf_monte_carlo_bruteforce(
        self,
        composed_distribution: Any,
        N_MC_MAX: int = int(1e9),
        N_GEN_MAX: int = int(1e7),
        batch_size: int = 500000,
        PF_STAB: float = 1e-6,
        threshold: float = 0.0,
    ) -> float:
        """
        Estimate failure probabilities via highly optimized brute force Monte Carlo random sampling loops.

        Sample sequentially from the provided statistical distribution boundary configurations, map features 
        into standard normalization metrics spaces, pass elements through evaluation layers, and monitor 
        convergence variances against tracking limits.

        Parameters
        ----------
        composed_distribution : Any
            Statistical multivariate distribution template generating coordinate variations.
        N_MC_MAX : int, default=1000000000
            The ultimate absolute sampling ceiling threshold bounding execution parameters.
        N_GEN_MAX : int, default=10000000
            The generation buffer block size managing separate structural sampling intervals.
        batch_size : int, default=500000
            The structural segmentation sizes parsing inference tensor blocks.
        PF_STAB : float, default=1e-6
            The target stability variance checking boundary delta used to confirm optimization stability.
        threshold : float, default=0.0
            The objective scalar valuation defining the transition boundary into failure conditions.

        Returns
        -------
        float
            The final calculated empirical probability of failure score indicator.
        """
        with torch.no_grad():
            N_FIN = 0  # Final size of monte carlo
            X_std, X_mean = (
                self.X_std.clone().detach().to(DEVICE),
                self.X_mean.clone().detach().to(DEVICE),
            )
            y_std, y_mean = (
                self.y_std.clone().detach().to(DEVICE),
                self.y_mean.clone().detach().to(DEVICE),
            )
            self.eval()
            self.to(DEVICE)
            means = list(composed_distribution.getMean())
            stnds = list(composed_distribution.getStandardDeviation())
            torchDist = torch.distributions.Normal(torch.Tensor(means), torch.Tensor(stnds))
            pf_array = np.array([], dtype="float32")
            failures = (
                torch.Tensor().to(torch.int8).to_sparse().to(DEVICE)
            )  # Array to store the failures
            for i in range(N_MC_MAX // N_GEN_MAX):
                sample = torchDist.sample((N_GEN_MAX,)).to("cpu")
                sample = (sample - X_mean.cpu()) / X_std.cpu()

                for j in range(0, N_GEN_MAX, batch_size):
                    batch = sample[j : j + batch_size].to(DEVICE)
                    if self.output_normalization:
                        output = self(batch) * y_std + y_mean
                    output = (
                        torch.squeeze(torch.where(output < threshold, 1, 0))
                        .to(torch.int8)
                        .to_sparse()
                    )

                    failures = torch.cat((failures, output), 0)
                    total_failures = failures.sum().item()
                    total_samples_processed = failures.numel()

                    # Calculate the current probability of failure
                    pf_cpu = total_failures / total_samples_processed
                    pf_cpu_mod = (total_failures + 1) / total_samples_processed

                    pf_array = np.append(pf_array, pf_cpu)

                    # Check the stopping criterion within the inner loop
                    if abs(pf_cpu - pf_cpu_mod) < PF_STAB:
                        print(
                            f"Finished at iteration {int((i+1)*(j/batch_size+1))} with {total_samples_processed} experiments. Pf: {pf_cpu}"
                        )
                        return pf_array[-1]  # or np.mean(pf_array) to return the average Pf

        return pf_array[-1]  # or np.mean(pf_array) if averaging is preferred

    def get_train_test_data(self) -> None:
        """
        Partition structural data sources into distinct localized training and validation slices.

        Utilize proportional tracking division configurations to build randomized test boundaries.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, train_size=self.train_size, shuffle=True
        )
        self.X_train = X_train
        self.y_train = torch.atleast_2d(y_train)
        self.X_test = X_test
        self.y_test = torch.atleast_2d(y_test)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Execute forward propagation transitions mapping inputs through standard hidden processing layers.

        Parameters
        ----------
        x : torch.Tensor
            The normalized feature tensor block submitted for predictive transformation.

        Returns
        -------
        torch.Tensor
            The direct unscaled target linear prediction layers response.
        """
        logits = self.model(x)
        return logits

    def train_model(self) -> None:
        """
        Execute the structural multi-epoch optimization training routine sequence loops.

        Manage batch data streams distributions, coordinate backward auto-differentiation passes, track loss 
        and validation metric states histories, evaluate convergence metrics, and cache optimized parameter profiles.
        """
        self.to(DEVICE)
        self.model.to(DEVICE)
        if otaf.common.is_running_in_notebook():
            tq = tqdm.tqdm_notebook
        else:
            tq = tqdm.tqdm

        try:
            # training loop
            y_test = self.y_test.to(DEVICE)
            X_test = self.X_test.to(DEVICE)
            for epoch in range(self.n_epochs):
                # Model training
                self.train(True)
                with tq(
                    self.batch_start,
                    unit="batch",
                    mininterval=0.05,
                    disable=self.display_progress_disable,
                ) as bar:
                    bar.set_description(f"Epoch {epoch:03d}")
                    for start in bar:
                        # take a batch
                        X_batch = self.X_train[start : start + self.batch_size]
                        y_batch = self.y_train[start : start + self.batch_size]
                        # Add noise if needed:
                        if self.noise_dims and self.noise_level > 0:
                            X_batch[:, self.noise_dims] = otaf.surrogate.add_gaussian_noise(
                                X_batch[:, self.noise_dims], self.noise_level
                            )

                        # backward pass
                        self.optimizer.zero_grad()
                        # forward pass
                        y_pred = self(X_batch.to(DEVICE))
                        loss = self.loss_fn(y_pred, y_batch.to(DEVICE))
                        loss.backward()
                        # update weights
                        self.optimizer.step()
                        # print progress
                        bar.set_postfix(loss=loss.item(), lr=self.optimizer.param_groups[0]["lr"])

                        if self.clamping:
                            for p in self.parameters():
                                p.data.clamp_(-1.0, 1.0)

                # validation
                self.eval()
                y_pred = self(X_test)
                loss = self.loss_fn(y_pred, y_test)
                loss_f = float(loss)
                self.history_loss.append(loss_f)
                self.metric.update(y_test, y_pred)
                R2 = float(self.metric.compute())
                self.history_metric.append(R2)
                self.metric.reset()

                print(f"Epoch {epoch + 1:03d}, Val Loss: {loss_f:.6f}, Val R2: {R2:.6f}")

                if loss_f < self.best_loss:
                    self.best_metric = R2
                    self.best_loss = loss_f
                    self.best_weights = copy.deepcopy(self.state_dict())

                if self.training_stopping_criterion(epoch):
                    break

                if self.scheduler:
                    self.scheduler.step()

        except KeyboardInterrupt:
            print("Training interrupted by user")

        print(
            f"Finished training at epoch {epoch+1} with best loss {self.best_loss:.6f} and R2 of {self.best_metric:.6f}"
        )

        # restore self.model and return best accuracy
        self.load_state_dict(self.best_weights)
        torch.cuda.empty_cache()

    def plot_results(self) -> None:
        """
        Visualize multi-axis training history performance tracking diagrams.

        Render historical evaluation sequences contrasting training loss declines against tracked target 
        performance metrics development pathways across execution cycles.
        """
        fig = plt.figure()
        ax1 = fig.add_subplot(111, label="1")
        ax2 = fig.add_subplot(111, label="2", frame_on=False)

        ax1.plot(self.history_loss, color="C0")
        ax1.set_xlabel("Epoch", color="C0")
        ax1.set_ylabel("Loss", color="C0")
        ax1.tick_params(axis="x", colors="C0")
        ax1.tick_params(axis="y", colors="C0")

        ax2.plot(self.history_metric, color="C1")
        ax2.yaxis.tick_right()
        ax2.set_ylabel("Metric", color="C1")
        ax2.yaxis.set_label_position("right")
        ax2.tick_params(axis="y", colors="C1")
        ax2.set_xticks([])

        plt.show()

    def save_model(self) -> None:
        """Serialize current weight attributes and parameters properties targeting the configured destination path."""
        torch.save(self.state_dict(), self.save_path)

    def load_model(self) -> None:
        """Restore structural model attributes configurations from target destination parameters storage paths."""
        self.load_state_dict(torch.load(self.save_path))
        self.best_weights = copy.deepcopy(self.state_dict())

    def get_model_as_openturns_function(self, batch_size: int = 50000) -> ot.PythonFunction:
        """
        Wrap the localized surrogate network instance inside an OpenTURNS Python compatibility module container.

        Expose functional evaluation wrappers, gradients tracking pipelines, and structural second-order partial 
        derivative Hessian interfaces matching backend data tracking requirements.

        Parameters
        ----------
        batch_size : int, default=50000
            The block streaming partition limits tracking data evaluations.

        Returns
        -------
        ot.PythonFunction
            The instantiated OpenTURNS functional interface object tracking the neural surrogate model mappings.
        """
        func = lambda x: np.array(
            self.evaluate_model_non_standard_space(x, batch_size).detach().numpy()
        )
        otFunc = ot.PythonFunction(
            self.input_dim,
            self.output_dim,
            # func=func,
            func_sample=func,
            gradient=self.gradient,
            hessian=self.hessian,
        )
        otFunc.setName("OpenTURNS Python function linking to AI surrogate.")
        if self.input_description:
            otFunc.setInputDescription(self.input_description)
        return otFunc

    def gradient(self, inP: Any) -> np.ndarray:
        """
        Calculate first-order partial derivative Jacobian values targeting input tracking vectors configurations.

        Parameters
        ----------
        inP : Any
            The coordinate input array profile subjected to differentiation tracking.

        Returns
        -------
        np.ndarray
            The converted array tracking structural gradient slopes.
        """
        inP = torch.tensor(np.array(inP), dtype=torch.float32, requires_grad=True)
        y = self.evaluate_model_non_standard_space(inP)
        return np.array(jacobian(y, inP))

    def hessian(self, inP: Any) -> np.ndarray:
        """
        Calculate second-order partial derivative Hessian values targeting input tracking vectors configurations.

        Parameters
        ----------
        inP : Any
            The coordinate input array profile subjected to second-order differentiation tracking.

        Returns
        -------
        np.ndarray
            The converted array matrix tracking structural curvature parameters components.
        """
        inP = torch.tensor(np.array(inP), dtype=torch.float32, requires_grad=True)
        y = self.evaluate_model_non_standard_space(inP)
        return np.array(hessian(y, inP))

    def training_stopping_criterion(self, epoch: int) -> bool:
        """
        Determine whether training should stop based on various criteria.

        Assess performance indicators including minimum loss tolerances achievement thresholds 
        and validation performance requirements completion targets.

        Parameters
        ----------
        epoch : int
            The identifier integer index tracking the active model optimization cycle.

        Returns
        -------
        bool
            True if any termination threshold evaluates as fulfilled, directing operations to close.
        """
        # Check various conditions to determine if training should stop
        if epoch > self.finish_critertion_epoch:
            if self.best_loss <= self.loss_finish:
                print("Stopping: Loss below threshold.")
                return True
            elif self.best_metric >= self.metric_finish:
                print("Stopping: R2 score is high enough.")
                return True
        return False


def initialize_model_weights(
    model: nn.Module, 
    init_type: str = "xavier_uniform", 
    init_gain: float = 0.02
) -> None:
    """
    Initialize structural weights and biases across all compatible sub-modules.

    Apply a designated initialization distribution algorithm to parameterized 
    convolutional, linear, and batch normalization layers found within the network.

    Parameters
    ----------
    model : nn.Module
        The PyTorch neural network model instance to undergo weight initialization.
    init_type : str, default="xavier_uniform"
        The identification name of the initialization technique. Must be one of:
        'normal', 'xavier_uniform', 'xavier_normal', 'kaiming_uniform',
        'kaiming_normal', 'orthogonal', 'uniform'.
    init_gain : float, default=0.02
        Scaling multiplier configuration applied to normal, xavier, orthogonal, 
        and uniform distributions.

    Raises
    ------
    NotImplementedError
        If the provided `init_type` string does not match any known initialization routine.
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier_uniform":
                nn.init.xavier_uniform_(m.weight.data, gain=init_gain)
            elif init_type == "xavier_normal":
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming_uniform":
                nn.init.kaiming_uniform_(m.weight.data, a=0, mode="fan_in", nonlinearity="relu")
            elif init_type == "kaiming_normal":
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in", nonlinearity="relu")
            elif init_type == "orthogonal":
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            elif init_type == "uniform":
                nn.init.uniform_(m.weight.data, -init_gain, init_gain)
            else:
                raise NotImplementedError(f"Initialization method [{init_type}] is not implemented")

            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

        elif classname.find("BatchNorm") != -1:
            nn.init.constant_(m.weight.data, 1.0)
            nn.init.constant_(m.bias.data, 0.0)

    model.apply(init_func)


## Custom jacobian and hessian in pytorch #########################################################


def jacobian(y: torch.Tensor, x: torch.Tensor, create_graph: bool = False) -> torch.Tensor:
    """
    Compute the Jacobian matrix of a tensor y with respect to an input tensor x.

    Evaluate first-order partial derivatives tracking individual element gradients 
    sequentially via vector-Jacobian products, reconstruction-shaping the aggregated 
    outputs to correspond to the tensor dimension combinations.

    Parameters
    ----------
    y : torch.Tensor
        The dependent target output tensor to differentiate.
    x : torch.Tensor
        The independent input tensor with respect to which gradients are evaluated.
    create_graph : bool, default=False
        If True, construct the graph during the computation, allowing for higher-order 
        derivative evaluations (such as Hessians).

    Returns
    -------
    torch.Tensor
        The compiled Jacobian matrix matching the combined shapes of `y` and `x`.
    """
    jac = []
    flat_y = y.reshape(-1)
    grad_y = torch.zeros_like(flat_y)
    for i in range(len(flat_y)):
        grad_y[i] = 1.0
        (grad_x,) = torch.autograd.grad(
            flat_y, x, grad_y, retain_graph=True, create_graph=create_graph
        )
        jac.append(grad_x.reshape(x.shape))
        grad_y[i] = 0.0
    return torch.stack(jac).reshape(y.shape + x.shape)


def hessian(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Compute the Hessian matrix of a scalar or tensor y with respect to x.

    Evaluate the second-order partial derivatives by executing nested auto-differentiation 
    jacobian operations, tracking gradients through the computational graph.

    Parameters
    ----------
    y : torch.Tensor
        The dependent target output tensor to differentiate.
    x : torch.Tensor
        The independent input tensor with respect to which second-order gradients are evaluated.

    Returns
    -------
    torch.Tensor
        The computed Hessian matrix containing the second-order partial derivative combinations.
    """
    return jacobian(jacobian(y, x, create_graph=True), x)
