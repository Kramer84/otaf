from __future__ import annotations

import re
import copy
import numpy as np
from enum import Enum
from typing import Tuple, Sequence
import argparse
import sys

import otaf
import sympy as sp
import torch

#These sripts will load the models then train the neural network, then save it.
#Let's start with some generalistic functions.

import argparse
import sys
import numpy as np
import torch
import otaf

class SurrogateTrainer:
    """ 
    Class to generate data, train a NeuralRegressorNetwork, and save the weights.
    """
    def __init__(self,
            model_name,
            system_of_constraints,
            distribution_function,
            dimension,
            tol=0.1,
            mult=1.25,
            sample_size=100000,
            architecture=None,
            lr=0.003,
            max_epochs=2500,
            batch_size=30000,
            save_path=None,
            sample_multiplier=None):

        self.model_name = model_name
        self.system_of_constraints = system_of_constraints
        self.distribution_function = distribution_function
        self.dim = dimension
        self.tol = tol
        self.mult = mult
        self.sample_size = sample_size
        self.architecture = architecture if architecture is not None else ["dim", 100, 70, 30, 1]
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.save_path = save_path if save_path else f"{model_name}_surrogate.pth"
        self.sample_multiplier = sample_multiplier if sample_multiplier is not None else np.eye(self.dim)


    def generate_data(self, seed=420):
        print(f"[{self.model_name}] Generating training data...")
        np.random.seed(seed)
        
        # 1. Get the base distribution using the specified tolerance
        base_dist, _, _, _ = self.distribution_function(tol=self.tol, capa=1.0)
        
        # 2. Apply the multiplier to increase dispersion for more failure samples
        dist = otaf.distribution.multiply_composed_distribution_standard_with_constants(
            base_dist, [self.mult]*self.dim)
        
        # 3. Generate the input sample
        sample = np.array(dist.getSample(self.sample_size), dtype="float32")
        sample = sample @ np.array(self.sample_multiplier.T, dtype="float32") #Apply the variable change if needed

        # 4. Compute the results
        results = otaf.uncertainty.compute_gap_optimizations_on_sample_batch(
            self.system_of_constraints,
            sample,
            bounds=None,
            n_cpu=-2,
            progress_bar=True,
            batch_size=500,
            dtype="float32"
        )

        self.X_train = sample
        self.y_train = results[:, -1]
        
        failure_ratio = np.where(self.y_train < 0, 1, 0).sum() / self.sample_size
        print(f"[{self.model_name}] Ratio of failed simulations in sample: {failure_ratio:.5f}")

    def train_and_save(self):
        print(f"[{self.model_name}] Initializing neural network...")

        self.neural_model = otaf.surrogate.NeuralRegressorNetwork(
            input_dim=self.dim,
            output_dim=1,
            X=self.X_train,
            y=self.y_train,
            clamping=True,
            finish_critertion_epoch=5,
            loss_finish=1e-6,
            metric_finish=0.9998,
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            compile_model=False,
            train_size=0.6,
            save_path=self.save_path,
            display_progress_disable=False
        )

        # Parse architecture list, resolving "dim" to actual dimension
        parsed_arch = [self.dim if str(a).lower() == 'dim' else int(a) for a in self.architecture]

        self.neural_model.model = torch.nn.Sequential(
            *otaf.surrogate.get_custom_mlp_layers(parsed_arch, activation_class=torch.nn.GELU)
        )

        self.neural_model.optimizer = torch.optim.AdamW(
            self.neural_model.parameters(), lr=self.lr, weight_decay=1e-4
        )
        otaf.surrogate.initialize_model_weights(self.neural_model)
        self.neural_model.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.neural_model.optimizer, 1.0001
        )
        self.neural_model.loss_fn = torch.nn.MSELoss()

        print(f"[{self.model_name}] Training started...")
        self.neural_model.train_model()

        print(f"[{self.model_name}] Saving model and metadata to {self.save_path}...")
        
        # Build the checkpoint dictionary
        checkpoint = {
            'model_state_dict': self.neural_model.model.state_dict(),
            'architecture': self.architecture,
            'input_dim': self.dim,
            'model_name': self.model_name,
            # If your wrapper stores normalization buffers, include them:
            'normalization_metadata': {
                'x_mean': self.neural_model.x_mean if hasattr(self.neural_model, 'x_mean') else None,
                'x_std': self.neural_model.x_std if hasattr(self.neural_model, 'x_std') else None,
            }
        }
        
        torch.save(checkpoint, self.save_path)
        print(f"[{self.model_name}] Finished.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate data and train Neural Surrogates for OTAF models.")


    parser.add_argument("--models", nargs="+", default=["model1_4_dof", "model2_16_dof", "model3_30_dof", "model4_50_dof"], help="Models to train")
    parser.add_argument("--tols", nargs="+", type=float, help="Base tolerance for each model")
    parser.add_argument("--mults", nargs="+", type=float, help="Multiplicator for each model")
    parser.add_argument("--sample-sizes", nargs="+", type=int, help="Sample size for training data per model")
    parser.add_argument("--architectures", nargs="+", type=str, help="Architecture per model, comma-separated. E.g., 'dim,100,70,30,1'")
    parser.add_argument("--save-paths", nargs="+", type=str, help="Save path for the PyTorch state_dict per model")

    args = parser.parse_args()

    available_models = {
        "model1_4_dof": otaf.example_models.model1,
        "model2_16_dof": otaf.example_models.model2,
        "model3_30_dof": otaf.example_models.model3,
        "model4_50_dof": otaf.example_models.model4
    }

    # Helper function to map list arguments to the current model index
    def get_param(param_list, index, default):
        if not param_list:
            return default
        if len(param_list) == 1:
            return param_list[0]
        if index < len(param_list):
            return param_list[index]
        return default

    for i, model_name in enumerate(args.models):
        if model_name not in available_models:
            print(f"Error: Model '{model_name}' not found.")
            continue

        model_module = available_models[model_name]
        system_of_constraints = model_module.getSystemOfConstraintsAssemblyModel()
        distribution_function = model_module.getDistributionParams
        dimension = int(model_module.dim)

        tol = get_param(args.tols, i, 0.1)
        mult = get_param(args.mults, i, 1.25)
        sample_size = get_param(args.sample_sizes, i, 100000)
        arch_str = get_param(args.architectures, i, "dim,100,70,30,1")
        save_path = get_param(args.save_paths, i, f"{model_name}_surrogate.pth")

        architecture = arch_str.split(',')

        trainer = SurrogateTrainer(
            model_name=model_name,
            system_of_constraints=system_of_constraints,
            distribution_function=distribution_function,
            dimension=dimension,
            tol=tol,
            mult=mult,
            sample_size=sample_size,
            architecture=architecture,
            save_path=save_path,
            sample_multiplier=model_module.sample_multiplier if hasattr(model_module, 'sample_multiplier') else None
        )

        trainer.generate_data()
        trainer.train_and_save()
""" 
python train_neural_networks.py --tols 0.31 0.16 0.1 0.21 --mults 1.35 1.21 1.26 1.15 --sample-sizes 200000 --architectures 'dim,16,8,4,1' 'dim,32,16,8,1' 'dim,64,32,16,8,1' 'dim,128,64,32,16,1'

python train_neural_networks.py --models model1_4_dof model2_16_dof  --tols 0.31 0.16 --mults 1.35 1.21 --sample-sizes 200000 --architectures 'dim,16,8,4,1' 'dim,32,16,8,1'

REGARDING HOW TO CALL THE MODEL:

Your method of calling it inside `model2` logic is conceptually fine, but contains redundant PyTorch overhead.

Specifically, looking at the code inside `_neural_surrogate.py` for `evaluate_model_non_standard_space`:

1. It already wraps the inference loop in a `with torch.no_grad():` block.
2. The outputs do not have a gradient attached.
3. Unless you pass `return_on_gpu=True`, the method explicitly maps the output back to the CPU via `output.cpu()`.

Because of this, `detach()` is completely unnecessary. All you need to do is call `.numpy()` directly on the returned tensor.

Furthermore, make sure you don't rebuild the model unnecessarily at inference time. You should instantiate a dummy `NeuralRegressorNetwork` using a tiny subset of X/y just to store the buffer normalizations, call `load_model()` to import your `.pth` state dict, and then set it to `.eval()`.

The cleanest way to map it in production looks like this:

def model2(x, sample=sample_gld):
# Vectorized broadcasting assumption (sample: [N, D], x: [D])
x_matrix = sample * x[np.newaxis, :]


# Evaluate batch -> slice out the tensor -> convert to numpy array -> drop trivial dimensions
return np.squeeze(neural_model.evaluate_model_non_standard_space(x_matrix).numpy())
"""
