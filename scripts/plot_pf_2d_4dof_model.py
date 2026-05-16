import argparse
import numpy as np
import matplotlib.pyplot as plt
from gldpy import GLD
import otaf

# Import the model definition you provided
from otaf.example_models import model1

def get_model_evaluator(sample, mu_vect, neural_model):
    def evaluate(x):
        x_transformed = (sample - mu_vect) * x + mu_vect
        prediction = neural_model.evaluate_model_non_standard_space(x_transformed)
        return np.squeeze(prediction.numpy())
    return evaluate

def main():
    parser = argparse.ArgumentParser(description="Grid Evaluation for 4-DOF OTAF Model")
    parser.add_argument("--surrogate-path", type=str, default="./model1_4_dof_surrogate.pth", help="Path to the .pth surrogate file")
    parser.add_argument("--slack", type=float, default=0.0, help="Failure slack threshold")
    parser.add_argument("--grid-res", type=int, default=20, help="Resolution of the 2D grid")
    parser.add_argument("--mc-size", type=int, default=100000, help="Monte Carlo sample size")
    args = parser.parse_args()

    # 1. Setup Models and Distributions
    print("Loading surrogate model and distributions...")
    model_sur = otaf.surrogate.NeuralRegressorNetwork.from_checkpoint(args.surrogate_path)
    
    jointDist, symbols, max_std_vect, mu_vect = model1.getDistributionParams()
    sample_gld = np.array(jointDist.getSample(args.mc_size))
    
    model_eval_fn = get_model_evaluator(sample_gld, mu_vect, model_sur)
    gld = GLD('VSL')

    # 2. Construct the Grid
    print(f"Building {args.grid_res}x{args.grid_res} evaluation grid...")
    u4_scales = np.linspace(0, 1, args.grid_res)
    u5_scales = np.linspace(0, 1, args.grid_res)
    
    U4_grid, U5_grid = np.meshgrid(u4_scales, u5_scales)
    PF_grid = np.zeros_like(U4_grid)

    # 3. Evaluate Each Point on the Grid
    for i in range(args.grid_res):
        for j in range(args.grid_res):
            u4 = U4_grid[i, j]
            u5 = U5_grid[i, j]
            
            # Analytically determine the boundary gamma scales
            gamma4 = np.sqrt(1.0 - u4**2)
            gamma5 = np.sqrt(1.0 - u5**2)
            
            x_scaled = np.array([u4, gamma4, u5, gamma5])
            
            # Evaluate slack via surrogate
            slack = model_eval_fn(x_scaled)
            
            # Compute Pf using GLD
            gld_params = gld.fit_LMM(slack, disp_fit=False, disp_optimizer=False)
            
            if np.any(np.isnan(gld_params)):
                # Fallback to empirical probability if GLD fails
                pf = np.where(slack < args.slack, 1, 0).mean()
            else:
                pf = gld.CDF_num(args.slack, gld_params, xtol=1e-6)
                
            PF_grid[i, j] = pf
            
        print(f"Row {i+1}/{args.grid_res} completed.")

    # 4. Plot the Results
    print("Generating contour plot...")
    plt.figure(figsize=(10, 8))
    
    # We plot the Log probability (or standard) depending on preference. 
    # Log10 is often better for Pf visualization if it spans multiple magnitudes.
    # Here we plot the raw Pf, but log-scaled contour levels.
    
    contour = plt.contourf(U4_grid, U5_grid, PF_grid, levels=50, cmap='viridis')
    plt.colorbar(contour, label=f'Probability of Failure (Pf) [Slack = {args.slack}]')
    
    plt.title('Failure Probability across Translation/Rotation Allocations')
    plt.xlabel('u_d_4 Allocation Scale (0=Max Rotation, 1=Max Translation)')
    plt.ylabel('u_d_5 Allocation Scale (0=Max Rotation, 1=Max Translation)')
    
    # Save and show
    plot_filename = f"Pf_Grid_slack_{args.slack}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully as {plot_filename}")

if __name__ == "__main__":
    main()