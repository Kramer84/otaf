import matplotlib
matplotlib.use('Agg')  # Force headless rendering to prevent GUI/backend hanging

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import matplotlib.patches as patches

# =========================================================================
# 1. GEOMETRY & DEFECT CONFIGURATION
# =========================================================================
# Outer nominal box: square from -3 to 3 (Counter-Clockwise: BL, BR, TR, TL)
outer_nominal = np.array([[-3.0, -3.0], [3.0, -3.0], [3.0, 3.0], [-3.0, 3.0]])

# Distinct manufacturing defects to ensure a visually dynamic optimization layout
outer_defects = np.array([
    [-0.1,  0.0],  # Bottom-Left corner shifted right/up
    [ -0.1,  0.1],  # Bottom-Right corner shifted right/up
    [ 0.2, -0.0],  # Top-Right corner shifted left/down
    [ 0.0, -0.0]   # Top-Left corner shifted right/down
])
outer_verts = outer_nominal + outer_defects

# Inner box nominal layout (slightly enlarged to tighten assembly clearances)
inner_nominal = np.array([[-2.0, -2.0], [2.0, -2.0], [2.0, 2.0], [-2.0, 2.0]])
inner_defects = np.array([
    [ 0.1, -0.1],
    [-0.1,  0.2],
    [ 0.2, -0.1],
    [-0.2, -0.2]
])
inner_verts_local = inner_nominal + inner_defects

# =========================================================================
# 2. KINEMATIC & CONSTRAINT FUNCTIONS
# =========================================================================
def get_inner_global(x, y, theta):
    """Transform local inner coordinates using translation and rotation parameters."""
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    return (inner_verts_local @ R.T) + np.array([x, y])

def compute_clearances(p):
    """Calculate the perpendicular distances from all inner points to all outer walls."""
    x, y, theta, s = p
    I_verts = get_inner_global(x, y, theta)
    distances = []
    projections = []
    
    # Step through each wall of the outer box configuration
    for i in range(4):
        A = outer_verts[i]
        B = outer_verts[(i + 1) % 4]
        dx, dy = B[0] - A[0], B[1] - A[1]
        L = np.hypot(dx, dy)
        inward_normal = np.array([-dy / L, dx / L])
        
        # Calculate distances and projection points for each inner vertex
        for j in range(4):
            P = I_verts[j]
            dist = np.dot(P - A, inward_normal)
            distances.append(dist)
            
            # Project point onto the infinite wall line
            t = np.dot(P - A, B - A) / (L * L)
            proj_point = A + t * (B - A)
            projections.append((P, proj_point, dist))
            
    return np.array(distances), I_verts, projections

def objective(p):
    return -p[3]  # Maximize the minimum clearance variable 's'

def constraint_engine(p):
    distances, _, _ = compute_clearances(p)
    return distances - p[3]  # Formulates as: distances >= s

# =========================================================================
# 3. SOLVER EXECUTION
# =========================================================================
# Initial guess optimization parameters: [x, y, theta, s]
initial_guess = [0.0, 0.0, 0.0, 0.1]
bound_constraints = {'type': 'ineq', 'fun': constraint_engine}

optimization_result = minimize(
    objective, 
    initial_guess, 
    method='SLSQP', 
    constraints=bound_constraints
)

# Extract optimized assembly states
opt_x, opt_y, opt_theta, opt_s = optimization_result.x
_, final_inner_verts, wall_projections = compute_clearances(optimization_result.x)

# =========================================================================
# 4. MATPLOTLIB LOGO RENDERING
# =========================================================================
# Configure professional open-science styling palette
color_outer = "#2B2D42"      # Dark Slate Gray
color_inner = "#1D3557"      # Deep Scientific Navy
color_fill  = "#BAF7BD"      # Crisp Light Background Slate
color_slack = "#DF3131"      # Vibrant Crimson Red for Active s-Values

if __name__ == "__main__":

    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    ax.set_aspect('equal')
    ax.axis('off')

    # Automatically identify and plot active binding slack vectors (s-value limits)
    for start_pt, end_pt, distance in wall_projections:
        if np.abs(distance - opt_s) < 1e-3:
            ax.annotate('', xy=end_pt, xytext=start_pt, zorder=1,
                        arrowprops=dict(arrowstyle="-|>", color=color_slack, 
                                        lw=3.5, mutation_scale=15, shrinkA=0.1, shrinkB=0))


    # Plot Defective Outer Envelope Boundary
    outer_polygon = patches.Polygon(outer_verts, closed=True, edgecolor=color_outer, 
                                    facecolor='none', linewidth=4, joinstyle='round')
    ax.add_patch(outer_polygon)

    # Plot Optimized Inner Part with a Solid Background Fill
    inner_polygon = patches.Polygon(final_inner_verts, closed=True, edgecolor=color_inner, 
                                    facecolor=color_fill, linewidth=4, joinstyle='round')
    ax.add_patch(inner_polygon)

    # Render centered framework branding text inside the optimized component coordinates
    ax.text(opt_x, opt_y - 0.05, 'otaf', color=color_inner, fontsize=54,
            fontfamily='sans-serif', fontweight='bold', ha='center', va='center')

    # Set tight spatial framing limits around the canvas
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)

    # Export clean, high-resolution distribution asset
    plt.savefig('logo.png', bbox_inches='tight', pad_inches=0.1, transparent=True)
    plt.close()