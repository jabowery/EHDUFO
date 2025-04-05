import logging
import numba
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon
import time
import os
import shutil
tightness = None

# Create output directory
if not os.path.exists('output'):
    os.makedirs('output')

# Create frames directory structure
frames_dir = 'output/frames'
if os.path.exists(frames_dir):
    # Clean up existing frames if present
    shutil.rmtree(frames_dir)
os.makedirs(frames_dir)

# Create subdirectories for each plot type
plot_types = ['charge_density', 'air_flow', 'streamlines']
for plot_type in plot_types:
    os.makedirs(os.path.join(frames_dir, plot_type))

# Set up matplotlib for better visualization
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 10)
plt.rcParams['figure.dpi'] = 100

logging.debug("Initializing EHD simulation...")

# Physical Constants (using scaled values to improve numerical stability)
epsilon_0 = 8.85e-12     # Vacuum permittivity [F/m]
rho_fluid = 1.225        # Air density [kg/m^3]
mu = 1.81e-5             # Dynamic viscosity of air [Pa·s]
K = 1.8e-4               # Ion mobility in air [m^2/(V·s)]
diff_coef = 5.3e-5       # Diffusion coefficient of ions in air [m^2/s]

# Scale factors to improve numerical stability
voltage_scale = 1000.0   # Scale voltages by 1000V
charge_scale = 1.0e-3    # Scale charge density by 1e-3 C/m³

# Simulation Domain
width = 0.1              # Domain width [m]
height = 0.2             # Domain height [m]
def plot_bounds():
    figsize = (14, 14)
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)  # Margins
    ax.set_xlim(0, width)  # Fixed axis limits
    ax.set_ylim(0, height)
    return fig, ax

# Simulation Domain
center_x = width/2       # 0.05 m
center_y = height/2      # 0.10 m

# Electrode Configuration
# Emitter centered horizontally, positioned near top
emitter_x = center_x                   # X-center of emitter [m]
emitter_y = center_y + 0.06            # 0.16 m (original position)
emitter_radius = 0.005                 # Radius of emitter [m]
emitter_voltage = -5.0                 # Emitter voltage [kV] (scaled)

# Collection plate (screen) - centered horizontally, positioned below center
screen_y = center_y - 0.03             # 0.07 m (original position)
screen_start_x = center_x - 0.02       # 0.03 m (original position)
screen_end_x = center_x + 0.02         # 0.07 m (original position)
screen_voltage = 5.0                   # Screen voltage [kV] (scaled)

# Triangle obstruction - centered horizontally, positioned below emitter
triangle_x1, triangle_y1 = center_x, center_y + 0.02   # Apex (0.05, 0.12)
triangle_x2, triangle_y2 = center_x - 0.02, center_y - 0.02  # Bottom left (0.03, 0.08)
triangle_x3, triangle_y3 = center_x + 0.02, center_y - 0.02  # Bottom right (0.07, 0.08)

# Numerical Parameters
nx = 100                 # Number of points in x-direction
ny = 200                 # Number of points in y-direction
min_threshold = 1e-30    # Minimum threshold for numerical values

# Create grid
x = np.linspace(0, width, nx+1)
y = np.linspace(0, height, ny+1)
X, Y = np.meshgrid(x, y)
dx = width / nx
dy = height / ny

# Function to check if a point is inside the triangle
def in_triangle(x, y, x1, y1, x2, y2, x3, y3):
    def sign(p1x, p1y, p2x, p2y, p3x, p3y):
        return (p1x - p3x) * (p2y - p3y) - (p2x - p3x) * (p1y - p3y)
    
    d1 = sign(x, y, x1, y1, x2, y2)
    d2 = sign(x, y, x2, y2, x3, y3)
    d3 = sign(x, y, x3, y3, x1, y1)
    
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    
    return not (has_neg and has_pos)

# Initialize domain mask (0 = fluid, 1 = solid/obstacle)
domain_mask = np.zeros((ny+1, nx+1))
logging.debug("Setting up domain geometry...")

# Set up triangle obstruction in the mask
for j in range(ny+1):
    y_pos = j * height / ny
    for i in range(nx+1):
        x_pos = i * width / nx
        if in_triangle(x_pos, y_pos, triangle_x1, triangle_y1, triangle_x2, triangle_y2, triangle_x3, triangle_y3):
            domain_mask[j, i] = 1

# Initialize the potential array
potential = np.zeros((ny+1, nx+1))

# Set boundary conditions for the potential
# Top, bottom, left, right boundaries
potential[0, :] = 0      # Bottom boundary
potential[ny, :] = 0     # Top boundary
potential[:, 0] = 0      # Left boundary
potential[:, nx] = 0     # Right boundary

# Partial-width screen at specified height
screen_start_index = int(screen_start_x / width * nx)
screen_end_index = int(screen_end_x / width * nx)
screen_height_index = int(screen_y / height * ny)

# Set the screen voltage
potential[screen_height_index-1:screen_height_index+2, screen_start_index:screen_end_index+1] = screen_voltage

# Emitter (point near top)
emitter_index_x = int(emitter_x / width * nx)
emitter_index_y = int(emitter_y / height * ny)
emitter_radius_cells = int(emitter_radius / width * nx)

for j in range(max(0, emitter_index_y-emitter_radius_cells), min(ny+1, emitter_index_y+emitter_radius_cells+1)):
    for i in range(max(0, emitter_index_x-emitter_radius_cells), min(nx+1, emitter_index_x+emitter_radius_cells+1)):
        if ((i - emitter_index_x)**2 + (j - emitter_index_y)**2 <= emitter_radius_cells**2):
            potential[j, i] = emitter_voltage

# Create mask for boundary conditions and obstacles
bc_mask = np.zeros((ny+1, nx+1), dtype=bool)

# Set fixed-value points in the mask
bc_mask[0, :] = True      # Bottom boundary
bc_mask[ny, :] = True     # Top boundary
bc_mask[:, 0] = True      # Left boundary
bc_mask[:, nx] = True     # Right boundary
bc_mask[screen_height_index-1:screen_height_index+2, screen_start_index:screen_end_index+1] = True  # Screen
bc_mask[domain_mask == 1] = True  # Triangle

# Also mark the emitter in the mask
for j in range(max(0, emitter_index_y-emitter_radius_cells), min(ny+1, emitter_index_y+emitter_radius_cells+1)):
    for i in range(max(0, emitter_index_x-emitter_radius_cells), min(nx+1, emitter_index_x+emitter_radius_cells+1)):
        if ((i - emitter_index_x)**2 + (j - emitter_index_y)**2 <= emitter_radius_cells**2):
            bc_mask[j, i] = True

logging.debug("Solving electric potential...")
# Solve Laplace's equation using Jacobi iteration for better stability
max_iterations = 50000  # Increased max iterations
tolerance = 1.0e-6
omega = 1.5  # Reduced for stability

# Make a copy of the initial potential for reference
initial_potential = potential.copy()

# Helper function to check for invalid values
def has_invalid(arr):
    return np.any(np.isnan(arr)) or np.any(np.isinf(arr))

# Track convergence
residuals = []


@numba.njit(parallel=True)
def jacobi_update(potential, bc_mask, nx, ny):
    new_potential = potential.copy()
    for j in numba.prange(1, ny):
        for i in numba.prange(1, nx):
            if not bc_mask[j, i]:
                new_potential[j, i] = 0.25 * (
                    potential[j, i+1] + potential[j, i-1] +
                    potential[j+1, i] + potential[j-1, i]
                )
    return new_potential

for iter in range(max_iterations):
    # Create a copy of the current solution
    """
    new_potential = potential.copy()
    
    # Update interior points using Jacobi iteration (more stable than SOR for this problem)
    for j in range(1, ny):
        for i in range(1, nx):
            if bc_mask[j, i]:
                continue
            
            # Standard 5-point stencil for Laplace equation
            new_potential[j, i] = 0.25 * (
                potential[j, i+1] + potential[j, i-1] + 
                potential[j+1, i] + potential[j-1, i])
    """
    new_potential = jacobi_update(potential, bc_mask, nx, ny)
    
    # Maintain boundary conditions
    new_potential[bc_mask] = initial_potential[bc_mask]
    
    # Check for invalid values
    if has_invalid(new_potential):
        logging.debug(f"Warning: Invalid values detected at iteration {iter}, reverting to simple averaging")
        # Fall back to simple averaging
        for j in range(1, ny):
            for i in range(1, nx):
                if bc_mask[j, i]:
                    continue
                # Simple averaging of neighbors
                valid_neighbors = []
                if not np.isnan(potential[j, i+1]) and not np.isinf(potential[j, i+1]):
                    valid_neighbors.append(potential[j, i+1])
                if not np.isnan(potential[j, i-1]) and not np.isinf(potential[j, i-1]):
                    valid_neighbors.append(potential[j, i-1])
                if not np.isnan(potential[j+1, i]) and not np.isinf(potential[j+1, i]):
                    valid_neighbors.append(potential[j+1, i])
                if not np.isnan(potential[j-1, i]) and not np.isinf(potential[j-1, i]):
                    valid_neighbors.append(potential[j-1, i])
                
                if valid_neighbors:
                    new_potential[j, i] = np.mean(valid_neighbors)
                else:
                    new_potential[j, i] = 0  # Reset to zero if no valid neighbors
    
    # Calculate max change and residual
    diff = new_potential - potential
    max_change = np.max(np.abs(diff[~bc_mask]))
    residual = np.linalg.norm(diff[~bc_mask])
    residuals.append(residual)
    
    # Update the potential
    potential = new_potential.copy()
    
    # Print progress every 1000 iterations
    if iter % 1000 == 0:
        logging.debug(f"Iteration {iter}, max change: {max_change:.8f}, residual: {residual:.8f}")
    
    # Check convergence
    if max_change < tolerance:
        logging.debug(f"Electric potential solution converged after {iter+1} iterations")
        break

if iter == max_iterations - 1:
    logging.debug("Warning: Maximum iterations reached without convergence")

# Plot convergence history
plt.figure(figsize=(10, 6))
plt.semilogy(residuals)
plt.xlabel('Iteration')
plt.ylabel('Residual')
plt.title('Convergence History')
plt.grid(True)
plt.savefig('output/convergence.png', dpi=150, bbox_inches=tightness)
plt.close()

# Calculate electric field using central differences
logging.debug("Calculating electric field...")
E_x = np.zeros((ny+1, nx+1))
E_y = np.zeros((ny+1, nx+1))

for j in range(1, ny):
    for i in range(1, nx):
        if domain_mask[j, i] == 1:
            continue
        
        # Negative gradient of potential (scaled back to real units)
        E_x[j, i] = -(potential[j, i+1] - potential[j, i-1]) / (2 * dx) * voltage_scale
        E_y[j, i] = -(potential[j+1, i] - potential[j-1, i]) / (2 * dy) * voltage_scale

# Initialize charge density
charge_density = np.zeros((ny+1, nx+1))

# Set initial charge at emitter (using scaled value)
emitter_charge_density = -1.0  # Scaled units
for j in range(max(0, emitter_index_y-emitter_radius_cells), min(ny+1, emitter_index_y+emitter_radius_cells+1)):
    for i in range(max(0, emitter_index_x-emitter_radius_cells), min(nx+1, emitter_index_x+emitter_radius_cells+1)):
        if ((i - emitter_index_x)**2 + (j - emitter_index_y)**2 <= emitter_radius_cells**2):
            charge_density[j, i] = emitter_charge_density

# Initialize fluid velocity fields
fluid_velocity_x = np.zeros((ny+1, nx+1))
fluid_velocity_y = np.zeros((ny+1, nx+1))

# Initialize momentum tracking
fluid_momentum_x = np.zeros((ny+1, nx+1))
fluid_momentum_y = np.zeros((ny+1, nx+1))

# Time-stepping parameters
dt = 0.0001  # Time step [s]
total_time = 0.1  # Total simulation time [s]
num_time_steps = int(total_time / dt)

# Increase number of save points for smoother animations
num_frames = 50
save_interval = max(1, num_time_steps // num_frames)
save_times = [i * save_interval for i in range(num_frames + 1)]

# Storage for history
charge_history = [charge_density.copy()]
velocity_history = [(fluid_velocity_x.copy(), fluid_velocity_y.copy())]
momentum_history = [(fluid_momentum_x.copy(), fluid_momentum_y.copy())]

logging.debug(f"Starting time-stepping simulation ({num_time_steps} steps)...")
start_time = time.time()

# Time-stepping simulation
for time_step in range(1, num_time_steps + 1):
    # Progress indicator
    if time_step % (num_time_steps // 10) == 0:
        progress = time_step / num_time_steps * 100
        elapsed = time.time() - start_time
        eta = (elapsed / time_step) * (num_time_steps - time_step)
        logging.debug(f"Progress: {progress:.1f}%, ETA: {eta:.1f}s")

    @numba.njit
    def clip(value, lower, upper):
        """
        Numba-compatible scalar clip function.
        """
        return max(lower, min(value, upper))

    @numba.njit
    def has_invalid(arr):
        return np.any(np.isnan(arr)) or np.any(np.isinf(arr))

    @numba.njit(parallel=True)
    def update_grid(charge_density, fluid_velocity_x, fluid_velocity_y, E_x, E_y, bc_mask,
                domain_mask, K, diff_coef, dx, dy, dt, min_threshold, rho_fluid,
                    charge_scale, fluid_momentum_x, fluid_momentum_y, nx, ny):
        # Initialize new arrays
        new_charge = np.zeros((ny+1, nx+1))
        new_velocity_x = np.zeros((ny+1, nx+1))
        new_velocity_y = np.zeros((ny+1, nx+1))
        
        # Apply transport equation at each interior point
        for j in numba.prange(1, ny):
            for i in numba.prange(1, nx):
                # Skip points where boundary conditions are applied or inside triangle
                if bc_mask[j, i]:
                    continue
                
                # Local electric field
                Ex = E_x[j, i]
                Ey = E_y[j, i]
                
                # Current charge - apply threshold to avoid underflow
                current_charge = charge_density[j, i]
                if abs(current_charge) < min_threshold:
                    current_charge = 0
                
                # Calculate fluxes - simplified drift diffusion
                # Drift term (K is mobility, E is electric field)
                drift_x = K * Ex * current_charge
                drift_y = K * Ey * current_charge
                
                # Diffusion term - use central differences
                diff_x = diff_coef * (charge_density[j, i+1] - 2*current_charge + charge_density[j, i-1]) / dx**2
                diff_y = diff_coef * (charge_density[j+1, i] - 2*current_charge + charge_density[j-1, i]) / dy**2
                
                # Update charge density - upwind scheme for stability
                charge_update = dt * (
                    # Convection - upwind differencing
                    -max(drift_x, 0) * (current_charge - charge_density[j, i-1]) / dx -
                    min(drift_x, 0) * (charge_density[j, i+1] - current_charge) / dx -
                    max(drift_y, 0) * (current_charge - charge_density[j-1, i]) / dy -
                    min(drift_y, 0) * (charge_density[j+1, i] - current_charge) / dy +
                    # Diffusion
                    diff_x + diff_y
                )
                
                # Limit change for stability
                max_allowed_change = abs(current_charge) * 0.1 + 1e-10
                charge_update = clip(charge_update, -max_allowed_change, max_allowed_change)  # Use the custom clip function
                new_charge[j, i] = current_charge + charge_update
                
                # Apply threshold to avoid underflow
                if abs(new_charge[j, i]) < min_threshold:
                    new_charge[j, i] = 0.0
                
                # Calculate EHD body force
                # Actual charge density is charge_density * charge_scale
                force_x = current_charge * charge_scale * Ex
                force_y = current_charge * charge_scale * Ey
                
                # Track momentum transfer to fluid
                fluid_momentum_x[j, i] += dt * force_x
                fluid_momentum_y[j, i] += dt * force_y
                
                # Update fluid velocity using a simplified model
                # F = ma -> a = F/ρ, v = v0 + at
                new_velocity_x[j, i] = fluid_velocity_x[j, i] + dt * force_x / rho_fluid
                new_velocity_y[j, i] = fluid_velocity_y[j, i] + dt * force_y / rho_fluid
        
        # Apply boundary conditions
        # Emitter
        for j in numba.prange(max(0, emitter_index_y-emitter_radius_cells), min(ny+1, emitter_index_y+emitter_radius_cells+1)):
            for i in numba.prange(max(0, emitter_index_x-emitter_radius_cells), min(nx+1, emitter_index_x+emitter_radius_cells+1)):
                if ((i - emitter_index_x)**2 + (j - emitter_index_y)**2 <= emitter_radius_cells**2):
                    new_charge[j, i] = emitter_charge_density
                    new_velocity_x[j, i] = 0.0
                    new_velocity_y[j, i] = 0.0
        
        # Screen - neutralize charges
        new_charge[screen_height_index-1:screen_height_index+2, screen_start_index:screen_end_index+1] = 0.0
        new_velocity_x[screen_height_index-1:screen_height_index+2, screen_start_index:screen_end_index+1] = 0.0
        new_velocity_y[screen_height_index-1:screen_height_index+2, screen_start_index:screen_end_index+1] = 0.0
        #Inside triangle - no charges or flow (Method 2: Integer indexing - use if Method 1 fails)
        indices = np.where(domain_mask == 1)
        for i in numba.prange(len(indices[0])):
            new_charge[indices[0][i], indices[1][i]] = 0.0
            new_velocity_x[indices[0][i], indices[1][i]] = 0.0
            new_velocity_y[indices[0][i], indices[1][i]] = 0.0

#        # Inside triangle - no charges or flow
#        new_charge[domain_mask == 1] = 0.0
#        new_velocity_x[domain_mask == 1] = 0.0
#        new_velocity_y[domain_mask == 1] = 0.0
        
        # Apply simplified viscous diffusion for velocity
        viscosity_factor = 0.1  # Simplification parameter
        
        # Create temporary arrays for viscous diffusion
        temp_vx = new_velocity_x.copy()
        temp_vy = new_velocity_y.copy()
        
        for j in numba.prange(1, ny):
            for i in numba.prange(1, nx):
                # Skip obstacles and boundaries
                if bc_mask[j, i]:
                    continue
                
                # Simplified viscous diffusion - average with neighbors
                valid_neighbors_x = []
                valid_neighbors_y = []
                
                for ni, nj in [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]:
                    if 0 <= ni <= nx and 0 <= nj <= ny and not bc_mask[nj, ni]:
                        valid_neighbors_x.append(temp_vx[nj, ni])
                        valid_neighbors_y.append(temp_vy[nj, ni])
                # Inside the update_grid function's viscous diffusion loop:
                if valid_neighbors_x:
                    # Calculate mean manually for Numba compatibility
                    sum_x = sum(valid_neighbors_x)
                    mean_x = sum_x / len(valid_neighbors_x)
                    sum_y = sum(valid_neighbors_y)
                    mean_y = sum_y / len(valid_neighbors_y)
                    
                    new_velocity_x[j, i] = (1-viscosity_factor) * temp_vx[j, i] + viscosity_factor * mean_x
                    new_velocity_y[j, i] = (1-viscosity_factor) * temp_vy[j, i] + viscosity_factor * mean_y

        # Check for invalid values and clean them up
        if has_invalid(new_charge):
            logging.debug("Warning: Invalid charge values detected, cleaning up...")
            # With this Numba-compatible version:
            for j in numba.prange(new_charge.shape[0]):
                for i in numba.prange(new_charge.shape[1]):
                    if np.isnan(new_charge[j,i]) or np.isinf(new_charge[j,i]):
                        new_charge[j,i] = 0.0

        
        if has_invalid(new_velocity_x) or has_invalid(new_velocity_y):
            for j in numba.prange(new_velocity_x.shape[0]):
                for i in numba.prange(new_velocity_x.shape[1]):
                    if np.isnan(new_velocity_x[j,i]) or np.isinf(new_velocity_x[j,i]):
                        new_velocity_x[j,i] = 0.0
                    if np.isnan(new_velocity_y[j,i]) or np.isinf(new_velocity_y[j,i]):
                        new_velocity_y[j,i] = 0.0

        # Update charge density and fluid velocity
        charge_density = new_charge.copy()
        fluid_velocity_x = new_velocity_x.copy()
        fluid_velocity_y = new_velocity_y.copy()
        return charge_density, fluid_velocity_x, fluid_velocity_y, E_x, E_y, bc_mask, domain_mask, K, diff_coef, dx, dy, dt, min_threshold, rho_fluid, charge_scale, fluid_momentum_x, fluid_momentum_y, nx, ny
#        return domain_mask, charge_density, fluid_velocity_x, fluid_velocity_y, E_x, E_y, bc_mask, K, diff_coef, dx, dy, dt, min_threshold 

    charge_density, fluid_velocity_x, fluid_velocity_y, E_x, E_y, bc_mask, domain_mask, K, diff_coef, dx, dy, dt, min_threshold, rho_fluid, charge_scale, fluid_momentum_x, fluid_momentum_y, nx, ny = update_grid(charge_density, fluid_velocity_x, fluid_velocity_y, E_x, E_y, bc_mask, domain_mask, K, diff_coef, dx, dy, dt, min_threshold, rho_fluid, charge_scale, fluid_momentum_x, fluid_momentum_y, nx, ny)

    # Save state at specified times
    if time_step in save_times:
        frame_index = save_times.index(time_step)
        current_time = time_step * dt
        logging.debug(f"Saving frame {frame_index} at t = {current_time:.4f} s")
        
        # Save to histories for final processing
        charge_history.append(charge_density.copy())
        velocity_history.append((fluid_velocity_x.copy(), fluid_velocity_y.copy()))
        momentum_history.append((fluid_momentum_x.copy(), fluid_momentum_y.copy()))
        
        # Save frames to their respective directories
        # 1. Charge Density
        #fig, ax = plt.subplots(figsize=(12, 10))
        fig, ax = plot_bounds()
        contour = ax.contourf(X, Y, charge_density * charge_scale, 20, cmap='coolwarm')
        fig.colorbar(contour, ax=ax, label='Charge Density (C/m³)')
        circle = Circle((emitter_x, emitter_y), emitter_radius, color='blue', alpha=0.7)
        ax.add_patch(circle)
        rect = Rectangle((screen_start_x, screen_y-0.003), screen_end_x - screen_start_x, 0.006, 
                        color='red', alpha=0.7)
        ax.add_patch(rect)
        triangle = Polygon([(triangle_x1, triangle_y1), (triangle_x2, triangle_y2), (triangle_x3, triangle_y3)], 
                          closed=True, color='gray', alpha=0.5)
        ax.add_patch(triangle)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title(f'Charge Density at t = {current_time:.4f} s')
        ax.set_aspect('equal')
        plt.savefig(f'{frames_dir}/charge_density/frame_{frame_index:04d}.png', dpi=100, bbox_inches=tightness)
        plt.close(fig)
        
        # 2. Air Flow (Vectors)
        #fig, ax = plt.subplots(figsize=(12, 10))
        fig, ax = plot_bounds()
        v_mag = np.sqrt(fluid_velocity_x**2 + fluid_velocity_y**2)
        skip = 5
        quiver = ax.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                          fluid_velocity_x[::skip, ::skip], fluid_velocity_y[::skip, ::skip], 
                          v_mag[::skip, ::skip],
                          scale=0.5, cmap='plasma')
        fig.colorbar(quiver, ax=ax, label='Velocity Magnitude (m/s)')
        circle = Circle((emitter_x, emitter_y), emitter_radius, color='blue', alpha=0.7)
        ax.add_patch(circle)
        rect = Rectangle((screen_start_x, screen_y-0.003), screen_end_x - screen_start_x, 0.006, 
                        color='red', alpha=0.7)
        ax.add_patch(rect)
        triangle = Polygon([(triangle_x1, triangle_y1), (triangle_x2, triangle_y2), (triangle_x3, triangle_y3)], 
                          closed=True, color='gray', alpha=0.5)
        ax.add_patch(triangle)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title(f'Air Flow Field at t = {current_time:.4f} s')
        ax.set_aspect('equal')
        plt.savefig(f'{frames_dir}/air_flow/frame_{frame_index:04d}.png', dpi=100, bbox_inches=tightness)
        plt.close(fig)
        
        # 3. Streamlines
        #fig, ax = plt.subplots(figsize=(12, 10))
        fig, ax = plot_bounds()
        mask = ~bc_mask
        vx_masked = np.where(mask, fluid_velocity_x, 0)
        vy_masked = np.where(mask, fluid_velocity_y, 0)
        x_1d = np.linspace(0, width, nx+1)
        y_1d = np.linspace(0, height, ny+1)
        v_mag = np.sqrt(vx_masked**2 + vy_masked**2)
        strm = ax.streamplot(x_1d, y_1d, vx_masked, vy_masked, 
                            density=1.5, color=v_mag, cmap='plasma',
                            linewidth=1, arrowsize=1.2)
        fig.colorbar(strm.lines, ax=ax, label='Velocity Magnitude (m/s)')
        circle = Circle((emitter_x, emitter_y), emitter_radius, color='blue', alpha=0.7)
        ax.add_patch(circle)
        rect = Rectangle((screen_start_x, screen_y-0.003), screen_end_x - screen_start_x, 0.006, 
                        color='red', alpha=0.7)
        ax.add_patch(rect)
        triangle = Polygon([(triangle_x1, triangle_y1), (triangle_x2, triangle_y2), (triangle_x3, triangle_y3)], 
                          closed=True, color='gray', alpha=0.5)
        ax.add_patch(triangle)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title(f'Air Flow Streamlines at t = {current_time:.4f} s')
        ax.set_aspect('equal')
        plt.savefig(f'{frames_dir}/streamlines/frame_{frame_index:04d}.png', dpi=100, bbox_inches=tightness)
        plt.close(fig)

logging.debug(f"Time-stepping completed in {time.time() - start_time:.2f} seconds")

# Calculate total thrust over time
def calculate_thrust(momentum_x, momentum_y):
    # Use nansum to handle potential NaN values
    total_x = np.nansum(momentum_x)
    total_y = np.nansum(momentum_y)
    return np.sqrt(total_x**2 + total_y**2)

thrust_over_time = [(i * total_time / len(momentum_history), 
                    calculate_thrust(momentum_history[i][0], momentum_history[i][1])) 
                   for i in range(len(momentum_history))]

# Plotting functions
def plot_domain():
    #fig, ax = plt.subplots(figsize=(8, 14))
    fig, ax = plot_bounds()
    
    # Plot domain mask
    im = ax.imshow(domain_mask, extent=[0, width, 0, height], origin='lower', 
                  cmap='gray', alpha=0.3, vmin=0, vmax=1)
    
    # Add electrodes and obstacles
    # Emitter
    circle = Circle((emitter_x, emitter_y), emitter_radius, color='blue', alpha=0.7)
    ax.add_patch(circle)
    
    # Screen
    rect = Rectangle((screen_start_x, screen_y-0.003), screen_end_x - screen_start_x, 0.006, 
                    color='red', alpha=0.7)
    ax.add_patch(rect)
    
    # Triangle
    triangle = Polygon([(triangle_x1, triangle_y1), (triangle_x2, triangle_y2), (triangle_x3, triangle_y3)], 
                      closed=True, color='gray', alpha=0.5)
    ax.add_patch(triangle)
    
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('EHD Thruster Configuration')
    ax.set_aspect('equal')
    
    return fig, ax

def plot_potential():
    #fig, ax = plt.subplots(figsize=(8, 14))
    fig, ax = plot_bounds()
    
    # Plot potential contours (scale back to real units)
    contour = ax.contourf(X, Y, potential * voltage_scale, 20, cmap='coolwarm')
    fig.colorbar(contour, ax=ax, label='Electric Potential (V)')
    
    # Add electrodes and obstacles
    # Emitter
    circle = Circle((emitter_x, emitter_y), emitter_radius, color='blue', alpha=0.7)
    ax.add_patch(circle)
    
    # Screen
    rect = Rectangle((screen_start_x, screen_y-0.003), screen_end_x - screen_start_x, 0.006, 
                    color='red', alpha=0.7)
    ax.add_patch(rect)
    
    # Triangle
    triangle = Polygon([(triangle_x1, triangle_y1), (triangle_x2, triangle_y2), (triangle_x3, triangle_y3)], 
                      closed=True, color='gray', alpha=0.5)
    ax.add_patch(triangle)
    
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Electric Potential (V)')
    ax.set_aspect('equal')
    
    return fig, ax

def plot_electric_field():
    #fig, ax = plt.subplots(figsize=(8, 14))
    fig, ax = plot_bounds()
    
    # Create magnitude array for coloring
    E_mag = np.sqrt(E_x**2 + E_y**2)
    
    # Plot electric field vectors
    skip = 5
    quiver = ax.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                      E_x[::skip, ::skip], E_y[::skip, ::skip], 
                      E_mag[::skip, ::skip],
                      scale=5e4, cmap='viridis')
    fig.colorbar(quiver, ax=ax, label='Electric Field Magnitude (V/m)')
    
    # Add electrodes and obstacles
    # Emitter
    circle = Circle((emitter_x, emitter_y), emitter_radius, color='blue', alpha=0.7)
    ax.add_patch(circle)
    
    # Screen
    rect = Rectangle((screen_start_x, screen_y-0.003), screen_end_x - screen_start_x, 0.006, 
                    color='red', alpha=0.7)
    ax.add_patch(rect)
    
    # Triangle
    triangle = Polygon([(triangle_x1, triangle_y1), (triangle_x2, triangle_y2), (triangle_x3, triangle_y3)], 
                      closed=True, color='gray', alpha=0.5)
    ax.add_patch(triangle)
    
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Electric Field')
    ax.set_aspect('equal')
    
    return fig, ax

def plot_charge_density(time_index=-1):
    #fig, ax = plt.subplots(figsize=(8, 14))
    fig, ax = plot_bounds()
    
    # Plot charge density (scale back to real units)
    cd = charge_history[time_index] * charge_scale
    contour = ax.contourf(X, Y, cd, 20, cmap='coolwarm')
    fig.colorbar(contour, ax=ax, label='Charge Density (C/m³)')
    
    # Add electrodes and obstacles
    # Emitter
    circle = Circle((emitter_x, emitter_y), emitter_radius, color='blue', alpha=0.7)
    ax.add_patch(circle)
    
    # Screen
    rect = Rectangle((screen_start_x, screen_y-0.003), screen_end_x - screen_start_x, 0.006, 
                    color='red', alpha=0.7)
    ax.add_patch(rect)
    
    # Triangle
    triangle = Polygon([(triangle_x1, triangle_y1), (triangle_x2, triangle_y2), (triangle_x3, triangle_y3)], 
                      closed=True, color='gray', alpha=0.5)
    ax.add_patch(triangle)
    
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    time_value = time_index * total_time / (len(charge_history) - 1) if time_index != -1 else total_time
    ax.set_title(f'Charge Density at t = {time_value:.3f} s')
    ax.set_aspect('equal')
    
    return fig, ax

def plot_air_flow(time_index=-1):
    #fig, ax = plt.subplots(figsize=(8, 14))
    fig, ax = plot_bounds()
    
    # Get velocity components
    vx, vy = velocity_history[time_index]
    
    # Calculate magnitude for color mapping
    v_mag = np.sqrt(vx**2 + vy**2)
    
    # Plot velocity vectors - subsample for clarity
    skip = 5
    quiver = ax.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                      vx[::skip, ::skip], vy[::skip, ::skip], 
                      v_mag[::skip, ::skip],
                      scale=0.5, cmap='plasma')
    fig.colorbar(quiver, ax=ax, label='Velocity Magnitude (m/s)')
    
    # Add electrodes and obstacles
    # Emitter
    circle = Circle((emitter_x, emitter_y), emitter_radius, color='blue', alpha=0.7)
    ax.add_patch(circle)
    
    # Screen
    rect = Rectangle((screen_start_x, screen_y-0.003), screen_end_x - screen_start_x, 0.006, 
                    color='red', alpha=0.7)
    ax.add_patch(rect)
    
    # Triangle
    triangle = Polygon([(triangle_x1, triangle_y1), (triangle_x2, triangle_y2), (triangle_x3, triangle_y3)], 
                      closed=True, color='gray', alpha=0.5)
    ax.add_patch(triangle)
    
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    time_value = time_index * total_time / (len(velocity_history) - 1) if time_index != -1 else total_time
    ax.set_title(f'Air Flow Field at t = {time_value:.3f} s')
    ax.set_aspect('equal')
    
    return fig, ax

def plot_streamlines(time_index=-1):
    #fig, ax = plt.subplots(figsize=(8, 14))
    fig, ax = plot_bounds()
    
    # Get velocity components
    vx, vy = velocity_history[time_index]
    
    # Mask values inside obstacles
    mask = ~bc_mask
    vx_masked = np.where(mask, vx, 0)
    vy_masked = np.where(mask, vy, 0)
    
    # Use 1D arrays for streamplot
    x_1d = np.linspace(0, width, nx+1)
    y_1d = np.linspace(0, height, ny+1)
    
    # Calculate velocity magnitude for coloring
    v_mag = np.sqrt(vx_masked**2 + vy_masked**2)
    
    # Plot streamlines
    strm = ax.streamplot(x_1d, y_1d, vx_masked, vy_masked, 
                        density=1.5, color=v_mag, cmap='plasma',
                        linewidth=1, arrowsize=1.2)
    fig.colorbar(strm.lines, ax=ax, label='Velocity Magnitude (m/s)')
    
    # Add electrodes and obstacles
    # Emitter
    circle = Circle((emitter_x, emitter_y), emitter_radius, color='blue', alpha=0.7)
    ax.add_patch(circle)
    
    # Screen
    rect = Rectangle((screen_start_x, screen_y-0.003), screen_end_x - screen_start_x, 0.006, 
                    color='red', alpha=0.7)
    ax.add_patch(rect)
    
    # Triangle
    triangle = Polygon([(triangle_x1, triangle_y1), (triangle_x2, triangle_y2), (triangle_x3, triangle_y3)], 
                      closed=True, color='gray', alpha=0.5)
    ax.add_patch(triangle)
    
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    time_value = time_index * total_time / (len(velocity_history) - 1) if time_index != -1 else total_time
    ax.set_title(f'Air Flow Streamlines at t = {time_value:.3f} s')
    ax.set_aspect('equal')
    
    return fig, ax

def plot_thrust():
    #fig, ax = plt.subplots(figsize=(8, 6))
    fig, ax = plot_bounds()
    
    # Extract time and thrust values
    times = [t[0] for t in thrust_over_time]
    thrust_values = [t[1] for t in thrust_over_time]
    
    # Plot thrust vs time
    ax.plot(times, thrust_values, 'r-o', linewidth=2)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Thrust (arb. units)')
    ax.set_title('Total Thrust vs Time')
    ax.grid(True)
    
    return fig, ax
