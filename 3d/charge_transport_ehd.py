import logging
import logging
from ngsolve import *
import math
import numpy as np

def solve_charge_transport(rho_current, mesh, fes_rho, dt, ion_mobility, E_field, u_gf, u_vehicle, 
                           diffusivity=5e-5, min_ion_density=1e-9, stabilization_factor=1.0):
    """
    Solves the charge transport equation with improved stability and accuracy.
    Uses a combination of methods for robustness.
    
    Args:
        rho_current: Current charge density GridFunction
        mesh: Computational mesh
        fes_rho: Finite element space for charge density
        dt: Time step
        ion_mobility: Ion mobility coefficient
        E_field: Electric field CoefficientFunction
        u_gf: Fluid velocity GridFunction
        u_vehicle: Vehicle velocity scalar
        diffusivity: Ion diffusion coefficient
        min_ion_density: Minimum ion density to maintain
        stabilization_factor: Factor to adjust numerical stabilization (default=1.0)
        
    Returns:
        Updated charge density GridFunction
    """
    # Calculate ion velocity
    ion_velocity = ion_mobility * E_field + u_gf - CoefficientFunction((0, u_vehicle))
    
    # Calculate velocity magnitude for stabilization parameter
    velocity_magnitude = sqrt(ion_velocity[0]**2 + ion_velocity[1]**2)
    
    # Trial and test functions
    rho, v = fes_rho.TnT()
    
    # Create a new GridFunction that will hold our solution
    rho_new = GridFunction(fes_rho)
    rho_new.vec.data = rho_current.vec  # Start with current values
    
    # Bilinear form for the implicit part
    a = BilinearForm(fes_rho)
    
    # Mass term
    a += rho * v * dx
    
    # Add diffusion term with physical diffusivity
    a += dt * diffusivity * grad(rho) * grad(v) * dx
    
    # Streamline Upwind Petrov-Galerkin (SUPG) stabilization
    # Calculate mesh size parameter h for each element
    h = specialcf.mesh_size
    
    # SUPG parameter (tau)
    tau = stabilization_factor * h / (2 * velocity_magnitude + 1e-10)
    
    # Add SUPG stabilization term - helps prevent oscillations in advection-dominated flow
    a += tau * (ion_velocity * grad(rho)) * (ion_velocity * grad(v)) * dx
    
    # Create linear form for explicit treatment of convection
    f = LinearForm(fes_rho)
    f += rho_current * v * dx  # Previous solution
    
    # Explicit treatment of convection term (using DG upwinding at element interfaces)
    # Use a properly formatted upwind coefficient with specialcf.normal
    # For 2D problems, the dimension must be provided: specialcf.normal(2)
    
    # Define upwind term using symbolic boundary form for internal skeleton faces
    # The correct way to handle this in NGSolve:
    upwind_cf = IfPos(
        InnerProduct(ion_velocity, specialcf.normal(2)), 
        0,  # If flow is outward (v·n > 0), use 0
        1   # If flow is inward (v·n < 0), use 1
    )
    
    # Properly defined convection form using NGSolve's boundary integration
    convection_form = dt * (
        -ion_velocity * grad(v) * rho_current * dx
        + (rho_current - rho_current.Other()) * 
          InnerProduct(ion_velocity, specialcf.normal(2)) *
          upwind_cf * v * dx(skeleton=True)
    )
    
    f += convection_form
    
    # Assemble the system
    a.Assemble()
    f.Assemble()
    
    # Solve the system
    rho_new.vec.data = a.mat.Inverse(fes_rho.FreeDofs()) * f.vec
    
    # Enforce minimum density and prevent negative values
    for i in range(len(rho_new.vec)):
        if rho_new.vec[i] < min_ion_density:
            rho_new.vec[i] = min_ion_density
        # Also check for NaN values and replace them
        if math.isnan(rho_new.vec[i]):
            rho_new.vec[i] = min_ion_density
    
    return rho_new
if False:
    def solve_poisson_equation(mesh, fes_pot, rho_charge, epsilon, emitter_voltage=None, collector_voltage=None):
        """
        Solve Poisson's equation for the electric potential with improved boundary condition handling.
        
        Args:
            mesh: The computational mesh
            fes_pot: Finite element space for potential
            rho_charge: Charge density GridFunction
            epsilon: Permittivity CoefficientFunction
            emitter_voltage: Optional voltage value for emitter (for boundary verification)
            collector_voltage: Optional voltage value for collector (for boundary verification)
            
        Returns:
            Updated potential GridFunction
        """
        # Trial and test functions
        phi_pot, v_phi_pot = fes_pot.TnT()
        
        # Create potential GridFunction
        phi_pot_gf = GridFunction(fes_pot)
        
        # Make sure we preserve any existing boundary conditions
        # This ensures Dirichlet values are maintained when we start the solve
        if hasattr(fes_pot, 'dirichlet_dofs'):
            for dof in fes_pot.dirichlet_dofs:
                phi_pot_gf.vec[dof] = fes_pot.dirichlet_values[dof]
        
        # Bilinear form (left-hand side)
        a_pot = BilinearForm(fes_pot)
        a_pot += epsilon * grad(phi_pot) * grad(v_phi_pot) * dx
        
        # Linear form (right-hand side) - source term from charge density
        L_pot = LinearForm(fes_pot)
        L_pot += rho_charge * v_phi_pot * dx
        
        # Assemble the system
        a_pot.Assemble()
        L_pot.Assemble()
        
        # Create a residual vector for the update
        r = L_pot.vec.CreateVector()
        r.data = L_pot.vec - a_pot.mat * phi_pot_gf.vec
        
        # Update the potential while respecting Dirichlet boundary conditions
        phi_pot_gf.vec.data += a_pot.mat.Inverse(fes_pot.FreeDofs()) * r
        
        # Verify boundary conditions are preserved
        bc_error = False
        boundary_values = {
            "emitter": emitter_voltage,
            "collector": collector_voltage,
            "right": 0.0
        }
        
        for i, bn in enumerate(mesh.GetBoundaries()):
            if bn in boundary_values and boundary_values[bn] is not None:
                # Sample a point on this boundary to check value
                el = mesh[ElementId(BND, i)]
                vertices = []
                for v in el.vertices:
                    vertex = mesh[v]
                    vertices.append((vertex.point[0], vertex.point[1]))
                
                if len(vertices) >= 1:
                    r = vertices[0][0]
                    z = vertices[0][1]
                    # Sample potential at this boundary point
                    try:
                        mesh_point = mesh(r, z)
                        value = phi_pot_gf(mesh_point)
                        expected_value = boundary_values[bn]
                        
                        if abs(value - expected_value) > 1e-6:
                            logging.debug(f"Warning: Boundary {bn} has value {value:.2f}V instead of expected {expected_value:.2f}V")
                            bc_error = True
                            
                            # Check the actual DOF values for this boundary element
                            dofs = fes_pot.GetDofNrs(ElementId(BND, i))
                            if len(dofs) > 0:
                                dof_values = [phi_pot_gf.vec[dof] for dof in dofs]
                                logging.debug(f"  DOF values at boundary: {dof_values}")
                    except Exception as e:
                        logging.debug(f"Could not sample boundary {bn}: {e}")
        
        if bc_error:
            logging.debug("Attempting to restore boundary conditions...")
            # Force boundary condition compliance
            for bn in boundary_values:
                if boundary_values[bn] is not None and bn in mesh.GetBoundaries():
                    value = boundary_values[bn]
                    phi_pot_gf.Set(value, definedon=mesh.Boundaries(bn))
                    logging.debug(f"Re-applied {value:.2f}V to boundary {bn}")
            
            # Additional verification after correction
            logging.debug("Checking boundary conditions after restoration:")
            for bn in boundary_values:
                if boundary_values[bn] is not None and bn in mesh.GetBoundaries():
                    for i in range(mesh.nface):
                        if mesh.GetBoundaries()[i] == bn:
                            el = mesh[ElementId(BND, i)]
                            if len(el.vertices) >= 1:
                                vertex = mesh[el.vertices[0]]
                                r, z = vertex.point[0], vertex.point[1]
                                try:
                                    mesh_point = mesh(r, z)
                                    value = phi_pot_gf(mesh_point)
                                    logging.debug(f"  Boundary {bn}: {value:.2f}V (expected {boundary_values[bn]:.2f}V)")
                                    break
                                except Exception as e:
                                    logging.debug(f"  Could not verify boundary {bn}: {e}")
        
        return phi_pot_gf
else:
    def solve_poisson_equation(mesh, fes_pot, rho_charge, epsilon, emitter_voltage=None, collector_voltage=None):
        """
        Solve Poisson's equation for the electric potential with improved boundary condition handling.
        """
        # Trial and test functions
        phi_pot, v_phi_pot = fes_pot.TnT()
        
        # Create potential GridFunction
        phi_pot_gf = GridFunction(fes_pot)
        
        # Bilinear form (left-hand side)
        a_pot = BilinearForm(fes_pot)
        a_pot += epsilon * grad(phi_pot) * grad(v_phi_pot) * dx
        
        # Linear form (right-hand side) - source term from charge density
        L_pot = LinearForm(fes_pot)
        L_pot += rho_charge * v_phi_pot * dx
        
        # Assemble the system
        a_pot.Assemble()
        L_pot.Assemble()
        
        # Create a residual vector for the update
        r = L_pot.vec.CreateVector()
        r.data = L_pot.vec - a_pot.mat * phi_pot_gf.vec
        
        # Solve the system, respecting Dirichlet boundary conditions
        phi_pot_gf.vec.data += a_pot.mat.Inverse(fes_pot.FreeDofs()) * r
        
        # Verify boundary conditions are being properly applied
        logging.debug("Verifying boundary conditions:")
        boundaries = mesh.GetBoundaries()
        for i in range(len(boundaries)):
            bn = boundaries[i]
            if bn == "emitter" or bn == "collector":
                try:
                    # Find an element on this boundary
                    for el_id in range(mesh.ne):
                        el = mesh[ElementId(BND, el_id)]
                        if mesh.GetBoundaries()[el_id] == bn:
                            # Get the coordinates of the first vertex
                            vertex = mesh[el.vertices[0]]
                            r, z = vertex.point[0], vertex.point[1]
                            
                            # Sample the potential at this point
                            mesh_point = mesh(r, z)
                            value = phi_pot_gf(mesh_point)
                            logging.debug(f"  Boundary {bn}: sampled voltage at ({r:.2f}, {z:.2f}) = {value:.2f}V")
                            break
                except Exception as e:
                    logging.debug(f"  Could not sample boundary {bn}: {e}")
        
        return phi_pot_gf

def solve_navier_stokes(mesh, fes_vel, u_gf, rho_charge, E_field, dt, 
                        rho_air=1.225, mu_air=1.8e-5, ion_neutral_collision_freq=1e9):
    """
    Solve the Navier-Stokes equations for fluid velocity with ion drag force.
    
    Args:
        mesh: The computational mesh
        fes_vel: Finite element space for velocity
        u_gf: Current velocity GridFunction
        rho_charge: Charge density GridFunction
        E_field: Electric field CoefficientFunction
        dt: Time step
        rho_air: Air density (kg/m^3)
        mu_air: Air dynamic viscosity (Pa·s)
        ion_neutral_collision_freq: Collision frequency between ions and neutral molecules (1/s)
        
    Returns:
        Updated velocity GridFunction
    """
    # Trial and test functions
    u, v = fes_vel.TnT()
    
    # Create a bilinear form for the implicit part
    a_vel = BilinearForm(fes_vel)
    
    # Mass term
    a_vel += rho_air * InnerProduct(u, v) * dx
    
    # Viscosity term
    a_vel += dt * mu_air * InnerProduct(grad(u), grad(v)) * dx
    
    # Add convective term (linearized)
    a_vel += dt * rho_air * InnerProduct(grad(u) * u_gf, v) * dx
    
    # Create a linear form for the explicit part
    L_vel = LinearForm(fes_vel)
    
    # Previous velocity
    L_vel += rho_air * InnerProduct(u_gf, v) * dx
    
    # Calculate electric field magnitude
    E_norm = sqrt(E_field[0]**2 + E_field[1]**2)
    
    # Ion-drag force with mobility factor
    # Mobility decreases at high field strengths due to ion clustering/saturation effects
    mobility_factor = 1.0 / (1.0 + E_norm/1e6)
    
    # Collision factor - fraction of momentum transferred from ions to neutral air molecules
    collision_factor = (1.0 - exp(-ion_neutral_collision_freq * dt))
    
    # Ion momentum transfer term
    ion_momentum = rho_charge * E_field * mobility_factor * collision_factor
    
    # Add ion drag force to the linear form
    L_vel += dt * InnerProduct(ion_momentum, v) * dx
    
    # Assemble the system
    a_vel.Assemble()
    L_vel.Assemble()
    
    # Create new velocity field GridFunction
    u_new = GridFunction(fes_vel)
    
    # Solve the system
    u_new.vec.data = a_vel.mat.Inverse(fes_vel.FreeDofs()) * L_vel.vec
    
    return u_new

# These functions have been moved to their appropriate classes
# The thrust calculation is now in the EHDAircraft class
# The power calculation is now in the EHDAircraft class as calculate_system_power
