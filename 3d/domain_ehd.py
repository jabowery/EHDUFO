from ngsolve import *
from netgen.geom2d import EdgeInfo as EI, PointInfo as PI, Solid2d
from netgen.geom2d import CSG2d
import numpy as np
import matplotlib.pyplot as plt

class EHDDomain:
    """
    Class representing the computational domain for EHD simulations,
    including air properties and mesh generation.
    """
    def __init__(self, 
                 outer_r=20.0,
                 outer_z=40.0,
                 epsilon_0=8.854e-12,
                 ion_mobility=2e-4,
                 ion_diffusivity=5e-5,
                 rho_air=1.225,
                 mu_air=1.8e-5,
                 min_ion_density=1e-14,
                 ion_neutral_collision_freq=1e9):
        """
        Initialize the EHD domain with physical properties.
        
        Args:
            outer_r: Outer radius of computational domain (m)
            outer_z: Outer height of computational domain (m)
            epsilon_0: Vacuum permittivity (F/m)
            ion_mobility: Ion mobility coefficient (m^2/(V·s))
            ion_diffusivity: Ion diffusion coefficient (m^2/s)
            rho_air: Air density (kg/m^3)
            mu_air: Air dynamic viscosity (Pa·s)
            min_ion_density: Minimum ion density to maintain (C/m^3)
            ion_neutral_collision_freq: Collision frequency between ions and neutral molecules (1/s)
        """
        # Domain dimensions
        self.outer_r = outer_r
        self.outer_z = outer_z
        
        # Physical constants
        self.epsilon_0 = epsilon_0
        self.ion_mobility = ion_mobility
        self.ion_diffusivity = ion_diffusivity
        self.rho_air = rho_air
        self.mu_air = mu_air
        self.min_ion_density = min_ion_density
        self.ion_neutral_collision_freq = ion_neutral_collision_freq
        
        # Placeholders
        self.mesh = None
        self.materials = {
            "rect": {"epsilon_r": 1.0},      # Air region
            "insulator": {"epsilon_r": 4.0}  # Insulator region
        }
        
    def generate_mesh(self, aircraft, emitter_gridsize=0.1):
        """
        Generate a mesh for the combined air domain and aircraft.
        
        Args:
            aircraft: EHDAircraft object
            emitter_gridsize: Grid size near emitter for refinement
            
        Returns:
            NGSolve mesh
        """
        geo = CSG2d()
        
        # Create the outer rectangle (air domain)
        rect = Solid2d([
            (0, 0),
            (self.outer_r, 0),
            EI(bc="right"),
            (self.outer_r, self.outer_z),
            (0, self.outer_z),
            EI(maxh=0.4),
        ], mat="rect")
        
        # Get aircraft geometry (insulator)
        rhombus = aircraft.create_geometry()
        
        # Combine domains: air = outer rectangle - insulator
        geo.Add(rect - rhombus)
        geo.Add(rhombus)
        
        # Generate mesh
        self.mesh = Mesh(geo.GenerateMesh(maxh=0.9))
        return self.mesh
    
    def get_epsilon(self):
        """
        Get the permittivity coefficient function for the domain.
        
        Returns:
            CoefficientFunction for permittivity
        """
        if self.mesh is None:
            raise ValueError("Mesh must be generated before getting permittivity")
            
        epsilon_r = CoefficientFunction([self.materials[mat]["epsilon_r"] 
                                         for mat in self.mesh.GetMaterials()])
        return epsilon_r * self.epsilon_0
    
    def create_potential_profile(self, mesh, phi_pot_gf, emitter, collector, t, aircraft):
        """
        Creates 1D profiles of the potential field along different paths
        
        Args:
            mesh: The computational mesh
            phi_pot_gf: The potential GridFunction
            emitter, collector: Electrode objects
            t: Current simulation time
            aircraft: EHDAircraft object containing geometry information
        """
        # Create sampling points along the z-axis
        z_vals = np.linspace(0, self.outer_z, 200)
        r_val = 0.01  # Use the central axis or close to it
        
        # Sample potential at each point
        potentials = []
        valid_z = []
        
        for z in z_vals:
            try:
                mesh_point = mesh(r_val, z)
                potential = phi_pot_gf(mesh_point)
                potentials.append(potential)
                valid_z.append(z)
            except:
                pass  # Point might be outside the mesh or in an element
        
        # Create a plot
        plt.figure(figsize=(10, 6))
        plt.plot(valid_z, potentials, 'b-', linewidth=2)
        plt.axhline(y=emitter.voltage, color='r', linestyle='--', 
                   label=f'Emitter voltage: {emitter.voltage:.0f}V')
        plt.axhline(y=collector.voltage, color='g', linestyle='--', 
                   label=f'Collector voltage: {collector.voltage:.0f}V')
        
        # Mark emitter and collector positions
        plt.axvline(x=aircraft.emitter_z, color='r', linestyle=':', label='Emitter position')
        plt.axvline(x=aircraft.collector_z, color='g', linestyle=':', label='Collector position')
        
        plt.title(f'Potential Profile Along Z-Axis (r={r_val}) at t={t:.4f}s')
        plt.xlabel('z position (m)')
        plt.ylabel('Potential (V)')
        plt.legend()
        plt.grid(True)
        
        # Save the figure
        plt.savefig(f'ehd_output/potential_profile_t{t:.4f}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also create a sample along the insulator surface
        insulator_r_vals = np.linspace(aircraft.collector_r, aircraft.emitter_r, 100)
        insulator_z_vals = np.linspace(aircraft.collector_z, aircraft.emitter_z, 100)
        
        insulator_potentials = []
        valid_indices = []
        
        for i in range(len(insulator_r_vals)):
            try:
                r = insulator_r_vals[i]
                z = insulator_z_vals[i]
                mesh_point = mesh(r, z)
                potential = phi_pot_gf(mesh_point)
                insulator_potentials.append(potential)
                valid_indices.append(i)
            except:
                pass
        
        # Create a plot for insulator surface potential
        if valid_indices:
            valid_r = [insulator_r_vals[i] for i in valid_indices]
            valid_z = [insulator_z_vals[i] for i in valid_indices]
            
            plt.figure(figsize=(10, 6))
            plt.subplot(211)
            plt.plot(valid_r, insulator_potentials, 'b-', linewidth=2)
            plt.title(f'Potential Along Insulator (r coordinate) at t={t:.4f}s')
            plt.ylabel('Potential (V)')
            plt.axhline(y=emitter.voltage, color='r', linestyle='--', 
                       label=f'Emitter voltage: {emitter.voltage:.0f}V')
            plt.axhline(y=collector.voltage, color='g', linestyle='--', 
                       label=f'Collector voltage: {collector.voltage:.0f}V')
            plt.grid(True)
            plt.legend()
            
            plt.subplot(212)
            plt.plot(valid_z, insulator_potentials, 'b-', linewidth=2)
            plt.title(f'Potential Along Insulator (z coordinate) at t={t:.4f}s')
            plt.xlabel('z position (m)')
            plt.ylabel('Potential (V)')
            plt.axhline(y=emitter.voltage, color='r', linestyle='--', 
                       label=f'Emitter voltage: {emitter.voltage:.0f}V')
            plt.axhline(y=collector.voltage, color='g', linestyle='--', 
                       label=f'Collector voltage: {collector.voltage:.0f}V')
            plt.grid(True)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(f'ehd_output/insulator_potential_t{t:.4f}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def sample_potential_at_point(self, mesh, phi_pot_gf, r, z):
        """
        Sample the potential field at a specified point (r, z).
        
        Args:
            mesh: The computational mesh
            phi_pot_gf: The potential field GridFunction
            r, z: Coordinates of the point to sample
            
        Returns:
            The potential value at the specified point
        """
        try:
            mesh_point = mesh(r, z)
            potential = phi_pot_gf(mesh_point)
            return potential
        except Exception as e:
            print(f"Warning: Could not sample potential at point ({r}, {z}): {e}")
            return None
