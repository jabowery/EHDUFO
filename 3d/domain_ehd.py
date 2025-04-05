from ngsolve import *
from netgen.geom2d import EdgeInfo as EI, PointInfo as PI, Solid2d
from netgen.geom2d import CSG2d
import numpy as np
import matplotlib.pyplot as plt
from geometric_object import GeometricObject

class EHDDomain(GeometricObject):
    """Domain class that manages EHD-specific physics and mesh."""
    
    def __init__(self, outer_r=20.0, outer_z=40.0, epsilon_0=8.854e-12, 
                 boundary_voltage=0.0, **kwargs):
        super().__init__()

        self.outer_r = outer_r
        self.outer_z = outer_z
        self.epsilon_0 = epsilon_0
        self.dirichlet_boundaries = {}

        # Physics properties
        self.ion_mobility = kwargs.get('ion_mobility', 2e-4)
        self.ion_diffusivity = kwargs.get('ion_diffusivity', 5e-5)
        self.rho_air = kwargs.get('rho_air', 1.225)
        self.mu_air = kwargs.get('mu_air', 1.8e-5)
        self.min_ion_density = kwargs.get('min_ion_density', 1e-14)

        # Domain attributes
        self.materials = {"air": {"epsilon_r": 1.0}}

        # Simulation objects
        self.mesh = None
        
        # Add grid functions to domain attributes
        self.phi_pot_gf = None  # Potential
        self.rho_charge_gf = None  # Charge density
        self.u_gf = None  # Velocity
        self.E_field = None  # Electric field (derived)
    
        
    def generate_composite_mesh(self, maxh=0.9):
        """Generate mesh including all contained objects."""
        geo = CSG2d()

        # Create the base domain geometry
        base_rect = Solid2d([
            (0, 0),
            (self.outer_r, 0),
            EI(bc="right"),
            (self.outer_r, self.outer_z),
            (0, self.outer_z),
            EI(maxh=0.4),
        ], mat="air")

        # Add domain boundary conditions
        self.dirichlet_boundaries.update({'right': {'volts': 0}})

        # Start with complete domain
        domain_geo = base_rect

        # Apply CSG operations for each contained object
        for obj_info in self.contained_objects:
            obj = obj_info['object']
            obj_geo = obj.create_geometry()

            # Subtract object from domain
            domain_geo = domain_geo - obj_geo

            # Add the object with its material
            geo.Add(obj_geo)

        # Add the final domain geometry
        geo.Add(domain_geo)

        # Generate the mesh
        self.mesh = Mesh(geo.GenerateMesh(maxh=maxh))

        # Now that we have a mesh, make sure all objects inform us about their boundaries
        for obj_info in self.contained_objects:
            obj = obj_info['object']
            obj.apply_boundary_conditions(self)

        # Now create finite element spaces with collected boundaries
        # Filter boundaries based on their purpose - only use 'volts' boundaries for potential
        potential_boundaries = [bn for bn, props in self.dirichlet_boundaries.items()
                               if 'volts' in props]
        dirichlet_str = "|".join(potential_boundaries)
        self.fes_pot = H1(self.mesh, order=2, dirichlet=dirichlet_str)

        # For velocity, add no-flow boundaries
        vel_boundaries = [bn for bn, props in self.dirichlet_boundaries.items()
                         if 'no_flow' in props or 'volts' in props]  # Typically all boundaries are no-flow
        vel_dirichlet_str = "|".join(vel_boundaries)

        self.fes_rho = H1(self.mesh, order=1)
        self.fes_vel = VectorH1(self.mesh, order=2, dirichlet=vel_dirichlet_str)

        # Initialize grid functions
        self.phi_pot_gf = GridFunction(self.fes_pot)  # Potential
        self.rho_charge_gf = GridFunction(self.fes_rho)  # Charge density
        self.u_gf = GridFunction(self.fes_vel)  # Velocity

        # Set initial values
        self.rho_charge_gf.Set(self.min_ion_density)  # Minimum charge density
        self.u_gf.Set(CoefficientFunction((0, 0)))  # Zero initial velocity

        # Apply boundary conditions after creating grid functions
        self.apply_all_boundary_conditions()

        # Now that FES and GFs are created, notify objects
        for obj_info in self.contained_objects:
            obj = obj_info['object']
            obj.on_mesh_generated(self)

        return self.mesh

    def apply_all_boundary_conditions(self):
        """Apply all boundary conditions from domain and contained objects."""
        # Apply potential boundary conditions
        boundaries = self.mesh.GetBoundaries()
        for i in range(len(boundaries)):
            bn = boundaries[i]
            if bn in self.dirichlet_boundaries and 'volts' in self.dirichlet_boundaries[bn]:
                voltage = self.dirichlet_boundaries[bn]['volts']
                for dof in self.fes_pot.GetDofNrs(ElementId(BND, i)):
                    self.phi_pot_gf.vec[dof] = voltage
        
        # Apply velocity boundary conditions (zero at boundaries)
        for i in range(len(boundaries)):
            bn = boundaries[i]
            if bn in self.dirichlet_boundaries and ('no_flow' in self.dirichlet_boundaries[bn] or 'volts' in self.dirichlet_boundaries[bn]):
                for dof in self.fes_vel.GetDofNrs(ElementId(BND, i)):
                    self.u_gf.vec[dof] = 0

    def compute_electric_field(self):
        """Compute electric field from potential."""
        self.E_field = -grad(self.phi_pot_gf)
        return self.E_field

    def get_epsilon(self):
        """Get the permittivity coefficient function for the domain."""
        if self.mesh is None:
            raise ValueError("Mesh must be generated before getting permittivity")

        epsilon_r = CoefficientFunction([self.materials[mat]["epsilon_r"]
                                       for mat in self.mesh.GetMaterials()])
        return epsilon_r * self.epsilon_0

    def on_mesh_generated(self, domain):
        """Called when a mesh containing this electrode is generated.
        Binds the electrode to all relevant domain fields.
        """
        self.mesh = domain.mesh
        self.domain = domain
        
        # Store references to all relevant finite element spaces
        self.fes_pot = domain.fes_pot   # For potential field
        self.fes_rho = domain.fes_rho   # For charge density
        self.fes_vel = domain.fes_vel   # For velocity field
        
        # Find all DOFs on this electrode's boundary for each field
        # Potential DOFs (for Dirichlet BC)
        self.pot_boundary_dofs = set()
        # Charge DOFs (for emission/collection)
        self.rho_boundary_dofs = set()
        
        for i in range(self.mesh.nface):
            bn = self.mesh.GetBoundaries()[i]
            if bn == self.name:
                # Get potential DOFs
                for dof in self.fes_pot.GetDofNrs(ElementId(BND, i)):
                    self.pot_boundary_dofs.add(dof)
                
                # Get charge density DOFs
                for dof in self.fes_rho.GetDofNrs(ElementId(BND, i)):
                    self.rho_boundary_dofs.add(dof)
        
        # Calculate electrode geometry properties based on mesh
        self.length = self._calculate_total_length()
        self.area = self._calculate_approximate_area()
        
        print(f"Electrode {self.name} connected to mesh with:")
        print(f"  - {len(self.pot_boundary_dofs)} potential DOFs")
        print(f"  - {len(self.rho_boundary_dofs)} charge density DOFs")
        print(f"  - Length: {self.length:.6f} m, Area: {self.area:.6f} m²")
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

    # In domain_ehd.py, modify the sample_potential_at_point method
    def sample_potential_at_point(self, mesh, phi_pot_gf, r, z):
        """
        Sample the potential field at a specified point (r, z) with improved error handling.

        Args:
            mesh: The computational mesh
            phi_pot_gf: The potential field GridFunction
            r, z: Coordinates of the point to sample

        Returns:
            The potential value at the specified point or None if sampling fails
        """
        try:
            # Create mesh point using mesh coordinates
            mesh_point = mesh(r, z)
            potential = phi_pot_gf(mesh_point)
            return potential
        except Exception as e:
            # First fallback: Try a point slightly inside the domain
            try:
                # Adjust point slightly toward center of domain
                adjusted_r = 0.99 * r if r > 0.1 else 1.01 * r
                adjusted_z = 0.99 * z if z > self.outer_z/2 else 1.01 * z
                mesh_point = mesh(adjusted_r, adjusted_z)
                potential = phi_pot_gf(mesh_point)
                print(f"Used adjusted point ({adjusted_r}, {adjusted_z}) for sampling")
                return potential
            except Exception as e2:
                # Second fallback: Try to find the nearest point in the mesh
                try:
                    # Find closest mesh vertex and sample there
                    min_dist = float('inf')
                    closest_point = None

                    # Check for a few vertices manually (not an exhaustive search but fast)
                    for el in mesh.Elements():
                        for v in el.vertices:
                            vertex = mesh[v]
                            vr = vertex.point[0]
                            vz = vertex.point[1]
                            dist = (vr-r)**2 + (vz-z)**2
                            if dist < min_dist:
                                min_dist = dist
                                closest_point = (vr, vz)

                    if closest_point:
                        mesh_point = mesh(closest_point[0], closest_point[1])
                        potential = phi_pot_gf(mesh_point)
                        print(f"Used nearest point ({closest_point[0]}, {closest_point[1]}) for sampling")
                        return potential
                except Exception as e3:
                    print(f"Warning: Could not sample potential at point ({r}, {z}): {e}")
                    return None
