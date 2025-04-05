from ngsolve import *
import numpy as np
import math
from geometric_object import GeometricObject

class Electrode(GeometricObject):
    """
    A class representing an electrode in an EHD simulation.
    Handles operations specific to electrodes like charge distribution,
    field calculation, and other electrode-specific properties.
    """

    def __init__(self, name, initial_voltage=0.0, capacitance=1e-11):
        self.name = name
        self._voltage = initial_voltage
        self.capacitance = capacitance
        self._charge = self._voltage * self.capacitance
        
        # References
        self.aircraft = None
        self.mesh = None
        self.boundary_dofs = None
        
    @property
    def voltage(self):
        """Get the electrode voltage"""
        return self._voltage
    
    @voltage.setter
    def voltage(self, value):
        """
        Set the electrode voltage and update charge accordingly
        
        Args:
            value: New voltage value (V)
        """
        self._voltage = value
    
    @property
    def charge(self):
        """Get the electrode charge"""
        return self._charge
    
    @charge.setter
    def charge(self, value):
        """
        Set the electrode charge and update voltage accordingly
        
        Args:
            value: New charge value (C)
        """
        self._charge = value
    
    def _calculate_total_length(self):
        """Calculate the total length of the electrode boundary."""
        total_length = 0
        
        # Iterate through boundary elements
        for i, bn in enumerate(self.mesh.GetBoundaries()):
            if bn == self.name:
                # Get the element
                el = self.mesh[ElementId(BND, i)]
                
                # Get coordinates of vertices
                vertices = []
                for v in el.vertices:
                    vertex = self.mesh[v]
                    vertices.append((vertex.point[0], vertex.point[1]))
                
                # Calculate edge length
                if len(vertices) >= 2:
                    v1 = vertices[0]
                    v2 = vertices[1]
                    edge_length = math.sqrt((v1[0]-v2[0])**2 + (v1[1]-v2[1])**2)
                    total_length += edge_length
                    
        return total_length
    
    def _calculate_approximate_area(self):
        """
        Estimate the electrode area based on its boundary in axisymmetric coordinates.
        In 2D axisymmetric, the area is approximately the circumference (2πr) times the length.
        
        Returns:
            Approximate area (m²)
        """
        # Get points along the electrode
        points = self.get_mesh_points()
        
        # Calculate average radius
        avg_r = sum(point[0][0] for point in points) / max(len(points), 1)
        
        # In axisymmetric coordinates, area ≈ 2πr * length
        area = 2 * np.pi * avg_r * self.length
        
        return area
    
    def get_mesh_points(self, offset=0.0):
        """
        Get a list of mesh points along the electrode with optional offset.
        
        Args:
            offset: Distance to offset points from the boundary (normal direction)
        
        Returns:
            List of (r, z) coordinates and corresponding boundary elements
        """
        mesh_points = []
        
        # Iterate through boundary elements
        for i, bn in enumerate(self.mesh.GetBoundaries()):
            if bn == self.name:
                # Get the element
                el = self.mesh[ElementId(BND, i)]
                
                # Get coordinates of vertices
                vertices = []
                for v in el.vertices:
                    vertex = self.mesh[v]
                    vertices.append((vertex.point[0], vertex.point[1]))
                
                # Calculate midpoint
                if len(vertices) >= 2:
                    r = sum(v[0] for v in vertices) / len(vertices)
                    z = sum(v[1] for v in vertices) / len(vertices)
                    
                    # Calculate normal vector
                    if len(vertices) == 2:
                        v1 = vertices[0]
                        v2 = vertices[1]
                        
                        # Edge vector
                        edge_r = v2[0] - v1[0]
                        edge_z = v2[1] - v1[1]
                        
                        # Normal vector (rotated 90 degrees)
                        length = math.sqrt(edge_r**2 + edge_z**2)
                        if length > 0:
                            normal_r = -edge_z / length
                            normal_z = edge_r / length
                            
                            # Apply offset in normal direction
                            sample_r = r + offset * normal_r
                            sample_z = z + offset * normal_z
                        else:
                            sample_r = r
                            sample_z = z + offset  # Default to z-direction offset
                    else:
                        # Default offset if we can't calculate normal
                        sample_r = r
                        sample_z = z + offset
                    
                    mesh_points.append(((sample_r, sample_z), el))
                    
        return mesh_points
    
    def calculate_average_field(self, E_field, offset=0.01):
        """
        Calculate the average electric field magnitude along the electrode.
        
        Args:
            E_field: Electric field CoefficientFunction
            offset: Distance from electrode to sample field (m)
            
        Returns:
            Average electric field magnitude (V/m)
        """
        field_sum = 0.0
        total_length = 0.0
        
        # Get points along the electrode with offset
        sample_points = self.get_mesh_points(offset)
        
        for (sample_r, sample_z), el in sample_points:
            try:
                # Evaluate E-field at this point using the mesh() function to create a MeshPoint
                mesh_point = self.mesh(sample_r, sample_z)
                local_E_field = E_field(mesh_point)
                local_E_mag = math.sqrt(local_E_field[0]**2 + local_E_field[1]**2)
                
                # Find the edge length for this element
                vertices = []
                for v in el.vertices:
                    vertex = self.mesh[v]
                    vertices.append((vertex.point[0], vertex.point[1]))
                
                if len(vertices) >= 2:
                    v1 = vertices[0]
                    v2 = vertices[1]
                    edge_length = math.sqrt((v1[0]-v2[0])**2 + (v1[1]-v2[1])**2)
                    field_sum += local_E_mag * edge_length
                    total_length += edge_length
            except Exception as e:
                # Try a different approach - move point slightly inward if it's outside mesh
                fallback_r = sample_r * 0.99 if sample_r > 0 else sample_r * 1.01
                fallback_z = sample_z * 0.99
                try:
                    mesh_point = self.mesh(fallback_r, fallback_z)
                    local_E_field = E_field(mesh_point)
                    local_E_mag = math.sqrt(local_E_field[0]**2 + local_E_field[1]**2)
                    
                    # Find the edge length for this element
                    vertices = []
                    for v in el.vertices:
                        vertex = self.mesh[v]
                        vertices.append((vertex.point[0], vertex.point[1]))
                    
                    if len(vertices) >= 2:
                        v1 = vertices[0]
                        v2 = vertices[1]
                        edge_length = math.sqrt((v1[0]-v2[0])**2 + (v1[1]-v2[1])**2)
                        field_sum += local_E_mag * edge_length
                        total_length += edge_length
                except Exception as e2:
                    pass
        
        if total_length > 0:
            return field_sum / total_length
        else:
            return 0.0
    
    def emit(self, rho_charge, dt=1e-6, emission_coefficient=1e-9, field_enhancement_factor=1.0):
        """
        A more physically realistic emission model that emits charge based on
        electrode voltage, field strength, and field enhancement factors.
        
        Args:
            rho_charge: GridFunction to update with emitted charge
            dt: Time step size in seconds
            emission_coefficient: Base emission rate coefficient (C/V/s)
            field_enhancement_factor: Factor enhancing emission due to electrode geometry
            
        Returns:
            Amount of charge emitted (C)
        """
        # Calculate emission based on voltage magnitude, field, and enhancement
        voltage_magnitude = abs(self._voltage)
        
        # Apply nonlinear dependence on voltage (field emission-like behavior)
        # Higher voltages lead to exponentially higher emission rates
        voltage_factor = np.exp(voltage_magnitude / 10000) - 1
        emission_rate = emission_coefficient * voltage_factor * field_enhancement_factor
        total_emission = emission_rate * dt
        
        # Cap emission to prevent excessive charge removal
        # Limit to a percentage of current charge to maintain stability
        max_emission = 0.1 * abs(self._charge)
        if total_emission > max_emission:
            total_emission = max_emission
        
        # For negative electrodes (emitters), we emit negative charge
        if self._voltage < 0:
            total_emission = -total_emission
        
        # Create a boundary CoefficientFunction for the emission
        boundary_cf = CoefficientFunction(total_emission / self.length)
        
        # Create a temporary GridFunction to hold our emission
        emission_gf = GridFunction(self.fes_rho)
        
        # Set the emission on the boundary
        emission_gf.Set(boundary_cf, definedon=self.mesh.Boundaries(self.name))
        
        # Create a mass matrix to project this boundary emission into the domain
        mass_form = BilinearForm(self.fes_rho)
        u, v = self.fes_rho.TnT()
        mass_form += u*v*dx
        mass_form.Assemble()
        
        # Create a load vector from our boundary emission
        load_form = LinearForm(self.fes_rho)
        load_form += emission_gf*v*dx
        load_form.Assemble()
        
        # Solve to get the projection
        projected_emission = GridFunction(self.fes_rho)
        projected_emission.vec.data = mass_form.mat.Inverse() * load_form.vec
        
        # Calculate the actual total emission after projection
        actual_emission = Integrate(projected_emission, self.mesh)
        
        # Add the emission to the space charge (directly accumulate)
        # We need to manually add element by element since direct vector addition doesn't work as expected
        for i in range(len(rho_charge.vec)):
            rho_charge.vec[i] += projected_emission.vec[i]
        
        # Update the electrode charge
        self.charge -= actual_emission
        
        return actual_emission
    
    def collect_charge(self, rho_charge, ion_velocity, dt, min_ion_density=1e-14):
        """
        Collect charge from the space charge region at the collector electrode surface
        using a direct flux calculation method.
        
        Args:
            rho_charge: GridFunction containing space charge
            ion_velocity: GridFunction or CoefficientFunction of ion velocity
            dt: Time step in seconds
            min_ion_density: Minimum ion density to maintain
            
        Returns:
            Amount of charge collected (C)
        """
        # For flux calculation at boundary, we'll use direct integration
        # Define the normal component with bn = v·n
        # We need to provide the dimension for specialcf.normal
        bn = InnerProduct(ion_velocity, specialcf.normal(2))  # 2 is the dimension (2D)
        
        # Define inward flux (only collect when flow is into the electrode)
        # For axisymmetric coordinates, include 2πr factor (x is radius)
        inward_flux = IfPos(bn, 0, -bn * rho_charge * 2 * pi * x)
        
        # Directly integrate the flux over the boundary
        collected_flux = Integrate(
            inward_flux, 
            self.mesh, 
            BND, 
            definedon=self.mesh.Boundaries(self.name)
        )
        
        # Scale by time step to get amount collected in this step
        collected_charge = collected_flux * dt
        
        # Now create a boundary layer for charge depletion
        # We'll use a direct approach to set values
        
        # Create a function to hold our depletion field
        charge_depletion = GridFunction(self.fes_rho)
        
        # Create a diffusion problem to create a boundary layer
        # This creates a function that is highest at boundary and decays inward
        a = BilinearForm(self.fes_rho)
        u, v = self.fes_rho.TnT()
        
        # Simple Laplace problem with Dirichlet boundary
        a += grad(u) * grad(v) * dx
        a.Assemble()
        
        # Define right-hand side with boundary values
        f = LinearForm(self.fes_rho)
        
        # Add boundary term - this will be projected into domain
        # First calculate appropriate value based on collected charge
        if collected_charge > 0:
            # Safety checks to avoid division by zero
            boundary_length = Integrate(1.0, self.mesh, BND, definedon=self.mesh.Boundaries(self.name))
            
            if boundary_length > 0:
                # Set a scaling factor that will deplete appropriate amount of charge
                # We want to deplete approximately the collected charge amount
                # from a thin layer near the boundary
                
                # Estimate of boundary layer volume
                layer_thickness = 0.02  # Approximate thickness of boundary layer
                approx_layer_volume = boundary_length * layer_thickness * 2 * pi * (sum(p[0] for p in self.get_mesh_points()) / len(self.get_mesh_points()))
                
                # Scale to ensure we remove approximately the collected charge
                boundary_value = collected_charge / max(approx_layer_volume, 1e-10)
                
                # Add Dirichlet value at boundary
                f += boundary_value * v * ds(definedon=self.mesh.Boundaries(self.name))
                f.Assemble()
                
                # Solve the diffusion problem
                charge_depletion.vec.data = a.mat.Inverse(self.fes_rho.FreeDofs()) * f.vec
        
        # Calculate the actual amount we're depleting for conservation
        actual_depletion = Integrate(charge_depletion, self.mesh)
        
        # Scale to match collected charge if necessary
        if actual_depletion > 0 and collected_charge > 0:
            scaling_factor = collected_charge / max(actual_depletion, 1e-10)
            charge_depletion.vec.data *= scaling_factor
        
        # Apply the depletion to the charge density
        for i in range(len(rho_charge.vec)):
            rho_charge.vec[i] -= charge_depletion.vec[i]
            
            # Ensure minimum density is maintained
            if rho_charge.vec[i] < min_ion_density:
                rho_charge.vec[i] = min_ion_density
        
        # Update the electrode charge
        self.charge += collected_charge
        
        return collected_charge

    def set_dirichlet_bc(self, phi_pot_gf):
        """Apply voltage boundary condition to potential field."""
        if not hasattr(self, 'pot_boundary_dofs') or not self.pot_boundary_dofs:
            print(f"Warning: Electrode {self.name} doesn't have boundary DOFs set up")
            return
            
        # Set the voltage on all DOFs associated with this electrode's boundary
        for dof in self.pot_boundary_dofs:
            phi_pot_gf.vec[dof] = self._voltage
        
        # Also try setting directly using the boundary name
        try:
            phi_pot_gf.Set(self._voltage, definedon=self.mesh.Boundaries(self.name))
            print(f"Applied {self._voltage:.2f}V to boundary '{self.name}' using Set method")
        except Exception as e:
            print(f"Could not apply boundary condition using Set method: {e}")

    def apply_boundary_conditions(self, domain):
        """Apply boundary conditions for this aircraft."""
        # Let electrodes apply their boundary conditions
        if self.emitter_electrode:
            domain.dirichlet_boundaries.update({
                'emitter': {'volts': self.emitter_electrode.voltage}
            })
        
        if self.collector_electrode:
            domain.dirichlet_boundaries.update({
                'collector': {'volts': self.collector_electrode.voltage}
        })

    def on_mesh_generated(self, domain):
        """Called when a mesh containing this electrode is generated."""
        print(f"Electrode {self.name}: on_mesh_generated called")
        self.mesh = domain.mesh
        self.domain = domain
        
        # Store references to all relevant finite element spaces
        self.fes_pot = domain.fes_pot
        self.fes_rho = domain.fes_rho
        self.fes_vel = domain.fes_vel
        
        # Find all DOFs on this electrode's boundary
        self.pot_boundary_dofs = set()
        self.rho_boundary_dofs = set()
        
        boundaries = self.mesh.GetBoundaries()
        print(f"Electrode {self.name}: Searching for boundary DOFs among {len(boundaries)} boundaries")
        for i in range(len(boundaries)):
            bn = boundaries[i]
            if bn == self.name:
                print(f"Electrode {self.name}: Found matching boundary at index {i}")
                # Get potential DOFs
                pot_dofs = self.fes_pot.GetDofNrs(ElementId(BND, i))
                print(f"Electrode {self.name}: Found {len(pot_dofs)} potential DOFs")
                for dof in pot_dofs:
                    self.pot_boundary_dofs.add(dof)
                
                # Get charge density DOFs
                rho_dofs = self.fes_rho.GetDofNrs(ElementId(BND, i))
                print(f"Electrode {self.name}: Found {len(rho_dofs)} charge density DOFs")
                for dof in rho_dofs:
                    self.rho_boundary_dofs.add(dof)
        
        # Calculate electrode geometry properties
        self.length = self._calculate_total_length()
        self.area = self._calculate_approximate_area()
        
        print(f"Electrode {self.name} connected to mesh with:")
        print(f"  - {len(self.pot_boundary_dofs)} potential DOFs")
        print(f"  - {len(self.rho_boundary_dofs)} charge density DOFs")
        print(f"  - Length: {self.length:.6f} m, Area: {self.area:.6f} m²")
