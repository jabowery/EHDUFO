import logging
from ngsolve import *
from netgen.geom2d import EdgeInfo as EI, PointInfo as PI, Solid2d
from netgen.geom2d import CSG2d
import numpy as np
from geometric_object import GeometricObject

class EHDAircraft(GeometricObject):
    def __init__(self, emitter_radius=0.1, collector_radius=5.0, height=3.0, 
                 center_hole_radius=0.0, base_z=18.5, velocity=0.0, mass = 1.0, **kwargs):
        super().__init__()
        # Geometry parameters
        self.emitter_radius = emitter_radius
        self.collector_radius = collector_radius
        self.center_hole_radius = center_hole_radius
        self.height = height
        self.base_z = base_z
        self.velocity = velocity
        self.mass = mass
        
        # Derived geometry parameters
        self.collector_z = self.base_z
        self.emitter_z = self.base_z + self.height
        self.collector_r = self.collector_radius + self.center_hole_radius
        self.emitter_r = self.emitter_radius + self.center_hole_radius
        
        # Material properties
        self.materials = {"insulator": {"epsilon_r": 4.0}}
        
        # Electrodes
        self.emitter_electrode = None
        self.collector_electrode = None
    
    def create_geometry(self):
        """Create the geometric representation of the aircraft."""
        rhombus = Solid2d([
            (self.center_hole_radius, self.collector_z),
            PI(maxh=0.1),
            EI(bc="collector"),
            (self.collector_r, self.collector_z),
            PI(maxh=0.1),
            EI(bc="outer_insulator"),
            (self.emitter_r, self.emitter_z),
            PI(maxh=0.1),
            EI(bc="emitter"),
            (self.center_hole_radius, self.emitter_z),
            EI(bc="inner_insulator"),
            PI(maxh=0.1),
        ], mat="insulator")

        return rhombus
#        geo = CSG2d()
#        geo.Add(rhombus)
#        return geo
    if False:
        def apply_boundary_conditions(self, domain):
            """Apply boundary conditions for this aircraft."""
            # Add our boundaries to the domain
            if self.emitter_electrode:
                domain.dirichlet_boundaries.update({
                    'emitter': {'volts': self.emitter_electrode.voltage}
                })
            
            if self.collector_electrode:
                domain.dirichlet_boundaries.update({
                    'collector': {'volts': self.collector_electrode.voltage}
                })
            
            # Also add insulator boundaries if needed (with no voltage)
            domain.dirichlet_boundaries.update({
                'inner_insulator': {'no_flow': True},
                'outer_insulator': {'no_flow': True}
            })
    else:
        def apply_boundary_conditions(self, domain):
            """Apply boundary conditions for this aircraft."""
            # Let electrodes apply their boundary conditions directly
            if self.emitter_electrode:
                self.emitter_electrode.set_dirichlet_bc(domain.phi_pot_gf)
                logging.debug(f"Setting emitter boundary to {self.emitter_electrode.voltage:.2f}V")

            if self.collector_electrode:
                self.collector_electrode.set_dirichlet_bc(domain.phi_pot_gf)
                logging.debug(f"Setting collector boundary to {self.collector_electrode.voltage:.2f}V")

    def on_mesh_generated(self, domain):
        """Called when the domain generates a mesh with this aircraft."""
        logging.debug('In aircraft.on_mesh_generated')
        # Inform electrodes about the mesh and FES
        if self.emitter_electrode:
            logging.debug("Calling on_mesh_generated for emitter electrode")
            self.emitter_electrode.on_mesh_generated(domain)
        
        if self.collector_electrode:
            logging.debug("Calling on_mesh_generated for collector electrode")
            self.collector_electrode.on_mesh_generated(domain)


    def create_geometry(self):
        """
        Create the CSG2d geometric representation of the aircraft for mesh generation.
        
        Returns:
            Solid2d object representing the insulator structure
        """
        # Create a rhombus representing the insulating structure
        rhombus = Solid2d([
            (self.center_hole_radius, self.collector_z),
            PI(maxh=0.1),
            EI(bc="collector"),
            (self.collector_r, self.collector_z),
            PI(maxh=0.1),
            EI(bc="outer_insulator"),
            (self.emitter_r, self.emitter_z),
            PI(maxh=0.1),
            EI(bc="emitter"),
            (self.center_hole_radius, self.emitter_z),
            EI(bc="inner_insulator"),
            PI(maxh=0.1),
        ], mat="insulator")
        
        return rhombus
    
    def set_electrodes(self, emitter_electrode, collector_electrode):
        """Set the electrode objects for the aircraft."""
        self.emitter_electrode = emitter_electrode
        self.collector_electrode = collector_electrode
        
        # Set the aircraft reference in the electrodes
        emitter_electrode.aircraft = self
        collector_electrode.aircraft = self
        
        return self
    
    def update_charge_balance(self, emitted_charge, collected_charge, space_charge):
        """
        Update the charge balance of the aircraft system with a focus
        on maintaining proper voltage relationships.
        
        Args:
            emitted_charge: Change in emitted charge (C)
            collected_charge: Change in collected charge (C)
            space_charge: Current total space charge (C)
        """
        # The emitter loses charge, the collector gains charge
        self.emitter_electrode.charge -= emitted_charge
        self.collector_electrode.charge += collected_charge
        
        # Update space charge accounting
        self.space_charge = space_charge
        
        # Total system charge should remain constant
        self.total_charge = self.emitter_electrode.charge + self.collector_electrode.charge + self.space_charge
        
        # Update electrode voltages based on charges (Q = CV)
        # This will automatically happen through the charge property setters
        # But we can enforce additional constraints here if needed
        
        # Calculate the voltage difference between electrodes
        voltage_difference = self.emitter_electrode.voltage - self.collector_electrode.voltage
        
        # If voltage difference is too small, we might need to reestablish it
        # This simulates an external power supply maintaining the potential difference
        min_voltage_difference = 1000.0  # Minimum 1kV difference to maintain
        
        if abs(voltage_difference) < min_voltage_difference:
            # Calculate how much charge to add to maintain voltage difference
            # This represents external power supply action
            target_difference = min_voltage_difference * np.sign(voltage_difference)
            
            # Calculate additional charge needed
            if self.emitter_electrode.capacitance > 0 and self.collector_electrode.capacitance > 0:
                # For simplicity, split the adjustment evenly between electrodes
                charge_adjustment = (target_difference - voltage_difference) / 2
                
                # Apply adjustments
                emitter_adjustment = charge_adjustment * self.emitter_electrode.capacitance
                collector_adjustment = -charge_adjustment * self.collector_electrode.capacitance
                
                # Update electrode charges
                self.emitter_electrode.charge += emitter_adjustment
                self.collector_electrode.charge += collector_adjustment
                
                # Log this power supply intervention
                power_supply_action = abs(emitter_adjustment) + abs(collector_adjustment)
                logging.debug(f"Power supply intervened with {power_supply_action:.6e}C charge adjustment")
    
    def calculate_thrust(self, mesh, rho_charge_gf, u_gf, RHO_AIR=1.225):
        """
        Calculate the thrust of the aircraft based on the momentum transfer from ions to air.
        This is the physically correct model that accounts for the momentum of the air 
        against which the EHD phenomenon acts.
        
        Args:
            mesh: The computational mesh
            rho_charge_gf: The charge density grid function
            u_gf: The air velocity grid function
            RHO_AIR: Air density (kg/m³)
            
        Returns:
            Current thrust (N)
        """
        # Thrust is primarily from momentum transfer to the air
        # This is reflected in the air velocity field
        
        # Calculate momentum flux through a control surface
        # For axisymmetric coordinates, include 2πr factor
        # Momentum flux = ρ * u * u * A, where ρ is air density
        
        # Define a control surface at the downstream boundary (top of domain)
        # The thrust is primarily in the z-direction (index 1)
        z_momentum_flux = RHO_AIR * u_gf[1] * u_gf[1] * 2 * pi * x
        
        # Integrate over the top boundary or a control surface
        # This gives the total momentum flux which equals thrust
        self.thrust = Integrate(z_momentum_flux, mesh, BND, definedon=mesh.Boundaries("top"))
        
        # For increased accuracy, we can also account for pressure forces
        # But this simplified momentum flux approach captures the main physics
        
        return self.thrust
        
    def update_velocity(self, dt):
        """
        Update aircraft velocity based on thrust.
        
        Args:
            dt: Time step (s)
        """
        acceleration = self.thrust / self.mass
        self.velocity += acceleration * dt
        
    def calculate_system_power(self, dt):
        """
        Calculate total power consumption including both emitter and collector power.
        
        Args:
            dt: Time step (s)
            
        Returns:
            Dictionary with power components and total
        """
        # Calculate voltage difference
        voltage_difference = abs(self.emitter_electrode.voltage - self.collector_electrode.voltage)
        
        # Calculate currents at both electrodes
        # Current = rate of charge change
        emitter_current = abs(self.emitter_electrode.charge_rate) if hasattr(self.emitter_electrode, 'charge_rate') else 0
        collector_current = abs(self.collector_electrode.charge_rate) if hasattr(self.collector_electrode, 'charge_rate') else 0
        
        # If charge rates aren't available, estimate from recent changes
        if emitter_current == 0 and hasattr(self, 'last_emitter_charge'):
            emitter_current = abs(self.emitter_electrode.charge - self.last_emitter_charge) / dt
        
        if collector_current == 0 and hasattr(self, 'last_collector_charge'):
            collector_current = abs(self.collector_electrode.charge - self.last_collector_charge) / dt
        
        # Save current charges for next calculation
        self.last_emitter_charge = self.emitter_electrode.charge
        self.last_collector_charge = self.collector_electrode.charge
        
        # Calculate power components
        # P = I × V for each component
        emitter_power = emitter_current * abs(self.emitter_electrode.voltage)
        collector_power = collector_current * abs(self.collector_electrode.voltage)
        
        # Alternative power calculation based on circuit analysis
        # In a simple circuit model, total power = I × ΔV
        # Using average current through the system
        average_current = (emitter_current + collector_current) / 2
        total_power = average_current * voltage_difference
        
        return {
            "emitter_power": emitter_power,
            "collector_power": collector_power,
            "total_power": total_power,
            "voltage_difference": voltage_difference,
            "emitter_current": emitter_current,
            "collector_current": collector_current
        }
    
    def get_corners(self):
        """
        Get the key corner coordinates of the aircraft structure.
        
        Returns:
            Dictionary of corner coordinates
        """
        return {
            "collector_corner": (self.collector_r, self.collector_z),
            "emitter_corner": (self.emitter_r, self.emitter_z)
        }
    
    def calculate_characteristic_length(self):
        """
        Calculate the characteristic length of the aircraft.
        
        Returns:
            Characteristic length (m)
        """
        # Use distance between emitter and collector along the insulator
        dr = self.emitter_r - self.collector_r
        dz = self.emitter_z - self.collector_z
        return np.sqrt(dr**2 + dz**2)
    
    def calculate_electrode_areas(self):
        """
        Calculate approximate areas of the electrodes.
        
        Returns:
            Dictionary with electrode areas
        """
        # For circular electrodes in 2D axisymmetric geometry
        emitter_area = np.pi * (self.emitter_r**2 - self.center_hole_radius**2)
        collector_area = np.pi * (self.collector_r**2 - self.center_hole_radius**2)
        
        return {
            "emitter_area": emitter_area,
            "collector_area": collector_area
        }
