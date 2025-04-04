from ngsolve import *
import pyngcore as ngcore
import matplotlib.pyplot as plt
import numpy as np
import os
from field_output import FieldOutputManager

# Import our new classes
from aircraft_ehd import EHDAircraft
from domain_ehd import EHDDomain
from electrode_ehd import Electrode
from charge_transport_ehd import (
    solve_charge_transport, 
    solve_poisson_equation, 
    solve_navier_stokes
)

# Set up multi-threading
ngcore.SetNumThreads(20)

# ----------------------------------------------------------------------
# Physical constants and simulation parameters
# ----------------------------------------------------------------------
# Set up default values
EPSILON_0 = 8.854e-12  # Vacuum permittivity [F/m]
ION_MOBILITY = 2e-4    # Ion mobility [m^2/(V·s)]
ION_DIFFUSIVITY = 5e-5 # Ion diffusion coefficient [m^2/s]
RHO_AIR = 1.225        # Air density [kg/m^3]
MU_AIR = 1.8e-5        # Air dynamic viscosity [Pa·s]
MIN_ION_DENSITY = 1e-14 # Minimum ion density [C/m^3]
ION_NEUTRAL_COLLISION_FREQ = 1e9 # Collision frequency between ions and neutrals [1/s]
EMISSION_COEFFICIENT = 1e-9 # Base emission rate coefficient [C/V/s]
FIELD_ENHANCEMENT_FACTOR = 1e3 # Field enhancement factor due to electrode geometry
EMITTER_CAPACITANCE = 1e-11 # Emitter capacitance [F]
COLLECTOR_CAPACITANCE = 1e-11 # Collector capacitance [F]
ELECTRODE_VOLTAGE = 20e3 # Magnitude of electrode voltage [V]

# Initial voltages with correct physical setup:
# - Emitter at negative voltage to emit negative ions
# - Collector at positive voltage to attract negative ions
# - Domain boundary at zero (ground) voltage
INITIAL_EMITTER_VOLTAGE = -ELECTRODE_VOLTAGE
INITIAL_COLLECTOR_VOLTAGE = ELECTRODE_VOLTAGE

# Time stepping parameters
DT = 1e-4  # Time step [s]
T_FINAL = 0.01  # Final simulation time [s]
OUTPUT_INTERVAL = 10  # Save output every N time steps

# ----------------------------------------------------------------------
# Domain and aircraft setup
# ----------------------------------------------------------------------
# Create domain
domain = EHDDomain(
    outer_r=20.0,
    outer_z=40.0,
    epsilon_0=EPSILON_0,
    ion_mobility=ION_MOBILITY,
    ion_diffusivity=ION_DIFFUSIVITY,
    rho_air=RHO_AIR,
    mu_air=MU_AIR,
    min_ion_density=MIN_ION_DENSITY,
    ion_neutral_collision_freq=ION_NEUTRAL_COLLISION_FREQ
)

# Create aircraft
aircraft = EHDAircraft(
    mesh=None,  # Will be set after mesh generation
    emitter_radius=0.1,
    collector_radius=5.0,
    center_hole_radius=0.0,
    height=3.0,
    base_z=40.0/2 - 3.0/2,  # Center the aircraft in the domain
    initial_emitter_voltage=INITIAL_EMITTER_VOLTAGE,
    initial_collector_voltage=INITIAL_COLLECTOR_VOLTAGE
)

# Generate mesh
mesh = domain.generate_mesh(aircraft, emitter_gridsize=0.1)

# Update aircraft with mesh
aircraft.mesh = mesh

# ----------------------------------------------------------------------
# Finite element spaces
# ----------------------------------------------------------------------
# Create finite element spaces
fes_pot = H1(mesh, order=2, dirichlet="collector|emitter|right")
fes_rho = H1(mesh, order=1)
fes_vel = VectorH1(mesh, order=2, 
                  dirichlet="collector|emitter|outer_insulator|inner_insulator|right")

# ----------------------------------------------------------------------
# Initialize electrodes
# ----------------------------------------------------------------------
# Create electrodes with proper capacitance values and opposite voltages
# The emitter should be at negative voltage, collector at positive voltage,
# and domain boundaries at zero voltage
emitter = Electrode(
    mesh, 
    fes_rho, 
    "emitter", 
    initial_voltage=INITIAL_EMITTER_VOLTAGE,  # Negative voltage for emitter
    capacitance=EMITTER_CAPACITANCE
)

collector = Electrode(
    mesh, 
    fes_rho, 
    "collector", 
    initial_voltage=INITIAL_COLLECTOR_VOLTAGE,   # Positive voltage for collector
    capacitance=COLLECTOR_CAPACITANCE
)

# Set electrodes on aircraft
aircraft.set_electrodes(emitter, collector)

# ----------------------------------------------------------------------
# Initialize variables
# ----------------------------------------------------------------------
# Initialize trial and test functions
phi_pot, v_phi_pot = fes_pot.TnT()
rho_charge, v_rho_charge = fes_rho.TnT()
u, v = fes_vel.TnT()

# Initialize charge density with minimum value
rho_charge_gf = GridFunction(fes_rho)
rho_charge_gf.Set(MIN_ION_DENSITY)

# Initialize potential field
phi_pot_gf = GridFunction(fes_pot)

# Set Dirichlet boundary conditions for potential
# Emitter: negative voltage
emitter.set_dirichlet_bc(phi_pot_gf)
# Collector: positive voltage
collector.set_dirichlet_bc(phi_pot_gf)
# Domain boundary: zero voltage
phi_pot_gf.Set(0.0, definedon=mesh.Boundaries("right"))

# Initialize velocity field
u_gf = GridFunction(fes_vel)
u_gf.Set(CoefficientFunction((0, 0)))

# Get epsilon coefficient function
epsilon = domain.get_epsilon()

# ----------------------------------------------------------------------
# Set up field output manager
# ----------------------------------------------------------------------
# Ensure output directory exists
os.makedirs("ehd_output", exist_ok=True)

# Initialize VTK output
field_manager = FieldOutputManager({
    "potential": phi_pot_gf,
    "charge_density": rho_charge_gf,
    "velocity": u_gf,
    "electric_field": -grad(phi_pot_gf)
}, output_dir="ehd_output", mesh=mesh)

field_manager.create_metadata_file()

# Print initial voltage configuration
print(f"Initial voltage configuration:")
print(f"  Emitter voltage: {emitter.voltage:.2f}V")
print(f"  Collector voltage: {collector.voltage:.2f}V")
print(f"  Voltage difference: {abs(emitter.voltage - collector.voltage):.2f}V")
print(f"  Domain boundary voltage: 0.0V")

# ----------------------------------------------------------------------
# Time stepping loop
# ----------------------------------------------------------------------
t = 0
step = 0
total_emitted_charge = 0
total_collected_charge = 0
power_history = []
thrust_history = []
time_history = []
voltage_history = []

print("Starting simulation...")
print(f"Initial conditions:")
print(f"  Emitter voltage: {emitter.voltage:.2f} V")
print(f"  Collector voltage: {collector.voltage:.2f} V")
print(f"  Emitter charge: {emitter.charge:.6e} C")
print(f"  Collector charge: {collector.charge:.6e} C")

while t < T_FINAL:
    # -----------------------------------------------------------------
    # 1. Solve Poisson's equation for potential based on current charge
    # -----------------------------------------------------------------
    phi_pot_gf = solve_poisson_equation(mesh, fes_pot, rho_charge_gf, epsilon)
    
    # Set Dirichlet boundary conditions for electrodes
    # This must be done after solving to ensure proper voltage values are applied
    emitter.set_dirichlet_bc(phi_pot_gf)
    collector.set_dirichlet_bc(phi_pot_gf)
    phi_pot_gf.Set(0.0, definedon=mesh.Boundaries("right"))
    
    # -----------------------------------------------------------------
    # 2. Calculate electric field from potential
    # -----------------------------------------------------------------
    E_field = -grad(phi_pot_gf)
    E_norm = sqrt(E_field[0]**2 + E_field[1]**2)
    
    # -----------------------------------------------------------------
    # 3. Emit charge from emitter electrode into space charge
    # -----------------------------------------------------------------
    # Calculate electric field at emitter for field emission model
    E_field_at_emitter = emitter.calculate_average_field(E_field, offset=0.01)
    
    # Emit charge from emitter electrode
    emitted_charge = emitter.emit(
        rho_charge_gf, 
        dt=DT, 
        emission_coefficient=EMISSION_COEFFICIENT,
        field_enhancement_factor=FIELD_ENHANCEMENT_FACTOR * E_field_at_emitter/1e6
    )
    
    total_emitted_charge += emitted_charge
    
    # -----------------------------------------------------------------
    # 4. Collect charge at collector electrode from space charge
    # -----------------------------------------------------------------
    # Calculate ion velocity (mobility * E-field + fluid velocity)
    ion_velocity = ION_MOBILITY * E_field + u_gf

    # Account for vehicle motion in axisymmetric coordinates
    if abs(aircraft.velocity) > 1e-10:
        ion_velocity = CoefficientFunction((
            ion_velocity[0], 
            ion_velocity[1] - aircraft.velocity
        ))

    # Collect charge using direct flux calculation
    collected_charge = collector.collect_charge(
        rho_charge_gf, 
        ion_velocity, 
        DT,
        min_ion_density=MIN_ION_DENSITY
    )
    total_collected_charge += collected_charge

    # Update aircraft charge balance
    space_charge = Integrate(rho_charge_gf, mesh)
    aircraft.update_charge_balance(emitted_charge, collected_charge, space_charge)

    # Calculate voltage difference between electrodes
    voltage_difference = abs(emitter.voltage - collector.voltage)

    # Calculate power based on current and voltage difference
    current = collected_charge / DT  # Use collected charge for current calculation
    power = voltage_difference * abs(current)  # Power is always positive
    
    # -----------------------------------------------------------------
    # 5. Solve transport equation for charge density
    # -----------------------------------------------------------------
    rho_charge_gf = solve_charge_transport(
        rho_charge_gf, 
        mesh, 
        fes_rho, 
        DT, 
        ION_MOBILITY,
        E_field, 
        u_gf, 
        aircraft.velocity,
        diffusivity=ION_DIFFUSIVITY,
        min_ion_density=MIN_ION_DENSITY
    )
    
    # -----------------------------------------------------------------
    # 6. Update potential again after charge redistribution
    # -----------------------------------------------------------------
    phi_pot_gf = solve_poisson_equation(mesh, fes_pot, rho_charge_gf, epsilon)
    
    # Reapply boundary conditions
    emitter.set_dirichlet_bc(phi_pot_gf)
    collector.set_dirichlet_bc(phi_pot_gf)
    phi_pot_gf.Set(0.0, definedon=mesh.Boundaries("right"))
    
    # -----------------------------------------------------------------
    # 7. Calculate fluid velocity (air flow)
    # -----------------------------------------------------------------
    u_gf = solve_navier_stokes(
        mesh, 
        fes_vel, 
        u_gf, 
        rho_charge_gf, 
        E_field, 
        DT,
        rho_air=RHO_AIR,
        mu_air=MU_AIR,
        ion_neutral_collision_freq=ION_NEUTRAL_COLLISION_FREQ
    )
    
    # -----------------------------------------------------------------
    # 8. Calculate thrust and update aircraft velocity
    # -----------------------------------------------------------------
    thrust = aircraft.calculate_thrust(mesh, rho_charge_gf, u_gf)
    aircraft.update_velocity(DT)
    
    # -----------------------------------------------------------------
    # 9. Calculate complete power data
    # -----------------------------------------------------------------
    power_data = aircraft.calculate_system_power(DT)
    
    # Store history data
    time_history.append(t)
    thrust_history.append(thrust)
    power_history.append(power_data["total_power"])
    voltage_history.append((emitter.voltage, collector.voltage))
    
    # Output results at specified intervals
    if step % OUTPUT_INTERVAL == 0:
        # Save field data
        field_manager.save_fields(t=t)
        
        # Create potential profile
        domain.create_potential_profile(mesh, phi_pot_gf, emitter, collector, t, aircraft)
        
        # Output status
        print(f"t={t:.5f}s, "
              f"Emitter V={emitter.voltage:.2f}V, "
              f"Collector V={collector.voltage:.2f}V, "
              f"ΔV={power_data['voltage_difference']:.2f}V, "
              f"E charge={emitter.charge:.6e}C, "
              f"C charge={collector.charge:.6e}C, "
              f"Space charge={space_charge:.6e}C, "
              f"Total charge={aircraft.total_charge:.6e}C, "
              f"E current={power_data['emitter_current']:.6e}A, "
              f"C current={power_data['collector_current']:.6e}A, "
              f"Power={power_data['total_power']:.2f}W, "
              f"Thrust={thrust:.6f}N, "
              f"Velocity={aircraft.velocity:.6f}m/s"
              )
        
        # Sample potential at specific points
        try:
            # Sample above the emitter
            emitter_sample_point = domain.sample_potential_at_point(
                mesh, 
                phi_pot_gf, 
                aircraft.emitter_r / 2, 
                aircraft.emitter_z + 0.05
            )
            
            # Sample below the collector
            collector_sample_point = domain.sample_potential_at_point(
                mesh, 
                phi_pot_gf,
                aircraft.collector_r / 2,
                aircraft.collector_z - 0.05
            )
            
            # Sample in the middle
            middle_sample_point = domain.sample_potential_at_point(
                mesh,
                phi_pot_gf,
                (aircraft.emitter_r + aircraft.collector_r) / 4,
                (aircraft.emitter_z + aircraft.collector_z) / 2
            )
            
            print(f"  Potential above emitter: {emitter_sample_point:.2f}V, "
                  f"  Potential below collector: {collector_sample_point:.2f}V, "
                  f"  Potential in middle: {middle_sample_point:.2f}V")
        except Exception as e:
            print(f"  Warning: Could not sample potential at points: {e}")
    
    # Update time and step counter
    t += DT
    step += 1

# ----------------------------------------------------------------------
# Generate summary plots
# ----------------------------------------------------------------------
# Plot thrust history
plt.figure(figsize=(10, 6))
plt.plot(time_history, thrust_history, 'b-', linewidth=2)
plt.title('Thrust vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Thrust (N)')
plt.grid(True)
plt.savefig('ehd_output/thrust_history.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot power history
plt.figure(figsize=(10, 6))
plt.plot(time_history, power_history, 'r-', linewidth=2)
plt.title('Power Consumption vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Power (W)')
plt.grid(True)
plt.savefig('ehd_output/power_history.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot voltage history
plt.figure(figsize=(10, 6))
emitter_voltage = [v[0] for v in voltage_history]
collector_voltage = [v[1] for v in voltage_history]
plt.plot(time_history, emitter_voltage, 'b-', linewidth=2, label='Emitter')
plt.plot(time_history, collector_voltage, 'r-', linewidth=2, label='Collector')
plt.title('Electrode Voltages vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.legend()
plt.grid(True)
plt.savefig('ehd_output/voltage_history.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot efficiency (thrust/power)
plt.figure(figsize=(10, 6))
efficiency = [t/max(p, 1e-10) for t, p in zip(thrust_history, power_history)]
plt.plot(time_history, efficiency, 'g-', linewidth=2)
plt.title('Thrust Efficiency (N/W) vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Efficiency (N/W)')
plt.grid(True)
plt.savefig('ehd_output/efficiency_history.png', dpi=300, bbox_inches='tight')
plt.close()

print("Simulation completed successfully.")
print(f"Final results:")
print(f"  Simulation time: {t:.5f}s")
print(f"  Emitter voltage: {emitter.voltage:.2f}V")
print(f"  Collector voltage: {collector.voltage:.2f}V")
print(f"  Total emitted charge: {total_emitted_charge:.6e}C")
print(f"  Total collected charge: {total_collected_charge:.6e}C")
print(f"  Final thrust: {thrust:.6f}N")
print(f"  Final aircraft velocity: {aircraft.velocity:.6f}m/s")
print(f"  Final power consumption: {power:.2f}W")
print(f"  Average efficiency: {np.mean(efficiency):.6f}N/W")
