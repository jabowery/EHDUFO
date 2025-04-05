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

# Create domain and objects
domain = EHDDomain(outer_r=20.0, outer_z=40.0, boundary_voltage=0.0)
aircraft = EHDAircraft(emitter_radius=0.1, collector_radius=5.0)
emitter = Electrode("emitter", initial_voltage=-20e3)
collector = Electrode("collector", initial_voltage=20e3)

# Set up relationships
if False:
    aircraft.set_electrodes(emitter, collector)
    domain.add_object(aircraft, position=(0, 40/2 - 3/2))
else:
    aircraft.set_electrodes(emitter, collector)
    # Adjust aircraft position to better center it in the domain
    domain.add_object(aircraft, position=(domain.outer_r/4, domain.outer_z/2))

    # Use a finer mesh especially near the electrodes
    domain.generate_composite_mesh(maxh=0.5)

    # Print aircraft positions for verification
    print(f"Aircraft geometry:")
    print(f"  Emitter: r={aircraft.emitter_r}, z={aircraft.emitter_z}")
    print(f"  Collector: r={aircraft.collector_r}, z={aircraft.collector_z}")

# Generate mesh, FES, and GFs
domain.generate_composite_mesh()


# ----------------------------------------------------------------------
# Set up field output manager
# ----------------------------------------------------------------------
# Ensure output directory exists
os.makedirs("ehd_output", exist_ok=True)

field_manager = FieldOutputManager({
    "potential": domain.phi_pot_gf,
    "charge_density": domain.rho_charge_gf,
    "velocity": domain.u_gf,
    "electric_field": domain.E_field
}, output_dir="ehd_output", mesh=domain.mesh)

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

# ==== Time Stepping Loop ====
t = 0
step = 0


while t < T_FINAL:
    # 1. Solve Poisson's equation
    domain.phi_pot_gf = solve_poisson_equation(
        domain.mesh, domain.fes_pot, domain.rho_charge_gf, domain.epsilon_0)
    
    # Reapply boundary conditions
    domain.apply_all_boundary_conditions()
    
    # 2. Calculate electric field
    E_field = domain.compute_electric_field()
    
    # 3. Emit charge from emitter
    emitted_charge = emitter.emit(domain.rho_charge_gf, dt=DT)
    
    # 4. Calculate ion velocity and collect charge
    ion_velocity = domain.ion_mobility * E_field + domain.u_gf
    collected_charge = collector.collect_charge(domain.rho_charge_gf, ion_velocity, DT)
    
    # 5. Update aircraft charge balance
    space_charge = Integrate(domain.rho_charge_gf, domain.mesh)
    aircraft.update_charge_balance(emitted_charge, collected_charge, space_charge)

    # 6. Solve charge transport equation
    domain.rho_charge_gf = solve_charge_transport(
        domain.rho_charge_gf,
        domain.mesh,
        domain.fes_rho,
        DT,
        domain.ion_mobility,
        domain.E_field,
        domain.u_gf,
        aircraft.velocity,
        diffusivity=domain.ion_diffusivity,
        min_ion_density=domain.min_ion_density
    )

    # 7. Solve Navier-Stokes for fluid velocity
    domain.u_gf = solve_navier_stokes(
        domain.mesh,
        domain.fes_vel,
        domain.u_gf,
        domain.rho_charge_gf,
        domain.E_field,
        DT,
        rho_air=domain.rho_air,
        mu_air=domain.mu_air
    )

    # 8. Calculate thrust and update aircraft velocity
    thrust = aircraft.calculate_thrust(domain.mesh, domain.rho_charge_gf, domain.u_gf)
    aircraft.update_velocity(DT)

    # Store history data
    time_history.append(t)
    thrust_history.append(thrust)
    power_data = aircraft.calculate_system_power(DT)
    power_history.append(power_data["total_power"])
    voltage_history.append((emitter.voltage, collector.voltage))
    
    # Output results at specified intervals
    if step % OUTPUT_INTERVAL == 0:
        # Save field data
        field_manager.save_fields(t=t)
        
        # Create potential profile
        domain.create_potential_profile(domain.mesh, domain.phi_pot_gf, emitter, collector, t, aircraft)
        
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
                domain.mesh,
                domain.phi_pot_gf,
                aircraft.emitter_r / 2,
                aircraft.emitter_z + 0.05
            )

            # Sample below the collector
            collector_sample_point = domain.sample_potential_at_point(
                domain.mesh,
                domain.phi_pot_gf,
                aircraft.collector_r / 2,
                aircraft.collector_z - 0.05
            )

            # Sample in the middle
            middle_sample_point = domain.sample_potential_at_point(
                domain.mesh,
                domain.phi_pot_gf,
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
