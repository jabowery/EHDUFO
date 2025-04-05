import os
import numpy as np
from ngsolve import *
import time
import json

class FieldOutputManager:
    """
    Class for managing output of NGSolve field data to files.
    
    Organizes output by field name in separate directories and saves each timestep.
    Designed as an alternative to VTK/ParaView for NGSolve vector/scalar fields.
    """
    
    def __init__(self, fields_dict, output_dir="field_output", mesh=None):
        """
        Initialize the output manager.
        
        Parameters:
        -----------
        fields_dict : dict
            Dictionary with keys as field names and values as NGSolve GridFunctions or CoefficientFunctions
        output_dir : str
            Base directory for output (will be created if it doesn't exist)
        mesh : Mesh, optional
            NGSolve mesh object. If None, will try to extract from GridFunctions
        """
        self.fields_dict = fields_dict
        self.output_dir = output_dir
        self.timestep = 0
        self.last_output_time = None
        self.mesh = mesh
        
        # If mesh not provided, try to get it from GridFunctions
        if self.mesh is None:
            for field in fields_dict.values():
                if hasattr(field, 'mesh'):
                    self.mesh = field.mesh
                    break
                elif hasattr(field, 'space') and hasattr(field.space, 'mesh'):
                    self.mesh = field.space.mesh
                    break
        
        if self.mesh is None:
            raise ValueError("Could not find mesh in any of the provided fields. Please provide the mesh explicitly.")
        
        # Create directory structure
        self._create_directories()
        
        # Save mesh information
        self._save_mesh_info()

    def _create_directories(self):
        """Create the main output directory and subdirectories for each field."""
        # Create main directory
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # Create subdirectory for each field
        for field_name in self.fields_dict.keys():
            field_dir = os.path.join(self.output_dir, field_name)
            if not os.path.exists(field_dir):
                os.makedirs(field_dir)

    def _save_mesh_info(self):
        """Save basic mesh information for later reference."""
        mesh_info = {
            "num_elements": self.mesh.ne,
            "num_vertices": self.mesh.nv,
            "dim": self.mesh.dim,
            "materials": self.mesh.GetMaterials(),
            "boundaries": self.mesh.GetBoundaries()
        }
        
        # Save vertex coordinates
        points = []
        for i in range(self.mesh.nv):
            point = self.mesh.vertices[i].point
            # Handle both tuple and ndarray types
            if hasattr(point, 'tolist'):
                point_list = point.tolist()
            else:
                # If it's already a tuple or list, convert directly
                point_list = list(point)
            points.append(point_list)
        
        mesh_info["vertices"] = points
        
        # Save element connectivity
        elements = []
        for el in self.mesh.Elements():
            elements.append([v.nr for v in el.vertices])
        
        mesh_info["elements"] = elements
        
        # Save to JSON
        mesh_info_path = os.path.join(self.output_dir, "mesh_info.json")
        with open(mesh_info_path, 'w') as f:
            json.dump(mesh_info, f, indent=2)
    
    def save_fields(self, t=None):
        """
        Save the current state of all fields.
        
        Parameters:
        -----------
        t : float, optional
            Current simulation time. If None, uses the timestep count.
        """
        current_time = time.time()
        
        # If this is the first output or we have no record of last output time
        if self.last_output_time is None:
            self.last_output_time = current_time
        
        if t is None:
            t = self.timestep
        
        # Process each field
        for field_name, field in self.fields_dict.items():
            field_dir = os.path.join(self.output_dir, field_name)
            
            # Determine output filename
            filename = f"{field_name}_t{t:.6f}.npz"
            output_path = os.path.join(field_dir, filename)
            
            # Convert field to NumPy array
            if isinstance(field, GridFunction):
                # Check if it's a vector field
                if field.dim > 1:
                    # Extract components
                    components = []
                    for i in range(field.dim):
                        components.append(field.components[i].vec.FV().NumPy())
                    
                    # Save as vector field
                    np.savez(output_path, 
                             components=np.array(components),
                             is_vector=True,
                             dim=field.dim,
                             time=t)
                else:
                    # Save as scalar field
                    np.savez(output_path, 
                             values=field.vec.FV().NumPy(),
                             is_vector=False,
                             time=t)
            else:
                # For coefficient functions, sample at mesh vertices
                # This is a simplified version - may need adjustment for specific cases
                try:
                    # For evaluating coefficient functions, use mesh points
                    mesh_points = []
                    for i in range(self.mesh.nv):
                        mesh_points.append(self.mesh.vertices[i].point)
                    
                    # Create MeshPoints for evaluation
                    try:
                        # NGSolve 6.2.2102 and later
                        from ngsolve import MeshPoints
                        mp = MeshPoints(self.mesh, mesh_points)
                    except ImportError:
                        # Fallback for older NGSolve versions
                        mp = [self.mesh(p) for p in mesh_points]
                    
                    # Try to evaluate at the first point to determine field type
                    if isinstance(mp, list):
                        test_eval = field(mp[0])
                    else:
                        # Using MeshPoints - evaluate the whole field at once
                        all_values = field(mp)
                        if hasattr(all_values, 'shape') and len(all_values.shape) > 1:
                            # It's a vector field
                            np.savez(output_path, 
                                    components=all_values,
                                    is_vector=True,
                                    dim=all_values.shape[0],
                                    time=t)
                            continue
                        else:
                            # It's a scalar field
                            np.savez(output_path, 
                                    values=all_values,
                                    is_vector=False,
                                    time=t)
                            continue
                    
                    # For older versions or if MeshPoints evaluation didn't work
                    if isinstance(test_eval, tuple) or (hasattr(test_eval, '__len__') and not isinstance(test_eval, str)):
                        # It's a vector field
                        dim = len(test_eval)
                        values = np.zeros((dim, len(mesh_points)))
                        
                        for i, mp_i in enumerate(mp):
                            values_at_point = field(mp_i)
                            for j in range(dim):
                                values[j, i] = values_at_point[j]
                        
                        np.savez(output_path, 
                                components=values,
                                is_vector=True,
                                dim=dim,
                                time=t)
                    else:
                        # It's a scalar field
                        values = np.zeros(len(mesh_points))
                        
                        for i, mp_i in enumerate(mp):
                            values[i] = field(mp_i)
                        
                        np.savez(output_path, 
                                values=values,
                                is_vector=False,
                                time=t)
                except Exception as e:
                    print(f"Warning: Could not evaluate field {field_name} at mesh vertices: {e}")
                    continue
        
        # Update counters
        self.timestep += 1
        self.last_output_time = current_time
        
    def save_additional_data(self, data_dict, t=None):
        """
        Save additional numerical data like thrust, voltage, etc.
        
        Parameters:
        -----------
        data_dict : dict
            Dictionary with keys as data names and values as scalar values
        t : float, optional
            Current simulation time. If None, uses the timestep count.
        """
        if t is None:
            t = self.timestep
            
        # Create a directory for scalar data if it doesn't exist
        scalar_dir = os.path.join(self.output_dir, "scalar_data")
        if not os.path.exists(scalar_dir):
            os.makedirs(scalar_dir)
            
        # Append to the time series file for each scalar value
        for name, value in data_dict.items():
            # Use a CSV format for easy plotting in other tools
            filename = f"{name}.csv"
            filepath = os.path.join(scalar_dir, filename)
            
            # Check if file exists to write header
            file_exists = os.path.exists(filepath)
            
            with open(filepath, 'a') as f:
                if not file_exists:
                    f.write("time,value\n")
                f.write(f"{t},{value}\n")

    def create_metadata_file(self):
        """Create a metadata file with information about the saved fields."""
        metadata = {
            "fields": list(self.fields_dict.keys()),
            "total_timesteps": self.timestep,
            "output_directory": self.output_dir
        }
        
        # Add field-specific information
        field_info = {}
        for field_name, field in self.fields_dict.items():
            if isinstance(field, GridFunction):
                field_info[field_name] = {
                    "type": "GridFunction",
                    "dimension": field.dim,
                    "space": str(field.space),
                }
            else:
                field_info[field_name] = {
                    "type": "CoefficientFunction",
                }
        
        metadata["field_info"] = field_info
        
        # Save to JSON
        metadata_path = os.path.join(self.output_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

# Example usage:
'''
# Create instance with fields to save - explicitly pass the mesh
field_manager = FieldOutputManager({
    "potential": phi_pot_gf,
    "charge_density": rho_charge,
    "velocity": u_gf,
    "electric_field": E_field
}, output_dir="ehd_output", mesh=mesh)

# Inside time loop:
field_manager.save_fields(t=t)

# Save additional data
field_manager.save_additional_data({
    "thrust": thrust,
    "vehicle_velocity": u_vehicle,
    "voltage": emitter.voltage - collector.voltage
}, t=t)

# At the end of simulation:
field_manager.create_metadata_file()
'''
