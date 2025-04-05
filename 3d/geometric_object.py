from ngsolve import *
class GeometricObject:
    """Base class for objects that have a geometric representation."""
    
    def __init__(self):
        self.mesh = None
        self.contained_objects = []
    
    def create_geometry(self):
        """Create the geometric representation of this object."""
        raise NotImplementedError("Subclasses must implement create_geometry")
    
    def generate_standalone_mesh(self, maxh=1.0):
        """Generate a mesh for this object in isolation."""
        geo = self.create_geometry()
        self.mesh = Mesh(geo.GenerateMesh(maxh=maxh))
        return self.mesh
    
    def add_object(self, obj, position=(0,0), rotation=0, scale=1.0):
        """Add a contained object with transformation."""
        self.contained_objects.append({
            'object': obj,
            'position': position,
            'rotation': rotation,
            'scale': scale
        })
        return self
