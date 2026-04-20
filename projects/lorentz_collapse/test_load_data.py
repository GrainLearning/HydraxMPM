import zarr
import numpy as np

import os
base_path = os.path.dirname(os.path.abspath(__file__))

def load_simulation_as_dict(zarr_path):
    """
    Loads a Zarr simulation store into a standard Python dictionary.
    No JAX, Equinox, or Solver required.
    """
    store = zarr.open(zarr_path, mode='r')
    
    def recursive_load(group):
        data = {}
        # Iterate through arrays in this group
        for name, array in group.arrays():
            data[name] = np.array(array)
            
        # Iterate through sub-groups (folders)
        for name, subgroup in group.groups():
            data[name] = recursive_load(subgroup)
        return data

    return recursive_load(store)



path=os.path.join(base_path, "lorentz_collapse_trajectories.zarr")

root = zarr.open(path, mode='r')
print(root.tree())



results = load_simulation_as_dict(path)
