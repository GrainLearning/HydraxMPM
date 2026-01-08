# Copyright (c) 2024, Retiefasuarus
# SPDX-License-Identifier: BSD-3-Clause
#
# Part of HydraxMPM: https://github.com/GrainLearning/HydraxMPM

# -*- coding: utf-8 -*-
"""
Explanation:
    This module is basic input and output handling via the `SimIO` class for HydraxMPM simulations.

    It allows saving and loading of simulation states and solver configurations.

    It allows files to be loaded/saved as .eqx files using Equinox's serialization utilities.
"""

import os
import shutil
import json
import equinox as eqx
from datetime import datetime

class SimIO:
    """

    Simulation Input/Output handler for saving and loading simulation states and solver configurations.
    
    Usage:

    Initialize within a directly relative to current file. Then save the solver logic and
    metadata once at the start of the simulation.
    ```python
    recorder = SimIO(output_dir="sim_data", relative_to=__file__, overwrite=True)
    recorder.save_solver(solver, meta_info={"description": "Test simulation"})  
    ```

    In the main loop it can be called to save the solver state periodically
    ```python

    def log_output(state):
        recorder.save_step(state)

    # log_output called for example by a jax.debug.callback within the main simulation loop

    ```

    The state can be loaded later using

    ```python

        #... build script logic to create skeletons

        recorder = SimIO(output_dir="sim_data", relative_to=__file__)
        sim_state = recorder.load_simulation(directory="sim_data", 
                                            solver_skeleton=solver_skeleton,
                                            state_skeleton=state_skeleton,
                                            step=100)
    """

    def __init__(self, 
                 output_dir="output_data", 
                 relative_dir=None,
                 overwrite=True):
        """
        Create `SimIo` object to save files
        Args:
            output_dir: Folder to save .eqx files.
            relative_to: Pass __file__ to save relative to the script.
            overwrite: If True, deletes the folder before starting (be careful!).


        """
        # Set path
        if relative_dir:
            base_path = os.path.dirname(os.path.abspath(relative_dir))
            self.output_dir = os.path.join(base_path, output_dir)
        else:
            self.output_dir = output_dir

        # Setup folder
        if os.path.exists(self.output_dir) and overwrite:
            print(f"Cleaning existing data directory: {self.output_dir}")
            shutil.rmtree(self.output_dir)
        
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"Recorder initialized at: {self.output_dir}")
    
    def save_solver(self, solver, meta_info=None):
        """
        Saves the static configuration (Grid, Constitutive Models, etc.).
        
        Run this ONCE at the beginning.

        Args:
            solver: Solver instance to save.
            meta_info: Optional dictionary of extra metadata to save as JSON.
        """
        path = os.path.join(self.output_dir, "solver_config.eqx")
        eqx.tree_serialise_leaves(path, solver)
        
        # Optionally save human readable metadata
        meta = {
            "timestamp": str(datetime.now()),
            "type": type(solver).__name__
        }
        if meta_info:
            meta.update(meta_info)
            
        with open(os.path.join(self.output_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=4)
            
        print(f"Saved Solver Configuration to solver_config.eqx")

    def save_step(self, state, verbose=True):
        """
        Saves the full simulation state (particles, grid nodes, time).

        Run this every N steps within the main loop.
        
        Args:
            state: SimState instance to save.
            verbose: If True, prints a confirmation message.
        """
        step = int(state.step)
        filename = f"state_{int(step):05d}.eqx"
        path = os.path.join(self.output_dir, filename)
        
        # This saves arrays compressed and fast
        eqx.tree_serialise_leaves(path, state)


        if verbose:
            print(f"Saved simulation state to {filename}")


def load_simulation(directory, solver_skeleton, state_skeleton, step):
    """
    To load JAX data, you need the 'skeleton' (an instance of the class)
    to know the structure, then we fill it with the saved data.

    Note it is recommended to create the skeletons from the same code
    to ensure compatibility.

    """
    # 1. Load Solver
    solver_path = os.path.join(directory, "solver_config.eqx")
    loaded_solver = eqx.tree_deserialise_leaves(solver_path, solver_skeleton)
    
    # 2. Load State
    state_path = os.path.join(directory, f"state_{int(step):05d}.eqx")
    loaded_state = eqx.tree_deserialise_leaves(state_path, state_skeleton)
    
    return loaded_solver, loaded_state