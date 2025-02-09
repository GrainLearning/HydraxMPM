<!-- Fix code structure -->
<!-- Add learning objective -->
<!-- add images /gifs -->
# Tutorial 1. Granular column collapse
In this tutorial, we'll simulate a very simple basic granular column collapse with the modified Cam-Clay. The simulation involves two parts: (1) gravity pack and (2) collapse.

## Pre-requisites:
Before starting, please make sure that you have completed the following items:

1. Installed the [HydraxMPM](/index/).
2. Comfortable with [JAX - basics](https://jax.readthedocs.io/en/latest/quickstart.html), and  [JAX - The Sharp Bits](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html)
3. Read through an overview of the [code structure]()
4. Have [Paraview](https://www.paraview.org/download/)

It is recommended that you follow along with the tutorial. Stuck or in a rush? then see code available in GitHub under [/tutorials/t1_granular_column.py](https://github.com/GrainLearning/HydraxMPM/blob/main/tutorials/t1_granular_column.py).

## Learning objectives

By the end of this tutorial you will be able to 

- Understand the overall code style of a HydraxMPM script

## Step 1: create the project file and folders

Create a project directory containing the driver script and two sub-folders with as follows:

```
# driver script
/tutorial/t1_granular_column.py  

# output of gravity pack
/tutorial/output/t1_pack/ 

# output of collapse
/tutorial/output/t2_collapse  
```
The output is stored as [vtk](https://docs.vtk.org/en/latest/design_documents/VTKFileFormats.html) files in the `/tutorial/output/t1_pack/` and `/tutorial/output/t2_collapse/` folders, respectively.

## Step 2: import modules

Import HydraxMPM and supporting the JAX dependencies. 

```python {hl_lines="3"}

# --8<-- "t1_granular_column.py:2:5"

```

HydraxMPM is top-level package. It contains all the bits and pieces under the same namespace, e.g., `hdx.Particles` or `hdx.Gravity`.

## Step 3: create points representing initial granular column

Create rectangular column a rectangular column of material points. Particles are spaced evenly given a cell size and the number of particles per cell (in one direction). We pad so that material points do not touch the boundary. 

```python 

# --8<-- "t1_granular_column.py:8:23"

```

??? Tip
    There are several ways of initializing material points. See the [how-to initialize material points](/how-tos/initialize_material_points) .

??? Tip
    [Matplotlib](https://matplotlib.org/) or [Pyvista](https://docs.pyvista.org/) may be used visualize initial positions. To visualize the particles copy and paste the following (note plt.show works only when GUI is active). 
    ```python
    
        import matplotlib.pyplot as plt

        plt.scatter(*position_stack.T ) # unpack x and y and plot
        plt.show() # or plt.savefig("packing.png")
    ```

## Step 4: create the simulation config

This is a juicy sandwich of all common general simulation parameters.

```python 

# --8<-- "t1_granular_column.py:24:37"

```
 Ok... it seems like a lot, but actually pretty straight forward:

 - `hdx.MPMConfig` is the config dataclass used for all MPM simulations.
 - `origin` contains start x and y coordinates of the domain boundary
 - `end` contains end x and y coordinates of the domain
 - `project` is the name of the project - mainly used for output purposes
 - `ppc` is the number of particles per cell (in one direction). this is only used in `hdx.discretize` function (more on that later)
 - `cell_size` is background grid cell size
 - `num_points` is number of material points
 - `shapefunction` the interpolation type from particle to grid. Two shapefunctions are supported, either `linear` or `cubic`. Cubic is better in most cases.
 - `num_steps` the total iteration count
 - `store_every` output every nth step
 - `default_gpu_id` if you are working on a shared GPU workstation, this is the parameter you change to avoid making the other person(s) angry! Run `nvidia-smi`, in your terminal to find the id of an empty GPU. 
 - `dt` is a constant time step
 - `file=__file__` this records the path of your driver script, which is important to save relative output in the correct folder.

<!-- TODO fix dixtize docs -->
<!-- # add reference to callback etc.? on project -->

??? Tip
    `origin` and `end` will be of len 2 for 2D plane strain (x,y), and len 3 for 3D (x,y,z).

??? Note "why a master config?"
    This is an easy way to ensure, same size arrays are set throughout all dataclasses.


Lets see the summary of the config
```python 

# --8<-- "t1_granular_column.py:38:38"

```
If all went well you should get the following output:

```
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
project: tutorial1
dim: 2
num_points: 625
num_cells: 1400
num_interactions: 10000
domain origin: (0, 0)
domain end: (1.399999976158142, 0.6000000238418579)
dt: 3.0000000000000004e-05
total time: 12.000000000000002
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```



## Step 4: create the `Material`

Lets create a material with some initial bulk density and a particle density

<!-- ```python  -->

# --8<-- "t1_granular_column.py:40:44"

<!-- ``` -->

Now the material dataclass

<!-- ```python  -->

# --8<-- "t1_granular_column.py:45:56"

<!-- ``` -->

Lets break this down:

- `hdx.ModifiedCamClay` is the Modified Cam Clay material class
- We always pass `config` along when creating HydraxMPM dataclasses.
- The only non-parameter input is the reference solid volume fraction. 


!!! Tip 
    Materials in HydraxMPM are normally initialize either with a reference pressure or reference solid volume fraction. Given one, we can find the other.

??? Tip
    Material may be used in MPM analysis or single integration analysis. Sometimes its good to do both



## Step 5: create  the `Nodes` and `Particles`  data classes

We initialize particle stresses to match the known pressure state, given a predefined solid-volume fraction above. 


Here we finally make use of JAX [vmap](https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html), to get the stress tensor. 

<!-- ```python  -->

# --8<-- "t1_granular_column.py:58:61"

<!-- ``` -->

Pass all positions and stresses to the `Particles` dataclass.
<!-- ```python  -->

# --8<-- "t1_granular_column.py:63:65"

<!-- ``` -->
We can create background grid nodes via the config.

<!-- ```python  -->

# --8<-- "t1_granular_column.py:67:67"

<!-- ``` -->

The `discretize` function determines initial particle volume by dividing the number of particles in a cell by the cell size.
```python 

# --8<-- "t1_granular_column.py:69:72"

```


## Step 6: create the `Gravity` and `Domain`

Gravity is slowly ramped up 

```python 

# --8<-- "t1_granular_column.py:73:77"

```

Creating the outside domain box.

<!-- ```python 

# --8<-- "t1_granular_column.py:79:81"

``` -->

## Step 7: create the `Solver`

The solver determines how the background grid and material points interact.

<!-- ```python 

# --8<-- "t1_granular_column.py:82:82"

``` -->

??? Tip
    We recommend using the `hdx.ASFLIP` for granular  materials


## Step 8: Run gravity pack

- See [available callback functions]()


<!-- ```python 

# --8<-- "t1_granular_column.py:84:104"

``` -->



## Step 9: Modify script to incorporate collapse



<!-- ```python 

# --8<-- "t1_granular_column.py:108:121"

``` -->


## Step 10: Granular column collapse



<!-- ```python 

# --8<-- "t1_granular_column.py:125:143"

``` -->