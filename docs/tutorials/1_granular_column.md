<!-- Fix code structure -->
<!-- Fix link to github tutorial -->
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

It is recommended that you follow along with the tutorial. Stuck or in a rush? then see code available in GitHub under [/tutorials/t1_granular_column.py]().

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

Import HydraxMPM and supporting JAX dependencies. 

```python {hl_lines="3"}

--8<-- "t1_granular_column.py:2:5"

```

HydraxMPM is top-level package, containing the bits and pieces under the same namespace, e.g., `hdx.Particles` or `hdx.Gravity`.

## Step 3: create points representing initial granular column

Create rectangular column a rectangular column of material points.

We pad the material body so material points do not touch the boundary. Particles are spaced evenly given a cell size and particles per cell (in one direction).

```python 

--8<-- "t1_granular_column.py:8:23"

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

--8<-- "t1_granular_column.py:24:37"

```
 Ok... it seems like a lot, but actually pretty straight forward:

 - `hdx.MPMConfig` is the config dataclass used for all MPM simulations.
 - `origin` contains start x and y coordinates of the domain boundary
 - `end` contains end x and y coordinates of the domain
 - `project` the name of the project - mainly used for output purposes
 - `ppc` is the number of particles per cell (in one direction). this is only used in `hdx.discretize` function (more on that later)
 - `cell_size` background grid grid cell size
 - `num_points` number of material points
 - `shapefunction` two shapefunctions, either `linear` or `cubic` cubic is better in most cases.
 - `num_steps` counting all iterations to do
 - `store_every` plot every nth step
 - `default_gpu_id` if you are working on a shared GPU workstation, this is the parameter you change to avoid making the other person(s) angry! Run `nvidia-smi`, in your terminal to find the id of an empty GPU. 
 - `dt` constant time step
 - `file=__file__` this records the path of your driver script, which is important to save relative output in the correct folder.

<!-- TODO fix dixtize docs -->
<!-- # add reference to callback etc.? on project -->

??? Tip
    `origin` and `end` will be of len 2 for 2D plane strain (x,y), and len 3 for 3D (x,y,z).

??? Note "why a master config?"
    This is an easy way to ensure, same size arrays are set throughout all dataclasses.


Lets see the summary of the config
```python 

--8<-- "t1_granular_column.py:38:38"

```
If all went well you should get this output:

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



## Step 4: create Material

Lets create a material with some initial bulk density and a particle density

```python 

--8<-- "t1_granular_column.py:40:56"

```

Lets break it down:

1. All basic material e.g., `hdx.ModifiedCamClay`, are initialized in a similar manner
2. Notice we pass the `config` along, this helps to create a static background grid of a fixed size.
3. Constitutive model parameters are chosen at random, and they have a big impact of the simulation.

!!! Tip
    Material may be used in MPM analysis or single integration analysis. Sometimes its good to do both



## Step 5: create Nodes and Particles classes

Ah here we make use of JAX vmap feature. We create a symmetric stress tensor given a hydrostatic pressure.

```python 

--8<-- "t1_granular_column.py:58:61"

```

Initial positions and stresses are used to initialize the `hdx.Particles` class

```python 

--8<-- "t1_granular_column.py:63:65"

```
and background nodes

```python 

--8<-- "t1_granular_column.py:67:67"

```


## Step 6: create Forces

The first force is a slow linear ramp gravity
```python 

--8<-- "t1_granular_column.py:69:74"

```
Boundary so particles do not fall through, we are using 0 friction.
```python 

--8<-- "t1_granular_column.py:75:77"

```



## Step 7: create Solver

The solver is the bread and butter of the code. 
```python 

--8<-- "t1_granular_column.py:75:77"

```




??? Tip
    We recommend using the `hdx.ASFLIP` for granular  materials


## Step 8: Run gravity pack

- See [available callback functions]()




## Step 9: Modify script to incorporate collapse




## Step 10: Granular column collapse

