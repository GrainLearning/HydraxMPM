import jax.numpy as jnp
import jax
import pyperf

import pymudokon as pm



def run_p2g_batch(particles, nodes, shapefunctions):
        nodes = pm.solvers.usl.p2g_batch(
            nodes=nodes,
            particles=particles,
            shapefunctions=shapefunctions,
            dt=0.1,
        )
        return nodes

def run_p2g(particles, nodes, shapefunctions):
        nodes = pm.solvers.usl.p2g(
            nodes=nodes,
            particles=particles,
            shapefunctions=shapefunctions,
            dt=0.1,
        )
        return nodes

def create_system(num_particles, cell_size):
    key = jax.random.key(0)
    positions = jax.random.uniform(key, (num_particles, 2))
    
    particles = pm.Particles.create(positions)
    nodes = pm.Nodes.create(
                origin=jnp.array([0.0, 0.0]),
                end=jnp.array([1.0, 1.0]),
                node_spacing=cell_size,
            )
    shapefunctions = pm.LinearShapeFunction.create(num_particles, 2)

    shapefunctions = shapefunctions.get_interactions(particles, nodes)

    shapefunctions = shapefunctions.calculate_shapefunction(nodes)
    
    return particles, nodes, shapefunctions

runner = pyperf.Runner(loops=2)

runner.warmups = 2       # Number of warm-up runs
runner.samples = 2       # Number of benchmark runs


run_p2g_batch_jitted = jax.jit(run_p2g_batch)

run_p2g_jitted = jax.jit(run_p2g)


for num_particles in [10000]:
    for cell_size in [0.001]:
        particles, nodes, shapefunctions = create_system(num_particles, cell_size)
        runner.bench_func(f'p2g_batch/{num_particles}/{cell_size}', lambda: jax.block_until_ready(run_p2g_batch_jitted(particles, nodes, shapefunctions)))


for num_particles in [10000]:
    for cell_size in [0.001]:
        particles, nodes, shapefunctions = create_system(num_particles, cell_size)
        runner.bench_func(f'p2g/{num_particles}/{cell_size}', lambda: jax.block_until_ready(run_p2g_jitted(particles, nodes, shapefunctions)))
    
# runner = pyperf.Runner()
# for size_pow10 in range(5):
#     size = pow(10, size_pow10)

#     in1 = [4.2 for i in range(size)]
#     in2 = [2.4 for i in range(size)]
#     runner.bench_func(f'sum_append/{size}', lambda: vector_sum_append(in1, in2))