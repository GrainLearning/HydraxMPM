import jax
import jax.numpy as jnp
import pytest
import hydraxmpm as hdx


def test_create_state():
    model = hdx.ModifiedCamClay(
        nu=0.3, M=1.0, lam=0.2, kap=0.05, gamma=2.0
    )
    
    mp_state = hdx.MaterialPointState.create(
        position_stack=jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        mass_stack =jnp.array([[1000.0], [2000.0]]), 
        volume_stack=jnp.array([[1.0], [2.0]]),
        pressure_stack=jnp.array([[1000], [1000]])
    )

    p_s = jnp.array([[1000.0], [1000.0]])
    
    state = model.create_state(mp_state, p_s_stack=p_s)
    assert jnp.allclose(state.p_s_stack, p_s)
    assert jnp.allclose(state.stress_ref_stack, mp_state.stress_stack)

def test_mcc_precondition_case1():
    # Case 1: Input p and p_s -> output v and q
    mcc = hdx.ModifiedCamClay(
        nu=0.3,
        M=1.0,
        lam=0.2,
        kap=0.05,
        gamma=2.0,
        p_ref=1000.0
    )
    
    p = jnp.array([[2000.0]])
    p_s = jnp.array([[4000.0]])
    
    p_s_out, v_out, q_out = mcc.precondition(p_stack=p, p_s_stack=p_s)
    
    # Check expected calculations against product function modules
    v_csl = mcc.gamma * (mcc.p_ref / p_s)**mcc.lam
    v_elas = (p_s / p)**mcc.kap
    v_expected = v_csl * v_elas
    
    v_csl = 2.0 * (1000.0 / 4000.0)**0.2
    v_elas = (4000.0 / 2000.0)**0.05
    v_expected = v_csl * v_elas
    
    # q_p = M * sqrt(2*(p_s/p) - 1)
    q_p = 1.0 * jnp.sqrt(2 * (4000.0 / 2000.0) - 1)
    # q_expected = q_p * p
    q_expected = q_p * 2000.0

        
    assert jnp.allclose(v_out, v_expected)
    assert jnp.allclose(q_out, q_expected)
    assert jnp.allclose(p_s_out, p_s)


def test_mcc_precondition_case2():
    # Case 2: Input p and q -> output p_s and v
    mcc = hdx.ModifiedCamClay(
        nu=0.3,
        M=1.0,
        lam=0.2,
        kap=0.05,
        gamma=2.0,
        p_ref=1000.0
    )
    
    p = jnp.array([[2000.0]])
    q = jnp.array([[1500.0]])
    
    p_s_out, v_out, q_out = mcc.precondition(p_stack=p, q_stack=q)
    
    # Expected calculations
    # q_p = q / p
    # p_s = (0.5 * (q_p / M)**2 + 1.0) * p
    q_p = 1500.0 / 2000.0
    p_s_expected = (0.5 * (q_p / 1.0)**2 + 0.5) * 2000.0
    
    v_csl = 2.0 * (1000.0 / p_s_expected)**0.2
    v_elas = (p_s_expected / 2000.0)**0.05
    v_expected = v_csl * v_elas

    assert jnp.allclose(p_s_out, p_s_expected)
    assert jnp.allclose(v_out, v_expected)
    assert jnp.allclose(q_out, q)

def test_mcc_precondition_case3():
    # Case 3: Input p and v -> output p_s and q
    mcc = hdx.ModifiedCamClay(
        nu=0.3,
        M=1.0,
        lam=0.2,
        kap=0.05,
        gamma=2.0,
        p_ref=1000.0
    )
    
    p = jnp.array([[2000.0]])
    v = jnp.array([[1.5]]) # Arbitrary value
    
    p_s_out, v_out, q_out = mcc.precondition(p_stack=p, specific_volume_stack=v)
    
    # Expected calculations
    # p_s = [ (gamma * p_ref^lam) / (v * p^kap) ] ^ (1 / (lam - kap))
    term1 = 2.0 * 1000.0**0.2
    term2 = 1.5 * 2000.0**0.05
    p_s_expected = (term1 / term2)**(1 / (0.2 - 0.05))
    
    q_p = 1.0 * jnp.sqrt(2 * (p_s_expected / 2000.0) - 1)
    q_expected = q_p * 2000.0
    
    assert jnp.allclose(p_s_out, p_s_expected)
    assert jnp.allclose(q_out, q_expected)
    assert jnp.allclose(v_out, v)

def test_mcc_consistency():
    # Round trip test
    mcc = hdx.ModifiedCamClay(
        nu=0.3,
        M=1.0,
        lam=0.2,
        kap=0.05,
        gamma=2.0,
        p_ref=1000.0
    )
    
    p_in = jnp.array([[2000.0]])
    p_s_in = jnp.array([[4000.0]])
    
    # Case 1: p, p_s -> v, q
    p_s_1, v_1, q_1 = mcc.precondition(p_stack=p_in, p_s_stack=p_s_in)

    # # Case 2: p, q -> p_s, v
    p_s_2, v_2, q_2 = mcc.precondition(p_stack=p_in, q_stack=q_1)


    assert jnp.allclose(p_s_2, p_s_in)
    assert jnp.allclose(v_2, v_1)
    
    # Case 3: p, v -> p_s, q
    p_s_3, v_3, q_3 = mcc.precondition(p_stack=p_in, specific_volume_stack=v_1)
    
    assert jnp.allclose(p_s_3, p_s_in)
    assert jnp.allclose(q_3, q_1)





# Helper to create dummy material points
def create_dummy_mp(num_points=1):
    return MaterialPointState.create(
        position_stack=jnp.zeros((num_points, 3)),
        velocity_stack=jnp.zeros((num_points, 3)),
        mass_stack=jnp.ones((num_points, 1)),
        volume_stack=jnp.ones((num_points, 1)),
        volume0_stack=jnp.ones((num_points, 1)),
        L_stack=jnp.zeros((num_points, 3, 3)),
        stress_stack=jnp.zeros((num_points, 3, 3)),
        F_stack=jnp.eye(3)[None, ...].repeat(num_points, axis=0)
    )

def test_update_stress_elastic():
    """Perform small compression to test elastic only update."""
    nu = 0.3
    M = 1.0
    lam = 0.1
    kap = 0.02
    p_ref = 1000.0
    gamma = 2.0
    
    model = hdx.ModifiedCamClay(
        nu=nu, M=M, lam=lam, kap=kap, p_ref=p_ref, gamma=gamma,
        K_min=None, K_max=None
    )

    dt = 0.01
    
    # Small compression -> Elastic
    L = jnp.zeros((3, 3))
    L = L.at[0, 0].set(-0.001)
    L = L.at[1, 1].set(-0.001)
    L = L.at[2, 2].set(-0.001)
    
    eps_e_prev = jnp.zeros((3, 3))
    
    p_init = 2000.0
    stress_prev = -p_init * jnp.eye(3)
    stress_ref = stress_prev
    
    # Preconsolidation pressure far away -> Elastic
    p_s_prev = 5000.0 
    
    specific_volume = 2.0
    
    stress_next, eps_e_next, p_s_next = model._update_stress(
        L, eps_e_prev, stress_prev, p_s_prev, stress_ref, specific_volume, dt
    )
    
    # Check p_s didn't change (elastic)
    print(p_s_next,p_s_prev)
    assert jnp.isclose(p_s_next, p_s_prev)
    
    # Check stress changed
    assert not jnp.allclose(stress_next, stress_prev)
    
    # Check pressure increased
    p_next = -jnp.trace(stress_next) / 3.0

    assert p_next > p_init


def test_update_stress_plastic():
    """Test _update_stress in the plastic regime."""
    nu = 0.3
    M = 1.0
    lam = 0.1
    kap = 0.02
    p_ref = 1000.0
    gamma = 2.0
    
    model = hdx.ModifiedCamClay(
        nu=nu, M=M, lam=lam, kap=kap, p_ref=p_ref, gamma=gamma,
        K_min=1e3, K_max=1e9
    )

    dt = 0.01
    
    # Large compression -> Plastic
    L = jnp.zeros((3, 3))
    L = L.at[0, 0].set(-1.0)
    L = L.at[1, 1].set(-1.0)
    L = L.at[2, 2].set(-1.0)
    
    eps_e_prev = jnp.zeros((3, 3))
    
    p_init = 10000.0
    stress_prev = -p_init * jnp.eye(3)
    stress_ref = stress_prev
    
    # Normally consolidated
    # Yield function: (q/M)^2 + (p-p_s)^2 - p_s^2 = 0
    # q=0. (p-p_s)^2 - p_s^2 = 0 => p = 2*p_s.
    # So if p_init = 100, p_s_prev = 50.
    p_s_prev = 5000.0
    
    specific_volume = 2.0
    
    stress_next, eps_e_next, p_s_next = model._update_stress(
        L, eps_e_prev, stress_prev, p_s_prev, stress_ref, specific_volume, dt
    )
    
    # Check p_s changed (hardening)
    assert p_s_next > p_s_prev

    print("Previous ps:",p_s_next)
    
    # Check stress changed
    assert not jnp.allclose(stress_next, stress_prev)

test_update_stress_plastic()
