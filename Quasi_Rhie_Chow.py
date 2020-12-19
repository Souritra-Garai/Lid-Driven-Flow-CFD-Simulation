import numpy as np
from Boundary_Conditions import update_ghost_cell_velocities
from Pressure_Poisson_Equation_Solver import solve_pressure_poisson_eqn
from Discretized_Terms import conv_x, conv_y, diff_x, diff_y, x_grad_p, y_grad_p

# Convergence Criterion
def converged(u_n, v_n, p_n, u_np1, v_np1, p_np1, tol=1e-6) :

    return (    np.isclose(u_n, u_np1, atol=tol).all() and
                np.isclose(v_n, v_np1, atol=tol).all() and
                np.isclose(p_n, p_np1, atol=tol).all()  )

# Method for performing inner iteration in Rhie Chow method
def predictor_corrector_iteration(
    u_n, v_n,
    p_n,
    conv_x_term, conv_y_term,
    diff_x_term, diff_y_term,
    rho, g,
    Dt,
    Dx, Dy
) :

    u_p = np.zeros_like(u_n)
    v_p = np.zeros_like(v_n)

    u_p[1:-1, 1:-1] = u_n[1:-1, 1:-1] + (- conv_x_term + diff_x_term) * Dt / (Dx * Dy)

    v_p[1:-1, 1:-1] = v_n[1:-1, 1:-1] + (- conv_y_term + diff_y_term) * Dt / (Dx * Dy) - g * Dt

    update_ghost_cell_velocities(u_p, v_p)

    p_np1 = solve_pressure_poisson_eqn(p_n, u_p, v_p, rho, g, Dt, Dx, Dy)

    u_np1 = np.zeros_like(u_n)
    v_np1 = np.zeros_like(v_n)

    u_np1[1:-1, 1:-1] = u_p[1:-1, 1:-1] - x_grad_p(p_np1, Dx) * Dt / rho
    v_np1[1:-1, 1:-1] = v_p[1:-1, 1:-1] - y_grad_p(p_np1, Dy) * Dt / rho

    update_ghost_cell_velocities(u_np1, v_np1)

    return u_np1, v_np1, p_np1

# Method of performing Quasi Rhie Chow iteration for a time step
def quasi_Rhie_Chow_iteration(u_n, v_n, p_n, rho, nu, g, Dt, Dx, Dy) :

    u_k = np.copy(u_n)
    v_k = np.copy(v_n)
    p_k = np.copy(p_n)

    u_kp1, v_kp1, p_kp1 = predictor_corrector_iteration(
        u_n, v_n,
        p_k,
        conv_x(u_k, v_k, Dx, Dy), conv_y(u_k, v_k, Dx, Dy),
        diff_x(u_k, nu, Dx, Dy), diff_y(v_k, nu, Dx, Dy),
        rho, g,
        Dt, Dx, Dy
    )   

    while not converged(u_k, v_k, p_k, u_kp1, v_kp1, p_kp1) :

        u_k = np.copy(u_kp1)
        v_k = np.copy(v_kp1)
        p_k = np.copy(p_kp1)

        u_kp1, v_kp1, p_kp1 = predictor_corrector_iteration(
            u_n, v_n,
            p_k,
            conv_x(u_k, v_k, Dx, Dy), conv_y(u_k, v_k, Dx, Dy),
            diff_x(u_k, nu, Dx, Dy), diff_y(v_k, nu, Dx, Dy),
            rho, g,
            Dt, Dx, Dy
        )

    return u_kp1, v_kp1, p_kp1
