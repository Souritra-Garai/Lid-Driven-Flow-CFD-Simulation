import numpy as np
from TDMA_Solver import solver
from Discretized_Terms import get_divergence_velocities
from Boundary_Conditions import update_ghost_cell_pressure

# x implicit solvers
pressure_poisson_solver_x = None
pressure_poisson_solver_x_south_boundary = None
pressure_poisson_solver_x_north_boundary = None

# y implicit solvers
pressure_poisson_solver_y = None
pressure_poisson_solver_y_east_boundary = None
pressure_poisson_solver_y_west_boundary = None

# Function for Setting up Solvers

def setup_x_implicit_solvers(M, Dx, Dy) :

    global pressure_poisson_solver_x, pressure_poisson_solver_x_south_boundary, pressure_poisson_solver_x_north_boundary

    # Arrays for tridiagonal matrix
    E = np.zeros(M)
    F = np.zeros(M)
    G = np.zeros(M)

    # x Implicit TDMA solver for pressure poisson eqn over general domain
    E += Dy / Dx
    F -= 2 * Dy / Dx  + 2 * Dx / Dy
    G += Dx / Dy

    F[0]    += Dy / Dx
    F[-1]   += Dy / Dx

    pressure_poisson_solver_x = solver.generate_solver(E, F, G)

    # x Implicit TDMA solver for pressure poisson eqn at South Boundary
    F += Dx / Dy
    pressure_poisson_solver_x_south_boundary = solver.generate_solver(E, F, G)

    # x Implicit TDMA solver for pressure poisson eqn at North Boundary
    F[0] = 1
    E[0] = G[0] = 0
    pressure_poisson_solver_x_north_boundary = solver.generate_solver(E, F, G)

    pass

def setup_y_implicit_solvers(M, Dx, Dy) :

    global pressure_poisson_solver_y, pressure_poisson_solver_y_east_boundary, pressure_poisson_solver_y_west_boundary

    # Arrays for tridiagonal matrix
    E = np.zeros(M)
    F = np.zeros(M)
    G = np.zeros(M)

    # y Implicit TDMA solver for pressure poisson eqn over general domain
    E[:] = Dx / Dy
    F[:] = - (2 * Dx / Dy  + 2 * Dy / Dx)
    G[:] = Dx / Dy

    F[0]    += Dx / Dy
    F[-1]   += Dx / Dy

    pressure_poisson_solver_y = solver.generate_solver(E, F, G)

    # y Implicit TDMA solver for pressure poisson eqn at east boundary
    F += Dy / Dx
    pressure_poisson_solver_y_east_boundary = solver.generate_solver(E, F, G)

    # y Implicit TDMA solver for pressure poisson eqn at west boundary
    F[-1] = 1
    G[-1] = E[-1] = 0
    pressure_poisson_solver_y_west_boundary = solver.generate_solver(E, F, G)

    pass

def setup_solvers(M, Dx, Dy) :

    setup_x_implicit_solvers(M, Dx, Dy)
    setup_y_implicit_solvers(M, Dx, Dy)
    pass

# Functions for obtaining RHS of Matrix Equations

# RHS of Matrix Equation for y Implicit Equations
# General
def get_b_pressure_poisson_eqn_y(rho_div_predicted_vel_by_Dt, p, Dx, Dy) :

    return  np.transpose(rho_div_predicted_vel_by_Dt[1:-1, :] - (p[3:-1, 1:-1] + p[1:-3, 1:-1]) * Dy / Dx)

# East boundary
def get_b_pressure_poisson_eqn_y_east_boundary(rho_div_predicted_vel_by_Dt, p, rho, g, Dx, Dy) :

    b =  np.copy(rho_div_predicted_vel_by_Dt[-1,:])
    
    b += - p[-3,1:-1] * Dy / Dx + rho * g * Dx    # Cells adjacent East Boundary
    
    return b

# West boundary
def get_b_pressure_poisson_eqn_y_west_boundary(rho_div_predicted_vel_by_Dt, p, rho, g, Dx, Dy) :

    b =  np.copy(rho_div_predicted_vel_by_Dt[0, :])
    
    b += - p[2, 1:-1] * Dy / Dx - rho * g * Dx    # Cells adjacent West Boundary
    b[-1] = 0
    
    return b

# RHS of Matrix Equation for x Implicit Equations
# General
def get_b_pressure_poisson_eqn_x(rho_div_predicted_vel_by_2Dt, p, Dx, Dy) :

    return  rho_div_predicted_vel_by_2Dt[:, 1:-1] - (p[1:-1, 3:-1] + p[1:-1, 1:-3]) * Dx / Dy

# North boundary
def get_b_pressure_poisson_eqn_x_north_boundary(rho_div_predicted_vel_by_Dt, p, rho, g, Dx, Dy) :

    b =  np.copy(rho_div_predicted_vel_by_Dt[:,-1])
    
    b += - p[1:-1,-3] * Dx / Dy + rho * g * Dx    # Cells adjacent North Boundary
    b[0] = 0    # Fixing pressure at (0,1) to be zero
    
    return b

# South boundary
def get_b_pressure_poisson_eqn_x_south_boundary(rho_div_predicted_vel_by_Dt, p, rho, g, Dx, Dy) :

    b =  np.copy(rho_div_predicted_vel_by_Dt[:, 0])

    b += - p[1:-1, 2] * Dx / Dy - rho * g * Dx    # Cells adjacent South Boundary
    
    return b

# Methods for solving pressure poisson equations

def solve_pressure_poisson_eqn_implicit_x(rho_div_predicted_vel_by_Dt, p, rho, g, Dx, Dy) :

    new_p = np.zeros_like(p)

    b = get_b_pressure_poisson_eqn_x(rho_div_predicted_vel_by_Dt, p, Dx, Dy)
    new_p[1:-1, 2:-2] = pressure_poisson_solver_x.solve(b)

    b = get_b_pressure_poisson_eqn_x_north_boundary(rho_div_predicted_vel_by_Dt, p, rho, g, Dx, Dy)
    new_p[1:-1,-2] = pressure_poisson_solver_x_north_boundary.solve(b)

    b = get_b_pressure_poisson_eqn_x_south_boundary(rho_div_predicted_vel_by_Dt, p, rho, g, Dx, Dy)
    new_p[1:-1, 1] = pressure_poisson_solver_x_south_boundary.solve(b)

    update_ghost_cell_pressure(new_p, rho, g, Dx, Dy)

    return new_p

def solve_pressure_poisson_eqn_implicit_y(rho_div_predicted_vel_by_Dt, p, rho, g, Dx, Dy) :

    new_p = np.zeros_like(p)

    b = get_b_pressure_poisson_eqn_y(rho_div_predicted_vel_by_Dt, p, Dx, Dy)
    new_p[2:-2, 1:-1] = np.transpose(pressure_poisson_solver_y.solve(b))

    b = get_b_pressure_poisson_eqn_y_east_boundary(rho_div_predicted_vel_by_Dt, p, rho, g, Dx, Dy)
    new_p[-2,1:-1] = pressure_poisson_solver_y_east_boundary.solve(b)

    b = get_b_pressure_poisson_eqn_y_west_boundary(rho_div_predicted_vel_by_Dt, p, rho, g, Dx, Dy)
    new_p[1, 1:-1] = pressure_poisson_solver_y_west_boundary.solve(b)

    update_ghost_cell_pressure(new_p, rho, g, Dx, Dy)

    return new_p

def solve_pressure_poisson_eqn(p, predicted_u, predicted_v, rho, g, Dt, Dx, Dy) :
    ''' Solves the equivalent of pressure poisson eqn
        in Finite Volume method '''

    rho_div_predicted_vel_by_Dt = rho * get_divergence_velocities(predicted_u, predicted_v, Dx, Dy) / Dt

    new_p = solve_pressure_poisson_eqn_implicit_x(rho_div_predicted_vel_by_Dt, p, rho, g, Dx, Dy)
    new_p = solve_pressure_poisson_eqn_implicit_y(rho_div_predicted_vel_by_Dt, new_p, rho, g, Dx, Dy)

    return new_p
