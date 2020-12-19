# Functions for updating the ghost cells
# 1st Order Scheme are used for approximation
# Uniform grid assumed

# Updating ghost cells adjoining Dirichlet BC
def calc_ghost_cell_value_Dirichlet_BC(value_at_boundary, value_close_to_boundary) :
    ''' Returns the values at the cell centre of the ghost cells
        given the value at the boundary and the value at the cell 
        centre of the cell adjoining boundary within the domain,
        using linear interpolation  '''

    return 2 * value_at_boundary - value_close_to_boundary

# Updating ghost cells adjoining Neumann BC
def calc_ghost_cell_value_Neumann_BC(gradient_at_boundary, value_close_to_boundary, Delta) :
    ''' Returns the values at the cell centre of the ghost cells
        given the gradient at the boundary and the value at the cell
        centre of the cell adjoining the boundary within the domain,
        using 1st order scheme  '''

    return value_close_to_boundary + gradient_at_boundary * Delta

# Updating velocities in ghost cells
def calc_ghost_cell_u_vel(domain) :
    ''' Return the x - velocities, u at the ghost cells in the order
        East, North, West, South for the given domain,
        specific to the current problem '''

    return (
        calc_ghost_cell_value_Dirichlet_BC(0, domain[-1, :]),   # East
        calc_ghost_cell_value_Dirichlet_BC(1, domain[:, -1]),   # North
        calc_ghost_cell_value_Dirichlet_BC(0, domain[0,  :]),   # West
        calc_ghost_cell_value_Dirichlet_BC(0, domain[:,  0])    # South
    )

def calc_ghost_cell_v_vel(domain) :
    ''' Return the x - velocities, u at the ghost cells in the order
        East, North, West, South for the given domain,
        specific to the current problem '''

    return (
        calc_ghost_cell_value_Dirichlet_BC(0, domain[-1, :]),   # East
        calc_ghost_cell_value_Dirichlet_BC(0, domain[:, -1]),   # North
        calc_ghost_cell_value_Dirichlet_BC(0, domain[0,  :]),   # West
        calc_ghost_cell_value_Dirichlet_BC(0, domain[:,  0])    # South
    )

# Updating the Pressure in ghost cells
def calc_ghost_cell_pressure(domain, rho, g, Dx, Dy) :
    ''' Return the x - velocities, u at the ghost cells in the order
        East, North, West, South for the given domain,
        specific to the current problem '''

    return (
        calc_ghost_cell_value_Neumann_BC(0, domain[-1, :], Dx), # East
        calc_ghost_cell_value_Neumann_BC(- rho * g, domain[:, -1], Dy), # North
        calc_ghost_cell_value_Neumann_BC(0, domain[0,  :],-Dx), # West
        calc_ghost_cell_value_Neumann_BC(- rho * g, domain[:,  0],-Dy)  # South
    )

# Update Ghost cells

# Velocity
def update_ghost_cell_velocities(u, v) :
    ''' Given u, v (including the rows and columns for ghost cells)
        the velocity values in the ghost cells are updated according to
        the boundary conditions of the current problem  '''

    u[-1, 1:-1], u[1:-1, -1], u[0, 1:-1], u[1:-1, 0] = calc_ghost_cell_u_vel(u[1:-1, 1:-1])
    v[-1, 1:-1], v[1:-1, -1], v[0, 1:-1], v[1:-1, 0] = calc_ghost_cell_v_vel(v[1:-1, 1:-1])
    pass

# Pressure
def update_ghost_cell_pressure(p, rho, g, Dx, Dy) :
    ''' Given u, v (including the rows and columns for ghost cells)
        the velocity values in the ghost cells are updated according to
        the boundary conditions of the current problem  '''
    
    p[-1, 1:-1], p[1:-1, -1], p[0, 1:-1], p[1:-1, 0] = calc_ghost_cell_pressure(p[1:-1, 1:-1], rho, g, Dx, Dy)
    pass