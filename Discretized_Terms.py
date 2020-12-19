# Functions for calculating terms based on spatial derivatives

# Divergence of predicted velocities
def get_divergence_velocities(u, v, Dx, Dy) :
    ''' Returns divergence for each cell times cell volume  '''

    return  (   (u[2:, 1:-1] - u[:-2, 1:-1]) * Dy +
                (v[1:-1, 2:] - v[1:-1, :-2]) * Dx   ) / 2

# Convection term in x-Momentum
def conv_x(u, v, Dx, Dy) :
    ''' Returns the value of convection term in x-momentum equation,
        for each cell in the domain in matrix form.
        u and v need to include ghost cells. '''

    return 0.5  * ( Dy * (u[1:-1, 1:-1]**2 + u[2:, 1:-1]**2)
                +   Dx * (u[1:-1, 1:-1] * v[1:-1, 1:-1] + u[1:-1, 2:] * v[1:-1, 2:])
                -   Dy * (u[1:-1, 1:-1]**2 + u[:-2,1:-1]**2)
                -   Dx * (u[1:-1, 1:-1] * v[1:-1, 1:-1] + u[1:-1,:-2] * v[1:-1,:-2])    )

# Convection term in y-Momentum
def conv_y(u, v, Dx, Dy) :
    ''' Returns the value of convection term in y-momentum equation,
        for each cell in the domain in matrix form.
        u and v need to include ghost cells. '''

    return 0.5  * ( Dy * (u[1:-1, 1:-1] * v[1:-1, 1:-1] + u[2:, 1:-1] * v[2:, 1:-1])
                +   Dx * (v[1:-1, 1:-1]**2 + v[1:-1, 2:]**2)
                -   Dy * (u[1:-1, 1:-1] * v[1:-1, 1:-1] + u[:-2,1:-1] * v[:-2,1:-1])
                -   Dx * (v[1:-1, 1:-1]**2 + v[1:-1,:-2]**2)   )

# Diffusion term in x-Momentum
def diff_x(u, nu, Dx, Dy) :
    ''' Returns the value of diffusion term in x-momentum equation,
        for each cell in the domain in matrix form.
        u need to include ghost cells. '''

    return nu * (
        (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) * Dy / Dx +
        (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) * Dx / Dy
    )

# Diffusion term in y-Momentum
def diff_y(v, nu, Dx, Dy) :
    ''' Returns the value of convection term in y-momentum equation,
        for each cell in the domain in matrix form.
        u need to include ghost cells. '''

    return nu * (
        (v[2:, 1:-1] - 2 * v[1:-1, 1:-1] + v[:-2, 1:-1]) * Dy / Dx +
        (v[1:-1, 2:] - 2 * v[1:-1, 1:-1] + v[1:-1, :-2]) * Dx / Dy
    )

# Pressure gradient along x
def x_grad_p(p, Dx) :
    ''' Returns pressure gradient along x using central difference,
        for each cell in the domain in matrix form.
        p needs to include ghost cells  '''

    return (p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * Dx)

# Pressure gradient along y
def y_grad_p(p, Dy) :
    ''' Returns pressure gradient along y using central difference,
        for each cell in the domain in matrix form.
        p needs to include ghost cells  '''

    return (p[1:-1, 2:] - p[1:-1, :-2]) / (2 * Dy)