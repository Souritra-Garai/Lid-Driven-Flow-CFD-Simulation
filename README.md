# Lid-Driven-Flow-CFD-Simulation

Python code for solving Napier Stokes equation using finite volume method in the scenario of Lid driven flow. The Quasi Rhie Chow algorithm has been implemented to arrive at the solution.

Code requires the following python libraries to be installed -
 - Numpy
 - Matplotlib
 
The main code to obtain the solution is written in the jupyter notebook - "Project 4 18110166.ipynb"

The scripts -
 - Boundary_Conditions.py has methods for defining the boundary velocities and pressures using ghost cells.
 - Discretized_Terms.py has methods to estimate the convection term, diffusion term and pressure gradient terms in the finite volume method approach in a collocated grid.
 - Pressure_Poisson_Equation_Solver.py as the name suggests contains methods for solving the pressure Poisson equation required in correction steps of Quasi Rhie Chow algorithm.
 - Quasi_Rhie_Chow.py contains method to perform a complete iteration of Quasi Rhie Chow algorithm - the predictor and the corrector steps.
 - TDMA_Solver.py contains class definition for a solver to solve tridiagonal matrix equations.

The scripts to visualize the solution are -
 - Streamplots.py plots streamlines at steady state of for the the lid driven flow.
 - Flow_Animation.py creates a video animation showing the flow of fluid particles till steady state is reached.
 
Here is the streamplot at steady state for a 40 by 40 grid -
![Streamplot](/40_cross_40.png)


