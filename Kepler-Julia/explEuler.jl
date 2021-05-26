using DifferentialEquations

# G = 6.67430e-11 #m^3/(kg*s^2)
# m1 = 5.9723e+24 # kg
# m2 = 1.9884e+30 # kg m2 = 333*m1

G = 1.0
m1 = 0.9
m2 = 0.1
M = m1 + m2
mu = m1 * m2 / M

function kepler(t, u, du)
    du[1] = u[3]
    du[2] = u[4]
  
    norm_r = sprt(u[3]^2 + u[4]^2)^3
  
    du[3] = -u[1] / norm_r
    du[4] = -u[2] / norm_r
end

u0 = [1.0;0.0]
tspan = (0.0, 100.0)
problem = ODEProblem(kepler, u0, tspan)
solution = solve(problem, Euler(), reltol=1e-6, saveat=0.1, save_everystep=true)