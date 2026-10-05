import numpy as np

from physics.rigid_body import RigidBody
from physics.dynamics import compute_dynamics
from physics.integrator import integrate


body = RigidBody(
    mass=2.0,
    inertia_tensor=np.diag([0.02, 0.02, 0.04]),
)

body.dynamics.force = np.array(
    [10.0, 0.0, 0.0],
    dtype=np.float64,
)

dt = 0.01

for step in range(100):
    compute_dynamics(body)
    integrate(body, dt)

print("Position:", body.state.position)
print("Velocity:", body.state.velocity)
print("Acceleration:", body.dynamics.acceleration)