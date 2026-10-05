import numpy as np

from physics.rigid_body import RigidBody
from physics.integrator import integrate


body = RigidBody(
    mass=2.0,
    inertia_tensor=np.diag([0.02, 0.02, 0.04]),
)

body.state.angular_velocity = np.array(
    [0.0, 0.0, 1.0],
    dtype=np.float64,
)

dt = 0.01

for step in range(100):
    integrate(body, dt)

print("Angular velocity:", body.state.angular_velocity)
print("Orientation:", body.state.orientation)
print("Quaternion norm:", np.linalg.norm(body.state.orientation))