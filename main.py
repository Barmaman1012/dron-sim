import numpy as np

from physics.rigid_body import RigidBody
from physics.dynamics import compute_dynamics


body = RigidBody(
    mass=2.0,
    inertia_tensor=np.diag([0.02, 0.02, 0.04]),
)

body.dynamics.force = np.array(
    [10.0, 0.0, 0.0],
    dtype=np.float64,
)

body.dynamics.torque = np.array(
    [0.0, 0.0, 0.4],
    dtype=np.float64,
)

compute_dynamics(body)

print("Acceleration:", body.dynamics.acceleration)
print("Angular acceleration:", body.dynamics.angular_acceleration)