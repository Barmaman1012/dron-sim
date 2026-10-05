import numpy as np

from physics.rigid_body import RigidBody


body = RigidBody(
    mass=2.0,
    inertia_tensor=np.diag([0.02, 0.02, 0.04]),
)

print(body)