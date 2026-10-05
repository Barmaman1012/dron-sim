import numpy as np

from physics.rigid_body import RigidBody


def compute_dynamics(body: RigidBody) -> None:
    force = body.dynamics.force
    torque = body.dynamics.torque

    mass = body.mass
    inertia = body.inertia_tensor
    omega = body.state.angular_velocity

    acceleration = force / mass

    angular_momentum = inertia @ omega

    gyroscopic_term = np.cross(
        omega,
        angular_momentum,
    )

    angular_acceleration = np.linalg.solve(
        inertia,
        torque - gyroscopic_term,
    )

    body.dynamics.acceleration = acceleration
    body.dynamics.angular_acceleration = angular_acceleration