import numpy as np

from physics.rigid_body import RigidBody


def compute_dynamics(body: RigidBody) -> None:
    """
    Calculates the linear and angular acceleration of a rigid body
    from the forces and torques currently acting on it.

    LINEAR DYNAMICS:
        Newton's second law:
            F = m * a
        Therefore:
            a = F / m

    ROTATIONAL DYNAMICS:
        Newton-Euler equation:
            tau = I * alpha + omega x (I * omega)

        Therefore:
            I * alpha = tau - omega x (I * omega)

        We solve this equation for alpha.

    where:
        F     = net force
        m     = mass
        a     = linear acceleration

        tau   = net torque
        I     = inertia tensor
        omega = angular velocity
        alpha = angular acceleration

    The function updates:
        body.dynamics.acceleration
        body.dynamics.angular_acceleration

    It does NOT update position, velocity, orientation, or angular velocity.
    Those will be updated later by the numerical integrator.
    """

    # Get the current forces and torques acting on the body.
    force = body.dynamics.force
    torque = body.dynamics.torque

    # Get the physical properties and current rotational state of the body.
    mass = body.mass
    inertia = body.inertia_tensor
    omega = body.state.angular_velocity

    # Newton's second law:
    # F = m*a  -->  a = F/m
    acceleration = force / mass

    # Angular momentum:
    # L = I * omega
    #
    # '@' performs matrix multiplication:
    # inertia is 3x3 and omega is a 3D vector.
    angular_momentum = inertia @ omega

    # Rotating rigid bodies contain an additional term:
    # omega x (I * omega) = omega x angular_momentum
    #
    # This accounts for the interaction between the body's current rotation
    # and its mass distribution.
    gyroscopic_term = np.cross(
        omega,
        angular_momentum,
    )

    # Newton-Euler rotational equation:
    # I * alpha = torque - gyroscopic_term
    #
    # np.linalg.solve(A, b) solves:
    # A * x = b
    #
    # Here:
    # A = inertia
    # x = angular_acceleration
    # b = torque - gyroscopic_term
    angular_acceleration = np.linalg.solve(
        inertia,
        torque - gyroscopic_term,
    )

    # Store the calculated accelerations in the body's current dynamics.
    body.dynamics.acceleration = acceleration
    body.dynamics.angular_acceleration = angular_acceleration