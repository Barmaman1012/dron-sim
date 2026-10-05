from physics.rigid_body import RigidBody


def integrate(body: RigidBody, dt: float) -> None:
    """
    Advances the rigid body's linear state by one simulation timestep.

    dt:
        Size of one simulation timestep in seconds.

    LINEAR INTEGRATION:
        Acceleration changes velocity:
            dv/dt = a

        Over a small timestep dt:
            v_new = v_old + a * dt

        Velocity changes position:
            dp/dt = v

        Over a small timestep dt:
            p_new = p_old + v_new * dt

    We update velocity before position, making this a semi-implicit
    Euler integrator.

    Smaller dt values generally produce a more accurate approximation
    of the continuous physical system, but require more calculations.

    This function currently integrates only linear motion.
    Rotation will be added separately.
    """

    # Get the linear acceleration calculated by the dynamics engine.
    acceleration = body.dynamics.acceleration

    # Update velocity using:
    # v_new = v_old + a * dt
    body.state.velocity += acceleration * dt

    # Update position using the newly calculated velocity:
    # p_new = p_old + v_new * dt
    body.state.position += body.state.velocity * dt