import numpy as np

from physics.rigid_body import RigidBody
from physics.quaternion import quaternion_multiply, normalize_quaternion


def integrate(body: RigidBody, dt: float) -> None:
    """
    Advances the rigid body's state by one simulation timestep.

    dt:
        Size of one simulation timestep in seconds.

    The complete rigid-body state is:
        X = (p, v, q, omega)

    The integrator updates:
        acceleration         --> velocity
        velocity             --> position
        angular acceleration --> angular velocity
        angular velocity     --> orientation

    LINEAR MOTION:
        dv/dt = a
        dp/dt = v

    ROTATIONAL MOTION:
        d(omega)/dt = alpha
        dq/dt = 0.5 * q * omega_q

    We use semi-implicit Euler integration:
        v_new = v_old + a * dt
        p_new = p_old + v_new * dt

        omega_new = omega_old + alpha * dt
        q_new = q_old + q_dot * dt

    Smaller dt values generally provide a better approximation of
    the continuous physical system but require more calculations.
    """

    # --- LINEAR INTEGRATION ---

    acceleration = body.dynamics.acceleration

    # Acceleration changes velocity:
    # v_new = v_old + a * dt
    body.state.velocity += acceleration * dt

    # Velocity changes position:
    # p_new = p_old + v_new * dt
    body.state.position += body.state.velocity * dt


    # --- ROTATIONAL INTEGRATION ---

    angular_acceleration = body.dynamics.angular_acceleration

    # Angular acceleration changes angular velocity:
    # omega_new = omega_old + alpha * dt
    body.state.angular_velocity += angular_acceleration * dt

    omega = body.state.angular_velocity
    orientation = body.state.orientation

    # Convert angular velocity:
    # omega = [wx, wy, wz]
    #
    # into a pure quaternion:
    # omega_q = [0, wx, wy, wz]
    #
    # The scalar component is zero because omega represents a vector,
    # not an orientation.
    omega_quaternion = np.array(
        [0.0, omega[0], omega[1], omega[2]],
        dtype=np.float64,
    )

    # For a body-to-world orientation quaternion q and angular velocity
    # expressed in the body frame:
    #
    # q_dot = 0.5 * q * omega_q
    #
    # Quaternion multiplication is used here, not normal vector
    # multiplication.
    orientation_derivative = (
        0.5
        * quaternion_multiply(
            orientation,
            omega_quaternion,
        )
    )

    # Integrate the orientation:
    # q_new = q_old + q_dot * dt
    orientation += orientation_derivative * dt

    # A rotation quaternion must remain a unit quaternion:
    # ||q|| = 1
    #
    # Numerical integration introduces small errors, so we normalize
    # the quaternion after every timestep.
    body.state.orientation = normalize_quaternion(orientation)