from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

# A Vector3 is a NumPy array containing float64 values.
# We use it for physical 3D quantities such as:
# position, velocity, force, torque, etc.
# Example:
# position = [x, y, z]
# This type hint does NOT enforce that the array has exactly 3 elements.
Vector3 = NDArray[np.float64]

# A quaternion is stored as four float64 values:
#     q = [w, x, y, z]
# Mathematically:
#     q = w + xi + yj + zk
# For orientation we use UNIT quaternions:
#     ||q|| = 1
# Although stored in a NumPy array, a quaternion is not simply a 4D vector.
# Quaternion multiplication follows quaternion algebra.
Quaternion = NDArray[np.float64]

def zero_vector() -> Vector3:
    """
    Create a new zero-valued 3D vector.
    Returns:
        [0.0, 0.0, 0.0]
    This function is used as a dataclass default_factory so that every
    VehicleState / VehicleDynamics instance receives its own NumPy array.
    """
    return np.zeros(3, dtype=np.float64)


def identity_quaternion() -> Quaternion:
    """
    Create the identity quaternion.
        q = [1, 0, 0, 0]
    This represents zero rotation: the body frame is aligned with
    the world frame.
    """
    return np.array(
        [1.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )

@dataclass
class VehicleState:
    """
    The persistent physical state of a rigid body.
    We represent the state as:
        X = (p, v, q, omega)
    where:
        p = position
        v = linear velocity
        q = orientation
        omega = angular velocity


    COORDINATE CONVENTIONS:
    Position and velocity are expressed in the WORLD frame.
        position = [x, y, z]
        velocity = [vx, vy, vz]
    Angular velocity is expressed in the BODY frame.
        angular_velocity = [wx, wy, wz]
    Orientation is represented by a unit quaternion:
        orientation = [w, x, y, z]
    Rotation from BODY coordinates to WORLD coordinates EXPLAINED:

        Rotation quaternion:
        q = cos(theta/2) + sin(theta/2) * (ux*i + uy*j + uz*k) where [ux, uy, uz] is the unit rotation axis.

        To rotate a vector p:
        p_rotated = q * p * q^-1 where p is represented as the pure quaternion:
    
        p = 0 + px*i + py*j + pz*k 
    
    
    DEFAULT STATE:
    position:
        [0, 0, 0]
        Vehicle begins at the world origin.
    velocity:
        [0, 0, 0]
        Vehicle begins stationary.
    orientation:
        [1, 0, 0, 0]
        Body frame initially aligned with world frame.
    angular_velocity:
        [0, 0, 0]
        Vehicle initially has no rotational motion.
    """
    position: Vector3 = field(default_factory=zero_vector)
    velocity: Vector3 = field(default_factory=zero_vector)
    orientation: Quaternion = field(default_factory=identity_quaternion)
    angular_velocity: Vector3 = field(default_factory=zero_vector)


# ---DYNAMICS---
@dataclass
class VehicleDynamics:
    """
    Quantities calculated from the forces acting on the vehicle during
    the current simulation step.
    These are NOT persistent state variables.
    Net force:
        F = [Fx, Fy, Fz]
    Net torque:
        tau = [tau_x, tau_y, tau_z]
    Linear acceleration:
        a = dv/dt
    Angular acceleration:
        alpha = d(omega)/dt
    
    The dynamics engine will calculate:
        forces + torques --> rigid-body equations --> acceleration + angular acceleration --> numerical integration --> new VehicleState
    """
    force: Vector3 = field(default_factory=zero_vector)
    torque: Vector3 = field(default_factory=zero_vector)
    acceleration: Vector3 = field(default_factory=zero_vector)
    angular_acceleration: Vector3 = field(default_factory=zero_vector)