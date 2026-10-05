from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from physics.state import VehicleState, VehicleDynamics

# A Matrix3 is a 3x3 matrix of float64 values.
# We use it here to represent the inertia tensor.
# This type hint does NOT enforce the 3x3 shape.
Matrix3 = NDArray[np.float64]


@dataclass
class RigidBody:
    """
    Represents a physical rigid body in the simulation.

    mass:
        Total mass of the body in kilograms.

    inertia_tensor:
        Describes how the body's mass is distributed relative to its
        body axes and therefore its resistance to angular acceleration.

        I = [[Ixx, Ixy, Ixz],
             [Iyx, Iyy, Iyz],
             [Izx, Izy, Izz]]

        For a symmetric body aligned with its principal axes:
        I = [[Ixx, 0,   0  ],
             [0,   Iyy, 0  ],
             [0,   0,   Izz]]

    state:
        Persistent motion state: position, velocity, orientation
        and angular velocity.

    dynamics:
        Current force, torque, linear acceleration and angular acceleration.
    """
    mass: float
    inertia_tensor: Matrix3

    state: VehicleState = field(default_factory=VehicleState)
    dynamics: VehicleDynamics = field(default_factory=VehicleDynamics)