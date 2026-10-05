import numpy as np

from physics.state import Quaternion


def quaternion_multiply(
    q1: Quaternion,
    q2: Quaternion,
) -> Quaternion:
    """
    Multiplies two quaternions.

    A quaternion is represented as:
        q = [w, x, y, z]

    Mathematically:
        q = w + xi + yj + zk

    Quaternion multiplication is NOT normal vector multiplication.
    It follows the quaternion algebra:
        i^2 = j^2 = k^2 = ijk = -1

    Quaternion multiplication is also order-dependent:
        q1 * q2 != q2 * q1

    This operation is used to compose rotations and to calculate
    changes in orientation.
    """

    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def normalize_quaternion(q: Quaternion) -> Quaternion:
    """
    Normalizes a quaternion so that its magnitude is 1.

        q_normalized = q / ||q||

    Orientation quaternions must be unit quaternions:
        ||q|| = 1

    Numerical calculations can slowly move the quaternion away from
    unit length, so normalization keeps it a valid rotation quaternion.

    A zero quaternion cannot be normalized because its magnitude is zero.
    """

    norm = np.linalg.norm(q)

    if norm == 0.0:
        raise ValueError("Cannot normalize a zero quaternion.")

    return q / norm