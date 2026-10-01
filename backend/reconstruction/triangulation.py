import cv2
import numpy as np


def triangulate_points(
    K,
    R,
    t,
    points1,
    points2
):

    P1 = K @ np.hstack(
        (
            np.eye(3),
            np.zeros((3, 1))
        )
    )

    P2 = K @ np.hstack(
        (
            R,
            t
        )
    )

    points4d = cv2.triangulatePoints(
        P1,
        P2,
        points1.T,
        points2.T
    )

    points4d /= points4d[3]

    points3d = points4d[:3].T

    valid = np.isfinite(
        points3d
    ).all(axis=1)

    return points3d[valid]