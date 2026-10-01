import cv2
import numpy as np


def create_camera_matrix(
    width,
    height
):

    focal_length = 0.9 * max(
        width,
        height
    )

    cx = width / 2.0
    cy = height / 2.0

    K = np.array(
        [
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ],
        dtype=np.float64
    )

    return K


def recover_camera_pose(
    points1,
    points2,
    K
):

    if len(points1) < 8:
        raise ValueError(
            "At least 8 matches are required."
        )

    E, mask = cv2.findEssentialMat(
        points1,
        points2,
        K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0
    )

    if E is None:
        raise ValueError(
            "Essential matrix could not be estimated."
        )

    inliers, R, t, pose_mask = cv2.recoverPose(
        E,
        points1,
        points2,
        K
    )

    return {
        "E": E,
        "R": R,
        "t": t,
        "mask": pose_mask,
        "inliers": int(inliers)
    }