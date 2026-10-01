import cv2
import numpy as np


def match_features(
    descriptors1,
    descriptors2,
    ratio=0.75
):

    if descriptors1 is None:
        return []

    if descriptors2 is None:
        return []

    if len(descriptors1) == 0:
        return []

    if len(descriptors2) == 0:
        return []

    matcher = cv2.BFMatcher(
        cv2.NORM_L2
    )

    matches = matcher.knnMatch(
        descriptors1,
        descriptors2,
        k=2
    )

    good_matches = []

    for pair in matches:

        if len(pair) != 2:
            continue

        m, n = pair

        if m.distance < ratio * n.distance:
            good_matches.append(m)

    return good_matches


def match_points(
    points1,
    descriptors1,
    points2,
    descriptors2
):

    good_matches = match_features(
        descriptors1,
        descriptors2
    )

    if len(good_matches) == 0:

        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            []
        )

    matched1 = np.array(
        [
            points1[m.queryIdx]
            for m in good_matches
        ],
        dtype=np.float32
    )

    matched2 = np.array(
        [
            points2[m.trainIdx]
            for m in good_matches
        ],
        dtype=np.float32
    )

    return matched1, matched2, good_matches