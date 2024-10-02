import cv2
import numpy as np
import matplotlib.pyplot as plt

from dataset.utils import Keypoints, _create_flip_indices, dense_to_sparse, get_modes, modes_to_array


class AnimalKeypoints(Keypoints):
    NAMES = [
        'left_eye',
        'right_eye',
        'nose',
        'neck',
        'root_of_tail',
        'left_shoulder',
        'left_elbow',
        'left_front_paw',
        'right_shoulder',
        'right_elbow',
        'right_front_paw',
        'left_hip',
        'left_knee',
        'left_back_paw',
        'right_hip',
        'right_knee',
        'right_back_paw',
    ]
    FLIP_MAP = {
        'left_eye': 'right_eye',
        'left_front_paw': 'right_front_paw',
        'left_shoulder': 'right_shoulder',
        'left_elbow': 'right_elbow',
        'left_hip': 'right_hip',
        'left_knee': 'right_knee',
        'left_back_paw': 'right_back_paw',
    }

AnimalKeypoints.FLIP_INDS = _create_flip_indices(AnimalKeypoints.NAMES, AnimalKeypoints.FLIP_MAP)


def kp_connections_animal(keypoints):
    kp_lines = [
        [keypoints.index('left_eye'), keypoints.index('right_eye')],
        [keypoints.index('left_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('nose')],
        [keypoints.index('nose'), keypoints.index('neck')],
        [keypoints.index('neck'), keypoints.index('left_shoulder')],
        [keypoints.index('left_shoulder'), keypoints.index('left_elbow')],
        [keypoints.index('left_elbow'), keypoints.index('left_front_paw')],
        [keypoints.index('neck'), keypoints.index('right_shoulder')],
        [keypoints.index('right_shoulder'), keypoints.index('right_elbow')],
        [keypoints.index('right_elbow'), keypoints.index('right_front_paw')],
        [keypoints.index('neck'), keypoints.index('root_of_tail')],
        [keypoints.index('root_of_tail'), keypoints.index('left_hip')],
        [keypoints.index('left_hip'), keypoints.index('left_knee')],
        [keypoints.index('left_knee'), keypoints.index('left_back_paw')],
        [keypoints.index('root_of_tail'), keypoints.index('right_hip')],
        [keypoints.index('right_hip'), keypoints.index('right_knee')],
        [keypoints.index('right_knee'), keypoints.index('right_back_paw')],
    ]
    return kp_lines


AnimalKeypoints.CONNECTIONS = kp_connections_animal(AnimalKeypoints.NAMES)


def vis_animal_keypoints(img, kps, kp_thresh=0.5, alpha=0.7, lth=1, crad=2):
    """Visualizes keypoints (adapted from vis_one_image).
    kps has shape (3, #keypoints) where 4 rows are (x, y, prob).
    """
    dataset_keypoints = AnimalKeypoints.NAMES
    kp_lines = AnimalKeypoints.CONNECTIONS

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)
    kps = kps.astype(np.int64)

    # Draw the keypoints.
    for l in range(len(kp_lines)):
        i1 = kp_lines[l][0]
        i2 = kp_lines[l][1]
        p1 = kps[0, i1], kps[1, i1]
        p2 = kps[0, i2], kps[1, i2]
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=lth, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(
                kp_mask, p1,
                radius=crad, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.circle(
                kp_mask, p2,
                radius=crad, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    vis = (1 - alpha) * img + kp_mask * alpha
    return vis