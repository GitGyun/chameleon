import torch
import numpy as np
import cv2
import os
import plyfile
import scipy
import scipy.spatial


fx = 572.41140
px = 325.26110
fy = 573.57043
py = 242.04899
intrinsic_matrix = np.array([[fx, 0, px], [0, fy, py], [0, 0, 1]])


def normalize(x):
    return (x - min(x)) / (max(x) - min(x))


class MeshModel:
    def __init__(self, model, scale=10, transform=None):
        self.vertices = None
        self.indices = None
        self.colors = None
        self.collated = None
        self.vertex_buffer = None
        self.index_buffer = None
        self.bb = None
        self.diameter = None
        self.scale = scale
        self.transform = transform
        self.frames = []
        self.load(model)

    def load(self, model):
        ply = plyfile.PlyData.read(model)
        self.vertices = np.zeros((ply['vertex'].count, 3))
        self.vertices[:, 0] = np.array(ply['vertex']['x']) / self.scale
        self.vertices[:, 1] = np.array(ply['vertex']['y']) / self.scale
        self.vertices[:, 2] = np.array(ply['vertex']['z']) / self.scale
        if self.transform is not None:
            self.vertices = np.dot(self.vertices, self.transform[:, :3].T) + self.transform[:, 3]

        self.bb = []
        self.minx, self.maxx = min(self.vertices[:, 0]), max(self.vertices[:, 0])
        self.miny, self.maxy = min(self.vertices[:, 1]), max(self.vertices[:, 1])
        self.minz, self.maxz = min(self.vertices[:, 2]), max(self.vertices[:, 2])
       
        self.bb.append([self.minx, self.miny, self.minz])
        self.bb.append([self.minx, self.maxy, self.minz])
        self.bb.append([self.minx, self.miny, self.maxz])
        self.bb.append([self.minx, self.maxy, self.maxz])
        self.bb.append([self.maxx, self.miny, self.minz])
        self.bb.append([self.maxx, self.maxy, self.minz])
        self.bb.append([self.maxx, self.miny, self.maxz])
        self.bb.append([self.maxx, self.maxy, self.maxz])
        self.bb = np.asarray(self.bb, dtype=np.float32)
        self.diameter = max(scipy.spatial.distance.pdist(self.bb, 'euclidean'))
        self.colors = np.zeros((ply['vertex'].count, 3))
        self.colors[:, 0] = normalize(self.vertices[:, 0]) * 255
        self.colors[:, 1] = normalize(self.vertices[:, 1]) * 255
        self.colors[:, 2] = normalize(self.vertices[:, 2]) * 255


def load_mesh_model(model_path):
    if os.path.exists(model_path):
        model = MeshModel(model_path)
    else:
        transform_path = model_path.replace('mesh.ply', 'transform.dat')
        model_path = model_path.replace('mesh.ply', 'OLDmesh.ply')
        transform = np.loadtxt(transform_path, skiprows=1)[:, 1]
        transform = np.reshape(transform, newshape=[3, 4])
        transform[:, 3] *= 100
        model = MeshModel(model_path, transform=transform)

    return model


def texture_to_6dpose(Y, model, coord=None):
    if Y.ndim == 4:
        if coord is not None:
            return np.stack([texture_to_6dpose(y, model, c) for y, c in zip(Y, coord)])
        else:
            return np.stack([texture_to_6dpose(y, model) for y in Y])

    Y = torch.where(Y[:1].bool(), Y[1:], torch.zeros_like(Y[1:])) 
    Y = Y.permute(1, 2, 0).cpu().numpy()
    pose = predict_pose_uvw(Y, model, coord)

    return pose


def predict_pose_uvw(uvw_region, model, coord=None, return_inliers=False):
    """
    Predict pose given UVW correspondences
    Args:
        cam: camera matrix
        uvw_region: UVW region
        model: object model
        return_inliers: bool to return inliers

    Returns: estimated pose

    """
    nonzero_mask = uvw_region[:, :, 0] > 0
    uvw_values = uvw_region[nonzero_mask]

    uvw_region_u = uvw_values[:, 0] * (model.maxx - model.minx) + model.minx
    uvw_region_v = uvw_values[:, 1] * (model.maxy - model.miny) + model.miny
    uvw_region_w = uvw_values[:, 2] * (model.maxz - model.minz) + model.minz
    points_3d = np.stack([uvw_region_u, uvw_region_v, uvw_region_w], axis=1)

    grid_row, grid_column = np.nonzero(nonzero_mask.astype(np.int64))

    image_points = np.empty((len(grid_row), 2))
    image_points[:, 0] = grid_row
    image_points[:, 1] = grid_column
    
    if coord is not None:
        image_points[:, 0] = (image_points[:, 0].astype(np.float64) / uvw_region.shape[0] * (coord[3] - coord[1]))
        image_points[:, 1] = (image_points[:, 1].astype(np.float64) / uvw_region.shape[1] * (coord[2] - coord[0]))
        image_points[:, 0] = (image_points[:, 0].astype(np.float64) + coord[1])
        image_points[:, 1] = (image_points[:, 1].astype(np.float64) + coord[0])

    object_points = points_3d

    if return_inliers:
        predicted_pose, n_inliers = solvePnP(intrinsic_matrix, image_points, object_points, return_inliers)
        predicted_pose = predicted_pose[:3]
        return predicted_pose, n_inliers
    else:
        predicted_pose = solvePnP(intrinsic_matrix, image_points, object_points, return_inliers)
        predicted_pose = predicted_pose[:3]
        return predicted_pose


def solvePnP(cam, image_points, object_points, return_inliers=False, ransac_iter=250):
    """
    Solve PnP problem using resulting correspondences
    Args:
        cam: Camera matrix
        image_points: Correspondence points on the image
        object_points: Correspondence points on the model
        return_inliers: Bool for inliers return
        ransac_iter: Number of RANSAC iterations

    Returns: Resulting object pose (+ number of inliers)

    """
    dist_coeffs = None  # Assuming no lens distortion
    if image_points.shape[0] < 4:
        pose = np.eye(4)
        inliers = []
    else:
        image_points[:, [0, 1]] = image_points[:, [1, 0]]
        object_points = np.expand_dims(object_points, 1)
        image_points = np.expand_dims(image_points, 1)

        try:
            success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(object_points, image_points.astype(float), cam,
                                                                                       dist_coeffs, iterationsCount=ransac_iter,
                                                                                       reprojectionError=1.)
        except:
            success = False
            inliers = None

        # Get a rotation matrix
        pose = np.eye(4)
        if success:
            pose[:3, :3] = cv2.Rodrigues(rotation_vector)[0]
            pose[:3, 3] = np.squeeze(translation_vector)

        if inliers is None:
            inliers = []

    if return_inliers:
        return pose, len(inliers)
    else:
        return pose


def create_bounding_box(img, pose, pt_cld_data, color=(1, 0, 0), thickness=1, base_size=(480, 640), coord=None):
    "Create a bounding box around the object"
    img = img.copy()
    if (pose == 0).all():
        return img
    
    # 8 corner points of the ptcld data
    min_x, min_y, min_z = pt_cld_data.min(axis=0)
    max_x, max_y, max_z = pt_cld_data.max(axis=0)
    corners_3D = np.array([[max_x, min_y, min_z],
                           [max_x, min_y, max_z],
                           [min_x, min_y, max_z],
                           [min_x, min_y, min_z],
                           [max_x, max_y, min_z],
                           [max_x, max_y, max_z],
                           [min_x, max_y, max_z],
                           [min_x, max_y, min_z]])

    # convert these 8 3D corners to 2D points
    ones = np.ones((corners_3D.shape[0], 1))
    homogenous_coordinate = np.append(corners_3D, ones, axis=1)

    # Perspective Projection to obtain 2D coordinates for masks
    homogenous_2D = intrinsic_matrix @ (pose @ homogenous_coordinate.T)
    coord_2D = homogenous_2D[:2, :] / homogenous_2D[2, :]
    if coord is not None and coord.astype('bool').any():
        coord_2D[0] -= coord[0]
        coord_2D[1] -= coord[1]
        coord_2D[0] = coord_2D[0] * img.shape[1] / (coord[2] - coord[0])
        coord_2D[1] = coord_2D[1] * img.shape[0] / (coord[3] - coord[1])
    else:
        coord_2D[0] = coord_2D[0] * img.shape[1] / base_size[1]
        coord_2D[1] = coord_2D[1] * img.shape[0] / base_size[0]
    
    thres = 2e3
    coord_2D = coord_2D.clip(-thres, thres)
    coord_2D = ((np.floor(coord_2D)).T).astype(int)
    
    # pad images if the 2D coordinates are outside the image
    pad_w1 = -min(0, coord_2D[:, 0].min())
    pad_w2 = max(img.shape[1], coord_2D[:, 0].max()) - img.shape[1]
    pad_h1 = -min(0, coord_2D[:, 1].min())
    pad_h2 = max(img.shape[0], coord_2D[:, 1].max()) - img.shape[0]
    img = cv2.copyMakeBorder(img, pad_h1, pad_h2, pad_w1, pad_w2, cv2.BORDER_CONSTANT, value=0)
    coord_2D[:, 0] += pad_w1
    coord_2D[:, 1] += pad_h1

    # Draw lines between these 8 points
    img = cv2.line(img, tuple(coord_2D[0]), tuple(coord_2D[1]), color, thickness)
    img = cv2.line(img, tuple(coord_2D[0]), tuple(coord_2D[3]), color, thickness)
    img = cv2.line(img, tuple(coord_2D[0]), tuple(coord_2D[4]), color, thickness)
    img = cv2.line(img, tuple(coord_2D[1]), tuple(coord_2D[2]), color, thickness)
    img = cv2.line(img, tuple(coord_2D[1]), tuple(coord_2D[5]), color, thickness)
    img = cv2.line(img, tuple(coord_2D[2]), tuple(coord_2D[3]), color, thickness)
    img = cv2.line(img, tuple(coord_2D[2]), tuple(coord_2D[6]), color, thickness)
    img = cv2.line(img, tuple(coord_2D[3]), tuple(coord_2D[7]), color, thickness)
    img = cv2.line(img, tuple(coord_2D[4]), tuple(coord_2D[7]), color, thickness)
    img = cv2.line(img, tuple(coord_2D[4]), tuple(coord_2D[5]), color, thickness)
    img = cv2.line(img, tuple(coord_2D[5]), tuple(coord_2D[6]), color, thickness)
    img = cv2.line(img, tuple(coord_2D[6]), tuple(coord_2D[7]), color, thickness)

    # crop the image to remove the padding
    img = img[pad_h1:img.shape[0] - pad_h2, pad_w1:img.shape[1] - pad_w2]

    return img


def vis_pose(label, img, mesh_model, coord=None, thickness=1):
    if label.ndim == 3:
        assert img.ndim == 4
        if coord is not None:
            assert coord.ndim == 2
            return torch.stack([vis_pose(label_, img_, mesh_model, coord_, thickness=thickness) for label_, img_, coord_ in zip(label, img, coord)])
        else:
            return torch.stack([vis_pose(label_, img_, mesh_model, thickness=thickness) for label_, img_ in zip(label, img)])
    
    img = img.permute(1, 2, 0).numpy()
    vis = create_bounding_box(img, label, mesh_model.vertices, coord=coord, thickness=thickness)
    vis = torch.from_numpy(vis).permute(2, 0, 1).float()
    return vis


def ADD_score(pt_cld, true_pose, pred_pose, diameter):
    "Evaluation metric - ADD score"
    pred_pose[0:3, 0:3][np.isnan(pred_pose[0:3, 0:3])] = 1
    pred_pose[:, 3][np.isnan(pred_pose[:, 3])] = 0
    target = pt_cld @ true_pose[0:3, 0:3] + np.array(
        [true_pose[0, 3], true_pose[1, 3], true_pose[2, 3]])
    output = pt_cld @ pred_pose[0:3, 0:3] + np.array(
        [pred_pose[0, 3], pred_pose[1, 3], pred_pose[2, 3]])
    avg_distance = np.linalg.norm(output - target, axis=1).mean()
    threshold = diameter * 0.1
    if avg_distance <= threshold:
        return 1., avg_distance
    else:
        return 0., avg_distance
    