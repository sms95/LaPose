import os
import sys
import h5py
import glob
import numpy as np
import _pickle as cPickle
from tqdm import tqdm
from pdb import set_trace
# import trimesh

# from prepare_data.lib.utils import sample_points_from_mesh

def random_point(face_vertices):
    """ Sampling point using Barycentric coordiante.

    """
    r1, r2 = np.random.random(2)
    sqrt_r1 = np.sqrt(r1)
    point = (1 - sqrt_r1) * face_vertices[0, :] + \
        sqrt_r1 * (1 - r2) * face_vertices[1, :] + \
        sqrt_r1 * r2 * face_vertices[2, :]

    return point

def uniform_sample(vertices, faces, n_samples, with_normal=False):
    """ Sampling points according to the area of mesh surface.

    """
    sampled_points = np.zeros((n_samples, 3), dtype=float)
    normals = np.zeros((n_samples, 3), dtype=float)
    faces = vertices[faces]
    vec_cross = np.cross(faces[:, 1, :] - faces[:, 0, :],
                         faces[:, 2, :] - faces[:, 0, :])
    face_area = 0.5 * np.linalg.norm(vec_cross, axis=1)
    cum_area = np.cumsum(face_area)
    for i in range(n_samples):
        face_id = np.searchsorted(cum_area, np.random.random() * cum_area[-1])
        sampled_points[i] = random_point(faces[face_id, :, :])
        normals[i] = vec_cross[face_id]
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    if with_normal:
        sampled_points = np.concatenate((sampled_points, normals), axis=1)
    return sampled_points

def pairwise_distance(A, B):
    """ Compute pairwise distance of two point clouds.point

    Args:
        A: n x 3 numpy array
        B: m x 3 numpy array

    Return:
        C: n x m numpy array

    """
    diff = A[:, :, None] - B[:, :, None].T
    C = np.sqrt(np.sum(diff**2, axis=1))

    return C

def farthest_point_sampling(points, n_samples):
    """ Farthest point sampling.

    """
    selected_pts = np.zeros((n_samples,), dtype=int)
    dist_mat = pairwise_distance(points, points)
    # start from first point
    pt_idx = 0
    dist_to_set = dist_mat[:, pt_idx]
    for i in range(n_samples):
        selected_pts[i] = pt_idx
        dist_to_set = np.minimum(dist_to_set, dist_mat[:, pt_idx])
        pt_idx = np.argmax(dist_to_set)
    return selected_pts

def load_obj(path_to_file):
    """ Load obj file.

    Args:
        path_to_file: path

    Returns:
        vertices: ndarray
        faces: ndarray, index of triangle vertices

    """
    vertices = []
    faces = []
    with open(path_to_file, 'r') as f:
        for line in f:
            if line[:2] == 'v ':
                vertex = line[2:].strip().split(' ')
                vertex = [float(xyz) for xyz in vertex]
                vertices.append(vertex)
            elif line[0] == 'f':
                face = line[1:].replace('//', '/').strip().split(' ')
                face = [int(idx.split('/')[0])-1 for idx in face]
                faces.append(face)
            else:
                continue
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    return vertices, faces

def sample_points_from_mesh(path, n_pts, with_normal=False, fps=False, ratio=2):
    """ Uniformly sampling points from mesh model.

    Args:
        path: path to OBJ file.
        n_pts: int, number of points being sampled.
        with_normal: return points with normal, approximated by mesh triangle normal
        fps: whether to use fps for post-processing, default False.
        ratio: int, if use fps, sample ratio*n_pts first, then use fps to sample final output.

    Returns:
        points: n_pts x 3, n_pts x 6 if with_normal = True

    """
    vertices, faces = load_obj(path)
    if faces.shape[0] == 0:
        return None
    if fps:
        points = uniform_sample(vertices, faces, ratio*n_pts, with_normal)
        pts_idx = farthest_point_sampling(points[:, :3], n_pts)
        points = points[pts_idx]
    else:
        points = uniform_sample(vertices, faces, n_pts, with_normal)
    return points

def save_nocs_model_to_file(obj_model_dir):
    """ Sampling points from mesh model and normalize to NOCS.
        Models are centered at origin, i.e. NOCS-0.5
    """
    mug_meta = {}
    # used for re-align mug category
    special_cases = {'3a7439cfaa9af51faf1af397e14a566d': np.array([0.115, 0.0, 0.0]),
                     '5b0c679eb8a2156c4314179664d18101': np.array([0.083, 0.0, -0.044]),
                     '649a51c711dc7f3b32e150233fdd42e9': np.array([0.0, 0.0, -0.017]),
                     'bf2b5e941b43d030138af902bc222a59': np.array([0.0534, 0.0, 0.0]),
                     'ca198dc3f7dc0cacec6338171298c66b': np.array([0.120, 0.0, 0.0]),
                     'f42a9784d165ad2f5e723252788c3d6e': np.array([0.117, 0.0, -0.026])}

    # CAMERA dataset
    for subset in ['train', 'val']:
        camera = {}
        for synsetId in ['02876657', '02880940', '02942699', '02946921', '03642806', '03797390']:
            synset_dir = os.path.join(obj_model_dir, subset, synsetId)
            inst_list = sorted(os.listdir(synset_dir))
            for instance in tqdm(inst_list):
                path_to_mesh_model = os.path.join(synset_dir, instance, 'model.obj')
                model_points = sample_points_from_mesh(path_to_mesh_model, 1024, fps=True, ratio=3)
                if model_points is None: continue
                # flip z-axis in CAMERA
                model_points = model_points * np.array([[1.0, 1.0, -1.0]])
                # re-align mug category
                if synsetId == '03797390':
                    if instance == 'b9be7cfe653740eb7633a2dd89cec754':
                        # skip this instance in train set, improper mug model, only influence training.
                        continue
                    if instance in special_cases.keys():
                        shift = special_cases[instance]
                    else:
                        shift_x = (np.amin(model_points[:, 2]) - np.amax(model_points[:, 2])) / 2 - np.amin(model_points[:, 0])
                        shift = np.array([shift_x, 0.0, 0.0])
                    model_points += shift
                    size = 2 * np.amax(np.abs(model_points), axis=0)
                    scale = 1 / np.linalg.norm(size)
                    model_points *= scale
                    mug_meta[instance] = [shift, scale]
                camera[instance] = model_points
        with open(os.path.join(obj_model_dir, 'camera_{}.pkl'.format(subset)), 'wb') as f:
            cPickle.dump(camera, f)
    # Real dataset
    for subset in ['train','real_test']:
        real = {}
        inst_list = glob.glob(os.path.join(obj_model_dir, subset, '*.obj'))
        for inst_path in tqdm(inst_list):
            instance = os.path.basename(inst_path).split('.')[0]
            bbox_file = inst_path.replace('.obj', '.txt')
            bbox_dims = np.loadtxt(bbox_file)
            scale = np.linalg.norm(bbox_dims)
            model_points = sample_points_from_mesh(inst_path, 1024, fps=True, ratio=3)
            model_points /= scale
            # relable mug category
            if 'mug' in instance:
                shift_x = (np.amin(model_points[:, 2]) - np.amax(model_points[:, 2])) / 2 - np.amin(model_points[:, 0])
                shift = np.array([shift_x, 0.0, 0.0])
                model_points += shift
                size = 2 * np.amax(np.abs(model_points), axis=0)
                scale = 1 / np.linalg.norm(size)
                model_points *= scale
                mug_meta[instance] = [shift, scale]
            real[instance] = model_points
        with open(os.path.join(obj_model_dir, '{}.pkl'.format(subset)), 'wb') as f:
            cPickle.dump(real, f)
    # save mug_meta information for re-labeling
    with open(os.path.join(obj_model_dir, 'mug_meta.pkl'), 'wb') as f:
        cPickle.dump(mug_meta, f)


def save_model_to_hdf5(obj_model_dir, n_points, fps=False, include_distractors=False, with_normal=False):
    """ Save object models (point cloud) to HDF5 file.
        Dataset used to train the auto-encoder.
        Only use models from ShapeNetCore.
        Background objects are not inlcuded as default. We did not observe that it helps
        to train the auto-encoder.
    """
    catId_to_synsetId = {1: '02876657', 2: '02880940', 3: '02942699', 4: '02946921', 5: '03642806', 6: '03797390'}
    distractors_synsetId = ['00000000', '02954340', '02992529', '03211117']
    with open(os.path.join(obj_model_dir, 'mug_meta.pkl'), 'rb') as f:
        mug_meta = cPickle.load(f)
    # read all the paths to models
    print('Sampling points from mesh model ...')
    if with_normal:
        train_data = np.zeros((3000, n_points, 6), dtype=np.float32)
        val_data = np.zeros((500, n_points, 6), dtype=np.float32)
    else:
        train_data = np.zeros((3000, n_points, 3), dtype=np.float32)
        val_data = np.zeros((500, n_points, 3), dtype=np.float32)
    train_label = []
    val_label = []
    train_count = 0
    val_count = 0
    # CAMERA
    for subset in ['train', 'val']:
        for catId in range(1, 7):
            synset_dir = os.path.join(obj_model_dir, subset, catId_to_synsetId[catId])
            inst_list = sorted(os.listdir(synset_dir))
            for instance in tqdm(inst_list):
                path_to_mesh_model = os.path.join(synset_dir, instance, 'model.obj')
                if instance == 'b9be7cfe653740eb7633a2dd89cec754':
                    continue
                model_points = sample_points_from_mesh(path_to_mesh_model, n_points, with_normal, fps=fps, ratio=2)
                if model_points is None: continue
                model_points = model_points * np.array([[1.0, 1.0, -1.0]])
                if catId == 6:
                    # if instance not in mug_meta.keys(): continue
                    shift = mug_meta[instance][0]
                    scale = mug_meta[instance][1]
                    model_points = scale * (model_points + shift)
                if subset == 'train':
                    train_data[train_count] = model_points
                    train_label.append(catId)
                    train_count += 1
                else:
                    val_data[val_count] = model_points
                    val_label.append(catId)
                    val_count += 1
        # distractors
        if include_distractors:
            for synsetId in distractors_synsetId:
                synset_dir = os.path.join(obj_model_dir, subset, synsetId)
                inst_list = sorted(os.listdir(synset_dir))
                for instance in tqdm(inst_list):
                    path_to_mesh_model = os.path.join(synset_dir, instance, 'model.obj')
                    model_points = sample_points_from_mesh(path_to_mesh_model, n_points, with_normal, fps=fps, ratio=2)
                    model_points = model_points * np.array([[1.0, 1.0, -1.0]])
                    if subset == 'train':
                        train_data[train_count] = model_points
                        train_label.append(0)
                        train_count += 1
                    else:
                        val_data[val_count] = model_points
                        val_label.append(0)
                        val_count += 1
    # Real
    for subset in ['train','real_test']:
        path_to_mesh_models = glob.glob(os.path.join(obj_model_dir, subset, '*.obj'))
        for inst_path in sorted(path_to_mesh_models):
            instance = os.path.basename(inst_path).split('.')[0]
            if instance.startswith('bottle'):
                catId = 1
            elif instance.startswith('bowl'):
                catId = 2
            elif instance.startswith('camera'):
                catId = 3
            elif instance.startswith('can'):
                catId = 4
            elif instance.startswith('laptop'):
                catId = 5
            elif instance.startswith('mug'):
                catId = 6
            else:
                raise NotImplementedError
            model_points = sample_points_from_mesh(inst_path, n_points, with_normal, fps=fps, ratio=2)
            bbox_file = inst_path.replace('.obj', '.txt')
            bbox_dims = np.loadtxt(bbox_file)
            model_points /= np.linalg.norm(bbox_dims)
            if catId == 6:
                shift = mug_meta[instance][0]
                scale = mug_meta[instance][1]
                model_points = scale * (model_points + shift)
            if subset == 'real_train':
                train_data[train_count] = model_points
                train_label.append(catId)
                train_count += 1
            else:
                val_data[val_count] = model_points
                val_label.append(catId)
                val_count += 1

    num_train_instances = len(train_label)
    num_val_instances = len(val_label)
    assert num_train_instances == train_count
    assert num_val_instances == val_count
    train_data = train_data[:num_train_instances]
    val_data = val_data[:num_val_instances]
    train_label = np.array(train_label, dtype=np.uint8)
    val_label = np.array(val_label, dtype=np.uint8)
    print('{} shapes found in train dataset'.format(num_train_instances))
    print('{} shapes found in val dataset'.format(num_val_instances))

    # write to HDF5 file
    print('Writing data to HDF5 file ...')
    if with_normal:
        filename = 'ShapeNetCore_{}_with_normal.h5'.format(n_points)
    else:
        filename = 'ShapeNetCore_{}.h5'.format(n_points)
    hfile = h5py.File(os.path.join(obj_model_dir, filename), 'w')
    train_dataset = hfile.create_group('train')
    train_dataset.attrs.create('len', num_train_instances)
    train_dataset.create_dataset('data', data=train_data, compression='gzip', dtype='float32')
    train_dataset.create_dataset('label', data=train_label, compression='gzip', dtype='uint8')
    val_dataset = hfile.create_group('val')
    val_dataset.attrs.create('len', num_val_instances)
    val_dataset.create_dataset('data', data=val_data, compression='gzip', dtype='float32')
    val_dataset.create_dataset('label', data=val_label, compression='gzip', dtype='uint8')
    hfile.close()


if __name__ == '__main__':
    obj_model_dir = './NOCS/obj_models'
    # Save ground truth models for training deform network
    save_nocs_model_to_file(obj_model_dir)
    # Save models to HDF5 file for training the auto-encoder.
    save_model_to_hdf5(obj_model_dir, n_points=4096, fps=False)
    # Save nmodels to HDF5 file, which used to generate mean shape.
    save_model_to_hdf5(obj_model_dir, n_points=2048, fps=True)