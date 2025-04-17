import argparse
import os
import sys

import numpy as np
import trimesh
import trimesh.sample
from source.base import fs, point_cloud


def run(args):
    input_mesh = trimesh.load(args.input)
    input_name = os.path.splitext(os.path.basename(args.input))[0]
    seed = fs.filename_to_hash(args.input)
    num_samples = args.num_samples
    pts: np.ndarray

    # TODO: Task 2 Start
    # TODO: get num_samples random points on the surface of input_mesh
    # TODO: use the provided seed for a random number generator
    # RNG: https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.random.html
    # Trimesh: https://trimsh.org/trimesh.html#trimesh.Trimesh
    # Point picking: https://mathworld.wolfram.com/TrianglePointPicking.html
    # Point picking: https://mathworld.wolfram.com/TriangleInterior.html
    # Be careful if you choose a method based on barycentric coordinates. Getting a uniform distribution there is not trivial.
    # pts = np.zeros(1)  # just a placeholder
    # raise NotImplementedError('Task 2 is not implemented')  # delete this line when done

    # Initialize random number generator
    rng = np.random.default_rng(seed)

    # Get face areas and compute sampling probabilities
    face_areas = input_mesh.area_faces
    probabilities = face_areas / np.sum(face_areas)

    # Randomly choose faces based on their area probabilities
    chosen_faces = rng.choice(len(input_mesh.faces), size=num_samples, p=probabilities)

    # Get vertex coordinates of chosen faces
    triangles = input_mesh.triangles[chosen_faces]  # shape: (num_samples, 3, 3)

    # Generate barycentric coordinates uniformly within triangles
    u = rng.random(num_samples)
    v = rng.random(num_samples)

    # Ensure points are uniformly distributed inside triangles
    mask = u + v > 1
    u[mask] = 1 - u[mask]
    v[mask] = 1 - v[mask]
    w = 1 - u - v

    # Compute the random points
    pts = (triangles[:, 0].T * u + triangles[:, 1].T * v + triangles[:, 2].T * w).T

    # TODO: Task 2 End

    # pts must of shape (num_samples, 3)
    if pts.shape[0] != num_samples:
        raise ValueError('Wrong number of points given!')
    if pts.shape[1] != 3:
        raise ValueError('Given points have the wrong number of dimensions!')

    pts_xyz_file = os.path.join(args.output_vis, input_name + '.xyz')
    pts_npy_file = os.path.join(args.output_npy, input_name + '.npy')
    np.save(pts_npy_file, pts.astype(np.float32))
    point_cloud.write_xyz(file_path=pts_xyz_file, points=pts)


def main(args):
    parser = argparse.ArgumentParser(prog="sample_surface")
    parser.add_argument("--num_samples", type=int, help="num_samples", default=25000)
    parser.add_argument("input", type=str, help="input file")
    parser.add_argument("output_npy", type=str, help="output location")
    parser.add_argument("output_vis", type=str, help="output vis location")
    args = parser.parse_args(args)
    run(args)


if __name__ == '__main__':
    main(sys.argv[1:])
