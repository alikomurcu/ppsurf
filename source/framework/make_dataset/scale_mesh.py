import os
import argparse
import sys

import trimesh
import numpy as np


def _normalize_mesh(file_in: str, file_out: str):

    mesh: trimesh.Trimesh = trimesh.load(file_in)

    # TODO: Task 1 Start
    # TODO: normalize the mesh to [-0.5...+0.5]
    # See documentation of Trimesh: https://trimsh.org/trimesh.html#trimesh.Trimesh
    # mesh.apply_transform(trimesh.transformations.identity_matrix())  # just a placeholder

    # Compute bounding box
    bounds = mesh.bounds  # shape: (2, 3) -> [[min_x, min_y, min_z], [max_x, max_y, max_z]]

    # Compute the center of the bounding box
    bbox_center = (bounds[0] + bounds[1]) / 2.0

    # Translate vertices to center at origin
    mesh.vertices -= bbox_center

    # Compute the scaling factor (largest extent of the bounding box)
    scale_factor = 1.0 / np.max(bounds[1] - bounds[0])

    # Apply uniform scaling to fit the mesh in [-0.5, 0.5]^3
    mesh.vertices *= scale_factor

    # TODO: Task 1 End

    if np.min(mesh.vertices) < -0.5 or np.max(mesh.vertices) > 0.5:
        raise ValueError('Given mesh exceeds the boundaries!')

    mesh.export(file_out)


def run(args):
    if args.input and args.output:
        file_base_name = os.path.basename(args.input)
        _normalize_mesh(args.input, os.path.join(args.output, file_base_name))


def main(args):
    parser = argparse.ArgumentParser(prog="scale_mesh")
    parser.add_argument("input", type=str, help="input file")
    parser.add_argument("output", type=str, help="output location")
    args = parser.parse_args(args)

    run(args)


if __name__ == '__main__':
    main(sys.argv[1:])
