from pathlib import Path
import open3d as o3d
import numpy as np


def create_mesh(
    input_file,
    output_file
):

    cloud = o3d.io.read_point_cloud(
        str(input_file)
    )

    if len(cloud.points) == 0:
        raise ValueError(
            "Point cloud is empty."
        )

    cloud.estimate_normals()

    cloud.orient_normals_consistent_tangent_plane(
        30
    )

    mesh, densities = (
        o3d.geometry.TriangleMesh
        .create_from_point_cloud_poisson(
            cloud,
            depth=9
        )
    )

    densities = np.asarray(
        densities
    )

    if len(densities) > 0:

        threshold = np.quantile(
            densities,
            0.03
        )

        vertices_to_remove = (
            densities < threshold
        )

        mesh.remove_vertices_by_mask(
            vertices_to_remove
        )

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    o3d.io.write_triangle_mesh(
        str(output_file),
        mesh
    )

    print(
        f"Mesh vertices: {len(mesh.vertices)}"
    )

    print(
        f"Mesh triangles: {len(mesh.triangles)}"
    )


if __name__ == "__main__":

    root = Path(__file__).resolve().parents[2]

    create_mesh(
        root / "outputs" / "dense_cloud.ply",
        root / "outputs" / "mesh.ply"
    )