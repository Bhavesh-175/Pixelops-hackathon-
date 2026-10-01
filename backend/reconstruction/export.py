from pathlib import Path
import open3d as o3d


def export_model():

    root = Path(__file__).resolve().parents[2]

    mesh_path = root / "outputs" / "mesh.ply"

    obj_path = root / "outputs" / "model.obj"

    glb_path = root / "outputs" / "model.glb"

    mesh = o3d.io.read_triangle_mesh(
        str(mesh_path)
    )

    if len(mesh.vertices) == 0:
        raise ValueError(
            "Mesh is empty."
        )

    mesh.compute_vertex_normals()

    o3d.io.write_triangle_mesh(
        str(obj_path),
        mesh
    )

    o3d.io.write_triangle_mesh(
        str(glb_path),
        mesh
    )

    print(f"OBJ: {obj_path}")
    print(f"GLB: {glb_path}")


if __name__ == "__main__":

    export_model()