from pathlib import Path
import open3d as o3d


def clean_cloud(
    input_file,
    output_file
):

    cloud = o3d.io.read_point_cloud(
        str(input_file)
    )

    if len(cloud.points) == 0:
        raise ValueError(
            "Point cloud contains no points."
        )

    cloud, _ = cloud.remove_statistical_outlier(
        nb_neighbors=20,
        std_ratio=2.0
    )

    cloud.estimate_normals()

    o3d.io.write_point_cloud(
        str(output_file),
        cloud
    )

    print(
        f"Saved cleaned cloud: {output_file}"
    )

    print(
        f"Points: {len(cloud.points)}"
    )


if __name__ == "__main__":

    root = Path(__file__).resolve().parents[2]

    clean_cloud(
        root / "outputs" / "dense_cloud.ply",
        root / "outputs" / "reconstruction.ply"
    )