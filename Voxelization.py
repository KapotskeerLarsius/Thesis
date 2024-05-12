import open3d as o3d
import numpy as np
from pyntcloud import PyntCloud
import os
"""
This script was used to make a voxel grid of a point cloud.
In the end this was not used for the report.
"""
# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the Measurements folder
measurements_folder = os.path.join(current_dir,"Measurements")

paht = measurements_folder+ "\\Measurements_volume\\Volume_ref\\measurement_1\\merged_apple\\merged_apple.ply"

pcd = o3d.io.read_point_cloud(paht)
o3d.visualization.draw_geometries([pcd])
pcd.normals = o3d.utility.Vector3dVector(np.zeros(
        (1, 3)))  # invalidate existing normals

pcd.estimate_normals()
tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)

for alpha in np.logspace(np.log10(0.5), np.log10(0.01), num=4):
    print(f"alpha={alpha:.3f}")
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        pcd, alpha, tetra_mesh, pt_map)
    hull, _ = mesh.compute_convex_hull()
    hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
    hull_ls.paint_uniform_color((1, 0, 0))
    cloud = PyntCloud.from_instance("open3d", mesh)
    converted_triangle_mesh = cloud.to_instance("open3d", mesh=True)  # mesh=True by defau
    convex_hull_id = cloud.add_structure("convex_hull")
    convex_hull = cloud.structures[convex_hull_id]
    print("volume apple in cm^3 = ", convex_hull.volume * 1000000)
    o3d.visualization.draw_geometries([mesh,hull_ls])
    N = 10000
    pcd = mesh.sample_points_poisson_disk(number_of_points=N, init_factor=5)
    
    # fit to unit cube
    pcd.scale(1 / np.max(pcd.get_max_bound() - pcd.get_min_bound()),
            center=pcd.get_center())
    
    pcd.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(N, 3)))
    o3d.visualization.draw_geometries([pcd])

    print('voxelization')
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,
                                                                voxel_size=0.025)
    o3d.visualization.draw_geometries([voxel_grid])
