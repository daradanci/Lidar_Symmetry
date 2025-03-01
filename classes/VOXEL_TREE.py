import numpy as np
import open3d as o3d
import matplotlib.cm as cm
from sklearn.cluster import DBSCAN


class VOXEL_TREE:
    def __init__(self, points=None, voxel_size=0.1):
        self.points = points
        self.voxel_size = voxel_size
        self.voxel_grid = None
        self.symmetry_scores_per_layer = []
        self.trunk_x = None
        self.trunk_y = None
        if points is not None:
            self.calculate_trunk_center()


    def voxelize(self):
        if self.points is None or len(self.points) == 0:
            raise ValueError("⚠ Облако точек пустое. Невозможно выполнить вокселизацию.")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)

        self.voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=self.voxel_size)
        print(f"✅ Вокселизация завершена. Размер вокселя: {self.voxel_size}")


    def get_voxel_centers(self):
        if self.voxel_grid is None:
            raise ValueError("⚠ Вокселизация не выполнена. Сначала вызовите метод voxelize().")

        voxel_centers = [voxel.grid_index for voxel in self.voxel_grid.get_voxels()]
        return np.array(voxel_centers)

    

    def calculate_trunk_center(self, z_threshold=0.1):
        if self.points is None or len(self.points) == 0:
            raise ValueError("⚠ Облако точек пустое. Невозможно вычислить центр ствола.")

        min_z = np.min(self.points[:, 2])
        layer_points = self.points[self.points[:, 2] <= min_z + z_threshold]

        if layer_points.shape[0] == 0:
            raise ValueError("⚠ В нижнем слое нет точек. Невозможно вычислить центр ствола.")

        self.trunk_x, self.trunk_y = np.mean(layer_points[:, :2], axis=0)
        print(f"✅ Центр ствола найден: X = {self.trunk_x:.2f}, Y = {self.trunk_y:.2f}")

   

    def calculate_symmetry(self, z_step=1.0):
        if self.voxel_grid is None:
            raise ValueError("⚠ Вокселизация не выполнена. Сначала вызовите метод voxelize().")

        self.calculate_trunk_center()

        voxel_centers = self.get_voxel_centers()
        z_min, z_max = np.min(voxel_centers[:, 2]), np.max(voxel_centers[:, 2])
        z_levels = np.arange(z_min, z_max, z_step)

        self.symmetry_scores_per_layer = []

        for z in z_levels:
            idx = np.where((voxel_centers[:, 2] >= z) & (voxel_centers[:, 2] < z + z_step))
            layer_voxels = voxel_centers[idx]

            if layer_voxels.shape[0] < 5:
                continue

            left_half = layer_voxels[:, 0] < self.trunk_x
            right_half = layer_voxels[:, 0] >= self.trunk_x

            score = min(np.sum(left_half), np.sum(right_half)) / max(np.sum(left_half), np.sum(right_half))
            self.symmetry_scores_per_layer.append(score)

        print("✅ Расчёт симметрии для каждого слоя завершён.")

    def visualize_voxels(self, color_map=["plasma", "viridis"], z_step=1.0, point_size=5.0):
        if self.voxel_grid is None:
            raise ValueError("⚠ Вокселизация не выполнена. Сначала вызовите метод voxelize().")

        self.calculate_trunk_center()

        voxel_centers = self.get_voxel_centers()
        z_min, z_max = np.min(voxel_centers[:, 2]), np.max(voxel_centers[:, 2])
        z_levels = np.arange(z_min, z_max, z_step)

        pcd = o3d.geometry.PointCloud()
        voxel_points = voxel_centers * self.voxel_size

        colors = np.zeros((voxel_points.shape[0], 3))

        trunk_center_coords = np.array([self.trunk_x, self.trunk_y])
        center_index = np.argmin(np.linalg.norm(voxel_points[:, :2] - trunk_center_coords, axis=1))

        for i, z in enumerate(z_levels):
            idx = np.where((voxel_centers[:, 2] >= z) & (voxel_centers[:, 2] < z + z_step))
            if len(idx[0]) == 0:
                continue

            colormap = cm.get_cmap(color_map[i % len(color_map)])
            distances = np.linalg.norm(voxel_points[idx][:, :2] - trunk_center_coords, axis=1)
            norm_distances = (distances - np.min(distances)) / (np.max(distances) - np.min(distances) + 1e-6)

            colors[idx] = colormap(norm_distances)[:, :3]

        colors[center_index] = [1, 0, 0]

        pcd.points = o3d.utility.Vector3dVector(voxel_points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)

        render_opt = vis.get_render_option()
        render_opt.background_color = np.array([0, 0, 0])
        render_opt.point_size = point_size

        vis.run()
        vis.destroy_window()

        print("✅ Визуализация вокселей завершена.")
