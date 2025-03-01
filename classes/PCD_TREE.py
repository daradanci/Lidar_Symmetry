from .PCD import PCD
from .PCD_UTILS import PCD_UTILS
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from time import time
from scipy.spatial.distance import cdist
from scipy.spatial.distance import euclidean

from tqdm import tqdm
import pyvista
import random
import open3d as o3d
import seaborn as sns
import matplotlib.pyplot as plt
import math
import circle_fit as cf
import statistics
from scipy.spatial import ConvexHull

class PCD_TREE(PCD):
    def __init__(self, points = None, intensity = None, RGBint = None, coordinate = None, polygon = None, lower_coordinate = None, upper_coordinate = None, offset = [0.0, 0.0],
                main_coordinate = None, height = None, length = None, diameter_LS = None, diameter_HLS = None, 
                crown_volume = None, crown_square = None, xy_crown_square = None, yz_crown_square = None, xz_crown_square = None,
                x_up = None, y_up = None, symmetry_score = None):
        super().__init__(points, intensity)
        self.coordinate = coordinate
        self.polygon = polygon
        self.lower_coordinate = lower_coordinate
        self.upper_coordinate = upper_coordinate
        self.offset = offset
        self.RGBint = RGBint
        self.main_coordinate = main_coordinate
        self.height = height
        self.length = length
        self.diameter_LS = diameter_LS
        self.diameter_HLS = diameter_HLS
        self.crown_volume = crown_volume
        self.crown_square = crown_square
        self.xy_crown_square = xy_crown_square
        self.yz_crown_square = yz_crown_square
        self.xz_crown_square = xz_crown_square
        self.x_up = x_up
        self.y_up = y_up
        self.symmetry_score = symmetry_score  

    def visual_layer(self, labels, main_cluster_id):
        p1 = pyvista.Plotter(window_size=[1000, 1000])
        
        for i in np.unique(labels):
            p1 = PCD_UTILS.visual_many(p1, self.points, i, labels, main_cluster_id)

        pdata = pyvista.PolyData([self.lower_coordinate[0], self.lower_coordinate[1], self.lower_coordinate[2]])
        sphere = pyvista.Sphere(radius=0.25, phi_resolution=10, theta_resolution=10)
        pc = pdata.glyph(scale=False, geom=sphere, orient=False)
        p1.add_mesh(pc)  

        pdata = pyvista.PolyData([self.upper_coordinate[0], self.upper_coordinate[1], self.upper_coordinate[2]])
        sphere = pyvista.Sphere(radius=0.25, phi_resolution=10, theta_resolution=10)
        pc = pdata.glyph(scale=False, geom=sphere, orient=False)
        p1.add_mesh(pc)  

        p1.show()

    def search_main_cluster(self, EPS, MIN_SAMPLES):
        P = pd.DataFrame(self.points, columns = ['X','Y','Z'])
        P['Intensity'] = self.intensity/max(self.intensity)
        P = P.fillna(0)
        X = np.asarray(P)

        clustering = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit(X)
        labels=clustering.labels_

        min_distances = []
        for i in np.unique(labels):
            if i>-1:
                idx_label = np.where(labels==i)
                i_data = self.points[idx_label]
                distances = cdist(i_data, [[self.lower_coordinate[0], self.lower_coordinate[1], self.points.min(axis=0)[2]]])
                min_distances.append(np.min(distances))  

        min_dist_i = min_distances.index(min(min_distances)) if (np.unique(labels).shape[0] > 1)|(np.unique(labels)[0] == 0) else -1

        pc_chosen = PCD_TREE(self.points, self.intensity)
        idx_chosen = np.where(labels==min_dist_i)
        pc_chosen.index_cut(idx_chosen)

        return pc_chosen, min_dist_i, labels
    
    def search_upper_coordinate(self, pc_chosen, main_cluster_id = 0, verbose = False, lbls = None):
        
        if pc_chosen.points.shape[0] > 0:
            center_chosen_data_m = np.asarray(self.lower_coordinate)[0:2]

            pc_for_center = PCD_TREE(pc_chosen.points, pc_chosen.intensity)

            CH = pd.DataFrame(pc_for_center.intensity/max(pc_for_center.intensity), columns = ['Intensity'])
            normals = pc_for_center.get_normals()
            CH['NormalsZ'] = normals[:,2]
            CH_norm = (CH-CH.min ())/ (CH.max () - CH.min ())
            XCH = np.asarray(CH_norm)

            try:
                clustering = DBSCAN(eps=0.05, min_samples=100).fit(XCH)
                labels=clustering.labels_

                idx_labels=np.where(labels<0)
                points_for_center = pc_for_center.points[idx_labels]
                points_for_center = PCD_UTILS.SOR(points_for_center)

                top_points_for_cen = points_for_center

                PC = pd.DataFrame(top_points_for_cen, columns = ['X','Y','Z'])
                ax = sns.kdeplot(x=PC.X, y=PC.Y, shade=True)

                centers_labels_plot = []
                for path in ax.collections[-1].get_paths():
                    x, y = path.vertices.mean(axis=0)
                    centers_labels_plot.append([x, y])
                #     ax.plot(x, y, "ro")
                # plt.show()
                centers_labels_plot = np.asarray(centers_labels_plot)

                if centers_labels_plot.shape[0]>0:
                    min_distance = float('inf')
                    closest_point = None
                    for center in centers_labels_plot:
                        distance = euclidean(center, center_chosen_data_m)
                        if distance < min_distance:
                            min_distance = distance
                            closest_point = center
                else: 
                    closest_point = self.lower_coordinate
            except ValueError:
                closest_point = self.lower_coordinate

            self.upper_coordinate = [closest_point[0], closest_point[1], self.points.max(axis=0)[2]]
            if verbose:
                self.visual_layer(lbls, main_cluster_id)

            self.points = pc_chosen.points
            self.intensity = pc_chosen.intensity
        else:
            self.upper_coordinate = self.lower_coordinate
            if verbose:
                self.visual_layer(lbls, main_cluster_id)
            self.points = np.asarray([[0,0,0]])
            self.intensity = [0]
        uc = np.asarray(self.upper_coordinate)
        lc = np.asarray(self.lower_coordinate)
        if np.linalg.norm(uc[0:2] - lc[0:2]) > 0.4:
            uc[0:2] = lc[0:2] + (uc[0:2] - lc[0:2])/4
            self.upper_coordinate = uc
        offsetX = self.upper_coordinate[0] - self.lower_coordinate[0]
        offsetY = self.upper_coordinate[1] - self.lower_coordinate[1]
        self.offset = [offsetX, offsetY]
    
    def process_layer(self, EPS, MIN_SAMPLES, verbose = False):
        if self.points.shape[0]>1:
            pc_chosen, main_cluster_id, lbls = self.search_main_cluster(EPS, MIN_SAMPLES)
            self.search_upper_coordinate(pc_chosen, main_cluster_id = main_cluster_id, verbose = verbose, lbls = lbls)
        else:
            self.upper_coordinate = self.lower_coordinate
            self.points = np.asarray([[0,0,0]])
            self.intensity = [0]

    def estimate_height(self):
        x_min, y_min, z_min = self.points.min(axis=0)
        x_max, y_max, z_max = self.points.max(axis=0)
        height_tree= z_max - z_min
        self.height = float(PCD_UTILS.toFixed(height_tree,5))
    
    def estimate_length(self):
        arg_x_min, arg_y_min, arg_z_min = self.points.argmin(axis=0)
        arg_x_max, arg_y_max, arg_z_max = self.points.argmax(axis=0)
        length_tree = math.sqrt((self.points[arg_z_max][0]-self.points[arg_z_min][0])**2+(self.points[arg_z_max][1]-self.points[arg_z_min][1])**2+(self.points[arg_z_max][2]-self.points[arg_z_min][2])**2)
        self.length = float(PCD_UTILS.toFixed(length_tree,5))

    def search_main_coordinate(self, df, fname):
        x_value = df.loc[df['Name_tree'] == fname, 'X'].values[0]
        y_value = df.loc[df['Name_tree'] == fname, 'Y'].values[0]
        self.main_coordinate = [x_value, y_value]

    def search_slice(self, intensity_cut = 0):
        pc_slice = PCD_TREE(points = self.points, intensity = self.intensity)
        idx_labels=np.where((pc_slice.points[:,2] > pc_slice.points.min(axis=0)[2]) & (pc_slice.points[:,2] <= pc_slice.points.min(axis=0)[2] + 2))
        pc_slice.index_cut(idx_labels)
        idx_labels = np.where(pc_slice.intensity >= intensity_cut)
        pc_slice.index_cut(idx_labels)
        pc_slice.RGBint = pc_slice.intensity/max(pc_slice.intensity)
        return pc_slice

    def expansion_via_spheres(check, chosen, main_coordinate):
        check_points, check_intensity = check.points, check.intensity
        chosen_data, chosen_intensity = chosen.points, chosen.intensity
        j = 0
        r_points = [[0,0,0]]
        r_points = np.asarray(r_points)
        for point_main in chosen_data:
            dist = math.sqrt((point_main[0] - main_coordinate[0])**2 + (point_main[1] - main_coordinate[1])**2)
            if dist <= 0.3:
                d = 0.1
                idx_labels=np.where((check_points[:,0]>point_main[0]-d) & (check_points[:,0]<point_main[0]+d) & (check_points[:,1]>point_main[1]-d) & (check_points[:,1]<point_main[1]+d)& (check_points[:,2]>point_main[2]-d) & (check_points[:,2]<point_main[2]+d))
                ch_points = check_points[idx_labels]
                ch_intensity = check_intensity[idx_labels]
                if j == 0:
                    r_points = np.vstack((chosen_data, ch_points))
                    r_intensity = np.hstack((chosen_intensity, ch_intensity))
                else:
                    r_points = np.vstack((r_points, ch_points))
                    r_intensity = np.hstack((r_intensity, ch_intensity))
                j += 1
        if r_points.shape[0]>1:
            r_points_set = list(set(tuple(x) for x in r_points.tolist()))
            r_points_set = np.asarray(r_points_set)
            r_points_set = r_points_set.tolist()
            ind_int = []
            for point in r_points_set:
                index = np.where((r_points == point).all(axis=1))[0][0]
                ind_int.append(index)
            ind_int = np.asarray(sorted(ind_int))
            r_intensity = r_intensity[ind_int] 
            r_points = np.asarray(r_points_set)
        else:
            r_points = chosen_data
            r_intensity = chosen_intensity
        pc_expsph = PCD(points = r_points, intensity = r_intensity)
        return pc_expsph
    
    def search_points_for_center(self, pc_slice, dim = 0.3):

        CH = pd.DataFrame(pc_slice.RGBint, columns = ['Intensity'])
        normals = pc_slice.get_normals()
        CH['NormalsZ'] = normals[:,2]
        XCH = np.asarray(CH)

        clustering = DBSCAN(eps=0.05, min_samples=100).fit(XCH)
        labels=clustering.labels_
        idx_labels=np.where(labels<0)

        pc_pfc = PCD_TREE(points = pc_slice.points, intensity = pc_slice.intensity)
        pc_pfc.index_cut(idx_labels)

        idx_labels=np.where((pc_pfc.points[:,0]>self.main_coordinate[0]-dim) & (pc_pfc.points[:,0]<self.main_coordinate[0]+dim) & (pc_pfc.points[:,1]>self.main_coordinate[1]-dim) & (pc_pfc.points[:,1]<self.main_coordinate[1]+dim))
        pc_pfc.index_cut(idx_labels)

        pc_chosen = PCD_TREE(points = self.points, intensity = self.intensity)
        idx_labels=np.where((pc_chosen.points[:,0]>self.main_coordinate[0]-dim) & (pc_chosen.points[:,0]<self.main_coordinate[0]+dim) & (pc_chosen.points[:,1]>self.main_coordinate[1]-dim) & (pc_chosen.points[:,1]<self.main_coordinate[1]+dim))
        pc_chosen.index_cut(idx_labels)

        pc_expsph = PCD_TREE.expansion_via_spheres(pc_chosen, pc_pfc, self.main_coordinate)

        return pc_expsph
    
    def estimate_diameter(self, pc_expsph, pc_slice):

        idx_labels = np.where(pc_expsph.intensity>=5000)
        pc_expsph.index_cut(idx_labels)
 
        idx_labels = np.where(pc_expsph.points[:,2]<=pc_expsph.points.min(axis=0)[2]+3)
        pc_expsph.index_cut(idx_labels)

        idx_labels = np.where(pc_slice.intensity>=5000)
        pc_slice.index_cut(idx_labels)
        idx_labels = np.where(pc_slice.points[2]<=pc_slice.points.min(axis=0)[2]+3)
        pc_slice.index_cut(idx_labels)

        r_points = pc_expsph.points

        x_min, y_min, z_min = r_points.min(axis=0)
        x_max, y_max, z_max = r_points.max(axis=0)

        num_layers = 4
        layer = (z_max-z_min)/num_layers
        rh_list = []
        r_list = []

        for i in range(num_layers):
            idx_labels = np.where((r_points[:,2]>=i*layer+z_min)&(r_points[:,2]<(i+1)*layer+z_min))
            points_layer_i = r_points[idx_labels]

            try:
                if points_layer_i.shape[0]>1:
                    xc,yc,r,_ = cf.least_squares_circle(points_layer_i)
                    xc,yc,rh,_ = cf.hyper_fit(points_layer_i)
                else:
                    xc,yc,r,rh = 0,0,0,0
            except:
                xc,yc,r,_ = 0,0,0,0
                xc,yc,rh,_ = 0,0,0,0
            rh_list.append(rh)
            r_list.append(r)

        if len(r_list) == 0:
            for i in range(num_layers):
                idx_labels = np.where((pc_slice.points[:,2]>=i*layer+z_min)&(pc_slice.points[:,2]<(i+1)*layer+z_min))
                points_layer_i = pc_slice.points[idx_labels]
            try:
                if points_layer_i.shape[0]>0:
                    xc,yc,r,_ = cf.least_squares_circle(points_layer_i)
                else:
                    xc,yc,r,_ = 0,0,0,0
            except:
                xc,yc,r,_ = 0,0,0,0
            r_list.append(r)

        if len(rh_list) == 0:
            for i in range(num_layers):
                idx_labels = np.where((pc_slice.points[:,2]>=i*layer+z_min)&(pc_slice.points[:,2]<(i+1)*layer+z_min))
                points_layer_i = pc_slice.points[idx_labels]
            try:
                if points_layer_i.shape[0]>0:
                    xc,yc,rh,_ = cf.hyper_fit(points_layer_i)
                else:
                    xc,yc,rh,_ = 0,0,0,0
            except:
                xc,yc,rh,_ = 0,0,0,0
            rh_list.append(rh)
        

        r_median = statistics.median(r_list)
        rh_median = statistics.median(rh_list)

        x_min, y_min, z_min = pc_expsph.points.min(axis=0)
        x_max, y_max, z_max = pc_expsph.points.max(axis=0)
        check_r_median = ((x_max - x_min) + (y_max - y_min))/4
        if (r_median > 0.65) or (r_median > 2.1*check_r_median) or (r_median == 0.0):
            r_median = check_r_median
        if (rh_median > 0.65) or (rh_median > 2.1*check_r_median) or (rh_median == 0.0):
            rh_median = check_r_median

        breast_diameter_tree = 100*float(PCD_UTILS.toFixed(r_median*2,4))
        breast_diameter_tree_hyper = 100*float(PCD_UTILS.toFixed(rh_median*2,4))
        breast_diameter_tree = float(PCD_UTILS.toFixed(breast_diameter_tree,2))
        breast_diameter_tree_hyper = float(PCD_UTILS.toFixed(breast_diameter_tree_hyper,2))

        self.diameter_LS = breast_diameter_tree
        self.diameter_HLS = breast_diameter_tree_hyper

    def search_points_no_trunk(self, dim = 0.5):
        P = pd.DataFrame(self.RGBint, columns = ['Intensity'])
        normals = self.get_normals()
        P['NormalsZ'] = normals[:,2]
        X = np.asarray(P)

        clustering = DBSCAN(eps=0.05, min_samples=100).fit(X)
        labels=clustering.labels_

        idx_layer=np.where(labels>-1)
        points_no_trunk = self.points[idx_layer]

        idx_labels=np.where(labels<0)
        points_trunk = self.points[idx_labels]

        points_trunk_sor = PCD_UTILS.SOR(points_trunk)
        points_trunk_sor_intensity = np.full(points_trunk_sor.shape[0], 0) ### need fix
        pc_trunk_sor = PCD(points = points_trunk_sor, intensity = points_trunk_sor_intensity)

        idx_labels=np.where((points_no_trunk[:,0]>self.main_coordinate[0]-dim) & (points_no_trunk[:,0]<self.main_coordinate[0]+dim) & (points_no_trunk[:,1]>self.main_coordinate[1]-dim) & (points_no_trunk[:,1]<self.main_coordinate[1]+dim))
        check_points = points_no_trunk[idx_labels]
        check_intensity = np.full(check_points.shape[0], 0)  ### need fix
        pc_check = PCD(points = check_points, intensity = check_intensity)

        pc_r_points = PCD_TREE.expansion_via_spheres(pc_check, pc_trunk_sor, self.main_coordinate)

        r_points = pc_r_points.points   ### need fix

        uniq_points = list(set(tuple(x) for x in self.points)) 
        uniq_points = np.asarray(uniq_points)
        points_no_trunk = [y for y in uniq_points if y not in r_points]
        points_no_trunk = np.asarray(points_no_trunk)

        return points_no_trunk
    
    def estimate_crown(self, points_no_trunk):
        clustering = DBSCAN(eps=1, min_samples=100).fit(points_no_trunk)
        labels=clustering.labels_

        r_points = np.array([[0,0,0]])

        if np.unique(labels).shape[0] != 1:      
            max_z_values = []
            for i in np.unique(labels):
                if i>-1:
                    idx_layer=np.where(labels==i)
                    i_data = points_no_trunk[idx_layer]
                    index = i_data[:, 2].argmin()
                    max_z_value = i_data[index]
                    max_z_values.append(max_z_value)
            max_z_values = np.asarray(max_z_values)
            
            idx_labels=np.where(max_z_values[:,2]>self.points.max(axis=0)[2]-2)
            max_z_values = max_z_values[idx_labels]

            for i in range(max_z_values.shape[0]):
                idx_layer=np.where(labels==i)
                i_data = points_no_trunk[idx_layer]
                if i==0:
                    r_points = np.copy(i_data)
                else: 
                    r_points = np.vstack((r_points, i_data))
            if r_points.shape[0] == 1:
                r_points = points_no_trunk
        else:
            r_points = points_no_trunk

        if (r_points.shape[0] != 1) & (r_points.shape[0] > 4):
            # x_minF, y_minF, z_minF = r_points.min(axis=0)
            # x_maxF, y_maxF, z_maxF = r_points.max(axis=0)
            # crown_height = z_maxF - z_minF
            # crown_to_height = z_minF - self.points.min(axis=0)[2]

            hull = ConvexHull(r_points)
            crown_volume = hull.volume
            self.crown_volume = PCD_UTILS.toFixed(crown_volume,5)

            crown_square = hull.area
            self.crown_square = PCD_UTILS.toFixed(crown_square,5)

            hull = ConvexHull(r_points[:,0:2])
            xy_crown_square = hull.area
            self.xy_crown_square = PCD_UTILS.toFixed(xy_crown_square,5)

            hull = ConvexHull(r_points[:,1:3])
            yz_crown_square = hull.area
            self.yz_crown_square = PCD_UTILS.toFixed(yz_crown_square,5)

            hull = ConvexHull(np.take(r_points, [0, 2], axis=1))
            xz_crown_square = hull.area
            self.xz_crown_square = PCD_UTILS.toFixed(xz_crown_square,5)
        else:
            self.crown_volume, self.crown_volume, self.crown_square, self.xy_crown_square, self.yz_crown_square, self.xz_crown_square = 0, 0, 0, 0, 0, 0

    def search_up_slice(self, down_point, intensity_cut = 0):
        pc_slice = PCD_TREE(points = self.points, intensity = self.intensity)
        idx_labels=np.where((pc_slice.points[:,2] > down_point) & (pc_slice.points[:,2] <= pc_slice.points.max(axis=0)[2]))
        pc_slice.index_cut(idx_labels)
        idx_labels = np.where(pc_slice.intensity >= intensity_cut)
        pc_slice.index_cut(idx_labels)
        return pc_slice

    def search_up_coord(self, pc_slice, mode = 'kde'):
        if mode == 'kde':
            try:
                points_for_center = PCD_UTILS.SOR(pc_slice.points)

                PC = pd.DataFrame(points_for_center, columns = ['X','Y','Z'])
                ax = sns.kdeplot(x=PC.X, y=PC.Y, shade=True)

                centers_labels_plot = []
                for path in ax.collections[-1].get_paths():
                    x, y = path.vertices.mean(axis=0)
                    centers_labels_plot.append([x, y])
                centers_labels_plot = np.asarray(centers_labels_plot)

                if centers_labels_plot.shape[0]>0:
                    min_distance = float('inf')
                    closest_point = None
                    for center in centers_labels_plot:
                        distance = euclidean(center, self.main_coordinate)
                        if distance < min_distance:
                            min_distance = distance
                            closest_point = center
                else: 
                    closest_point = self.main_coordinate
            except ValueError:
                closest_point = self.main_coordinate
        elif mode == 'median':
            try:
                points_for_center = PCD_UTILS.SOR(pc_slice.points)
                closest_point = PCD_UTILS.center_m(points_for_center[:,0:2])
            except ValueError:
                closest_point = self.main_coordinate
        elif mode == 'highest':
            try:
                points_for_center = PCD_UTILS.SOR(pc_slice.points)
                closest_point = max(points_for_center, key=lambda point: point[2])
            except ValueError:
                closest_point = self.main_coordinate
        else:
            print("There is no such algorithm. Choose from existing: 'kde', 'median', 'highest'")
            closest_point = self.main_coordinate
        
        self.x_up = closest_point[0]
        self.y_up = closest_point[1]



class PCD_TREE(PCD):
    def __init__(self, points=None, intensity=None, trunk_x=None, trunk_y=None, **kwargs):
        """
        Класс для представления дерева в формате облака точек.

        :param points: массив координат точек (x, y, z).
        :param intensity: массив значений интенсивности точек.
        :param trunk_x: координата X ствола.
        :param trunk_y: координата Y ствола.
        :param kwargs: дополнительные параметры.
        """
        super().__init__(points, intensity)
        self.trunk_x = trunk_x  # Координата X ствола
        self.trunk_y = trunk_y  # Координата Y ствола
        self.symmetry_score = None  # Общий коэффициент симметрии дерева
        self.symmetry_scores_per_layer = []  # Массив коэффициентов симметрии по слоям
        self.cluster_labels = None  # Метки кластеров для точек дерева
        self.clustered_points = None  # Сохраненные кластеризованные точки
        self.voxels = None  # Хранение вокселизированных точек

    def get_active_points(self):
        """
        Возвращает активное представление точек дерева — либо воксели, либо исходные точки.
        """
        return self.voxels if self.voxels is not None else self.points

    def voxelize_tree(self, voxel_size=0.1):
        """
        Применяет вокселизацию к облаку точек дерева.

        :param voxel_size: Размер вокселя.
        """
        if self.points is None or self.points.shape[0] == 0:
            print("⚠ Ошибка: У дерева нет точек для вокселизации.")
            return

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        voxel_grid = pcd.voxel_down_sample(voxel_size=voxel_size)
        self.voxels = np.asarray(voxel_grid.points)
        print(f"✅ Вокселизация завершена: {self.voxels.shape[0]} вокселей.")

    def cluster_tree(self, eps=0.1, min_samples=5, remove_noise=True):
        """
        Выполняет кластеризацию точек дерева с помощью DBSCAN.

        :param eps: Максимальное расстояние между точками для объединения в кластер.
        :param min_samples: Минимальное количество точек в кластере.
        :param remove_noise: Удалять ли шумовые точки (-1 метка в DBSCAN).
        :return: Список меток кластеров для каждой точки.
        """
        points = self.get_active_points()
        if points is None or points.shape[0] == 0:
            print("⚠ Ошибка: У дерева нет точек для кластеризации.")
            return None

        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        labels = clustering.labels_

        if remove_noise:
            valid_points = labels != -1
            self.clustered_points = points[valid_points]
            labels = labels[valid_points]
        else:
            self.clustered_points = points

        self.cluster_labels = labels
        print(f"✅ Кластеризация завершена: найдено {len(set(labels)) - (1 if -1 in labels else 0)} кластеров.")
        return labels
    

    def set_trunk_center(self, z_threshold=0.1, min_points=10, eps=0.05, min_samples=3):
        """
        Определяет центр ствола по первому слою точек с помощью DBSCAN.

        :param z_threshold: Максимальная высота слоя для поиска центра ствола.
        :param min_points: Минимальное количество точек для вычисления центра.
        :param eps: Максимальное расстояние между точками для объединения в кластер.
        :param min_samples: Минимальное количество точек в кластере для его выделения.
        """
        # Получаем точки (вокселизированные или исходные)
        points = self.get_active_points()

        if points is None or points.shape[0] == 0:
            print("⚠ Ошибка: Облако точек пустое.")
            return

        # Передаём массив `points` (а не `self`)
        center = find_trunk_center(points, z_threshold, min_points, eps, min_samples)
        
        if center:
            self.trunk_x, self.trunk_y = center
            print(f"✅ Центр ствола определён: X = {self.trunk_x:.2f}, Y = {self.trunk_y:.2f}")
        else:
            print("⚠ Недостаточно точек или кластер не найден.")



    def measure_tree_symmetry(self, z_step=1.0, angle_step=1):
        """
        Оценивает симметричность дерева, используя координаты ствола для всех слоёв.

        :param z_step: Высота слоя (размер шага по Z).
        :param angle_step: Угол поворота для оценки симметрии.
        :return: Средний коэффициент симметрии (диапазон 0-1).
        """
        points = self.get_active_points()
        if self.trunk_x is None or self.trunk_y is None:
            raise ValueError("Не заданы координаты ствола (trunk_x, trunk_y).")

        z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])
        z_levels = np.arange(z_min, z_max, z_step)
        self.symmetry_scores_per_layer = []

        for z in z_levels:
            idx = np.where((points[:, 2] >= z) & (points[:, 2] < z + z_step))
            slice_points = points[idx][:, :2]

            if slice_points.shape[0] < 10:
                continue

            trunk_center = np.array([self.trunk_x, self.trunk_y])
            angle_symmetry_scores = []

            for angle in np.arange(-45, 46, angle_step):
                rotation_matrix = np.array([
                    [np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
                    [np.sin(np.radians(angle)), np.cos(np.radians(angle))]
                ])
                rotated_points = (slice_points - trunk_center) @ rotation_matrix + trunk_center
                left_half = rotated_points[:, 0] < trunk_center[0]
                right_half = ~left_half
                score = min(np.sum(left_half), np.sum(right_half)) / max(np.sum(left_half), np.sum(right_half))
                angle_symmetry_scores.append(score)

            layer_symmetry = np.mean(angle_symmetry_scores)
            self.symmetry_scores_per_layer.append(layer_symmetry)

        self.symmetry_score = np.mean(self.symmetry_scores_per_layer) if self.symmetry_scores_per_layer else 0
        return self.symmetry_score






    def restore_symmetry(self, z_step=1.0, symmetry_threshold=0.9, voxel_size=0.1):
        """
        Восстанавливает симметрию дерева, используя воксельное представление.

        :param z_step: Высота слоя (по умолчанию 1.0 м)
        :param symmetry_threshold: Порог симметрии (если выше, не восстанавливаем)
        :param voxel_size: Размер вокселя для поиска соседних точек
        """
        if self.voxels is None:
            print("🔄 Вокселизация перед восстановлением...")
            self.voxelize_tree(voxel_size=voxel_size)

        if self.trunk_x is None or self.trunk_y is None:
            print("⚠ Ошибка: Не заданы координаты ствола (trunk_x, trunk_y).")
            return

        z_min, z_max = np.min(self.voxels[:, 2]), np.max(self.voxels[:, 2])
        z_levels = np.arange(z_min, z_max, z_step)

        print(f"🛠 Восстановление симметрии ({len(z_levels)} слоев) с учётом объёма...")

        new_voxels = self.voxels.copy()

        for i, z in enumerate(z_levels):
            idx = np.where((self.voxels[:, 2] >= z) & (self.voxels[:, 2] < z + z_step))
            layer_voxels = self.voxels[idx]

            if layer_voxels.shape[0] == 0:
                continue

            trunk_center = np.array([self.trunk_x, self.trunk_y])

            # Получаем коэффициент симметрии слоя (если есть, иначе 1)
            symmetry_factor = self.symmetry_scores_per_layer[i] if i < len(self.symmetry_scores_per_layer) else 1.0

            # Если слой уже почти симметричен, не трогаем его
            if symmetry_factor >= symmetry_threshold:
                continue

            # Масштабируемое восстановление: чем ниже симметрия, тем больше точек восстанавливается
            recovery_strength = 1 - symmetry_factor  # Чем ниже симметрия, тем выше recovery_strength

            left_half = layer_voxels[layer_voxels[:, 0] < self.trunk_x]
            right_half = layer_voxels[layer_voxels[:, 0] > self.trunk_x]

            missing_right = []
            missing_left = []

            for voxel in left_half:
                mirror_voxel = np.array([2 * self.trunk_x - voxel[0], voxel[1], voxel[2]])

                # Проверяем наличие зеркальной точки в правой части
                if not np.any(np.all(np.isclose(mirror_voxel[:2], right_half[:, :2], atol=voxel_size), axis=1)):
                    # Добавляем случайное смещение по Z внутри слоя
                    mirror_voxel[2] += np.random.uniform(-voxel_size / 2, voxel_size / 2)
                    if np.random.rand() < recovery_strength:
                        missing_right.append(mirror_voxel)

            for voxel in right_half:
                mirror_voxel = np.array([2 * self.trunk_x - voxel[0], voxel[1], voxel[2]])

                if not np.any(np.all(np.isclose(mirror_voxel[:2], left_half[:, :2], atol=voxel_size), axis=1)):
                    mirror_voxel[2] += np.random.uniform(-voxel_size / 2, voxel_size / 2)
                    if np.random.rand() < recovery_strength:
                        missing_left.append(mirror_voxel)

            # Преобразуем в numpy массивы и добавляем воксели в общую структуру
            missing_right = np.array(missing_right) if missing_right else np.empty((0, 3))
            missing_left = np.array(missing_left) if missing_left else np.empty((0, 3))

            new_voxels = np.vstack([new_voxels] + [arr for arr in [missing_right, missing_left] if arr.size > 0])

        self.voxels = new_voxels
        print(f"✅ Восстановление завершено с учётом объёма.")


  
from sklearn.cluster import DBSCAN

def find_trunk_center(points, z_threshold=0.1, min_points=10, eps=0.05, min_samples=5):
    """
    Определяет центр ствола по точкам нижнего слоя.

    :param points: Массив точек в формате numpy (X, Y, Z).
    :param z_threshold: Высота слоя, в пределах которой выбираются точки.
    :param min_points: Минимальное количество точек для определения центра.
    :param eps: Максимальное расстояние между точками для кластеризации (DBSCAN).
    :param min_samples: Минимальное количество точек в кластере (DBSCAN).
    :return: Координаты центра ствола (X, Y) или None, если точек недостаточно.
    """
    if points is None or points.shape[0] == 0:
        print("⚠ Ошибка: Нет точек для поиска центра ствола.")
        return None

    # Отбор точек на нижнем слое
    min_z = np.min(points[:, 2])
    layer_points = points[points[:, 2] <= min_z + z_threshold]

    if layer_points.shape[0] < min_points:
        print("⚠ Недостаточно точек в нижнем слое для поиска центра ствола.")
        return None

    # Применение DBSCAN для удаления выбросов и выделения основного кластера
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(layer_points[:, :2])
    labels = clustering.labels_

    # Оставляем только точки основного кластера (исключаем шум -1)
    core_points = layer_points[labels != -1]

    if core_points.shape[0] < min_points:
        print("⚠ Кластеризация не смогла выделить достаточный основной кластер.")
        return None

    # Вычисление центра ствола как среднего значения X и Y
    trunk_x, trunk_y = np.mean(core_points[:, :2], axis=0)
    print(f"✅ Центр ствола найден: X = {trunk_x:.2f}, Y = {trunk_y:.2f}")
    return trunk_x, trunk_y
