import csv
import os
from .PCD import PCD
from .PCD_UTILS import PCD_UTILS
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from time import time
from scipy.spatial.distance import cdist
from scipy.spatial.distance import euclidean
from scipy.spatial import KDTree

from tqdm import tqdm
import pyvista
import random
import open3d as o3d
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.path import Path
import math
import circle_fit as cf
import statistics
from scipy.spatial import ConvexHull

from scipy.spatial import cKDTree


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
        self.tree_top = None  # Координаты макушки дерева (X, Y, Z)
        self.layer_centers = {}  # Словарь центров каждого слоя
        self.symmetry_score = None  # Общий коэффициент симметрии дерева
        self.symmetry_scores_per_layer = []  # Массив коэффициентов симметрии по слоям
        self.cluster_labels = None  # Метки кластеров для точек дерева
        self.clustered_points = None  # Сохраненные кластеризованные точки
        self.voxels = None  # Хранение вокселизированных точек

    def merge_with_other_trees(self, other_trees, merge_voxels=False):
        """
        Объединяет точки текущего дерева с другими деревьями.

        :param other_trees: список экземпляров PCD_TREE для слияния.
        :param merge_voxels: если True — объединяются воксели, иначе — исходные точки.
        """
        merged_points = self.get_active_points() if merge_voxels else self.points

        for tree in other_trees:
            other_points = tree.get_active_points() if merge_voxels else tree.points
            if other_points is None or other_points.shape[0] == 0:
                print(f"⚠ Пропуск дерева без точек: {tree.file_path}")
                continue

            if merged_points is None:
                merged_points = other_points
            else:
                merged_points = np.vstack((merged_points, other_points))

        if merge_voxels:
            self.voxels = merged_points
            print(f"✅ Слияние вокселей завершено: {self.voxels.shape[0]} точек.")
        else:
            self.points = merged_points
            print(f"✅ Слияние точек завершено: {self.points.shape[0]} точек.")


    def compute_layer_center(self, z):
        """
        Вычисляет центр слоя по Z, интерполируя между основанием ствола и макушкой.

        :param z: Высота слоя.
        :return: Координаты центра слоя (X, Y).
        """
        if self.trunk_x is None or self.trunk_y is None or self.tree_top is None:
            print("⚠ Ошибка: Точка макушки или ствола не найдена.")
            print(f'trunk_x: {self.trunk_x}, trunk_y: {self.trunk_y}, tree_top: {self.tree_top}')
            return None

        # Интерполяция X, Y координат по Z
        z_bottom = np.min(self.get_active_points()[:, 2])
        z_top = self.tree_top[2]

        # Нормализуем уровень слоя в диапазоне [0,1]
        t = (z - z_bottom) / (z_top - z_bottom)

        # Линейная интерполяция между нижней и верхней точками
        x_center = (1 - t) * self.trunk_x + t * self.tree_top[0]
        y_center = (1 - t) * self.trunk_y + t * self.tree_top[1]

        return x_center, y_center

    def get_active_points(self):
        """
        Возвращает активное представление точек дерева — либо воксели, либо исходные точки.
        """
        return self.voxels if self.voxels is not None else self.points

    def voxelize_tree(self, voxel_size=0.1):
        """
        Применяет вокселизацию с построением регулярной сетки вокселей.

        :param voxel_size: Размер вокселя.
        """
        if self.points is None or self.points.shape[0] == 0:
            print("⚠ Ошибка: У дерева нет точек для вокселизации.")
            return

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)

        # Построение воксельной сетки
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)

        voxels = voxel_grid.get_voxels()

        voxel_centers = []
        origin = np.array(voxel_grid.origin)

        for voxel in voxels:
            grid_index = np.array(voxel.grid_index, dtype=float)
            center = origin + (grid_index + 0.5) * voxel_size  # Центр каждой ячейки
            voxel_centers.append(center)

        self.voxels = np.array(voxel_centers)
        print(f"✅ Вокселизация завершена: {self.voxels.shape[0]} вокселей в сетке.")


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
    
    def find_tree_top(self):
        """
        Определяет макушку дерева как самую верхнюю точку.

        :return: Координаты макушки дерева (X, Y, Z) или None, если точек недостаточно.
        """
        points = self.get_active_points()
        if points is None or points.shape[0] == 0:
            print("⚠ Ошибка: У дерева нет точек.")
            return None

        # 🔎 Поиск самой высокой точки (максимальное Z)
        top_idx = np.argmax(points[:, 2])
        tree_top = points[top_idx]

        self.tree_top = tuple(tree_top)  # Сохраняем в формате (X, Y, Z)
        print(f"🌲 Макушка дерева найдена: X = {tree_top[0]:.2f}, Y = {tree_top[1]:.2f}, Z = {tree_top[2]:.2f}")
        return self.tree_top

    def generate_layer_polygon(self, z, z_step=1.0, voxel_size=0.1):
        """
        Генерирует многоугольник слоя на высоте z, охватывая все точки слоя.

        :param z: нижняя граница слоя
        :param z_step: высота слоя
        :param voxel_size: расстояние между точками на рёбрах многоугольника
        """
        if self.voxels is None or self.voxels.shape[0] == 0:
            print("⚠ У дерева нет вокселей для построения многоугольников.")
            return

        # Отбираем все точки в заданном слое по высоте
        layer_voxels = self.voxels[(self.voxels[:, 2] >= z) & (self.voxels[:, 2] < z + z_step)]

        if layer_voxels.shape[0] < 3:
            return  # Недостаточно точек для построения оболочки

        points_2d = layer_voxels[:, :2]  # только X и Y координаты
        polygon_points = []

        try:
            # 📌 Построение выпуклой оболочки
            hull = ConvexHull(points_2d)
            polygon = points_2d[hull.vertices]
        except:
            # ❎ Если не удалось построить — fallback: правильный шестиугольник вокруг центра
            avg_radius = np.mean(np.linalg.norm(points_2d - np.array([self.trunk_x, self.trunk_y]), axis=1))
            angles = np.linspace(0, 2 * np.pi, 7)[:-1]
            polygon = np.column_stack((self.trunk_x + avg_radius * np.cos(angles),
                                    self.trunk_y + avg_radius * np.sin(angles)))

        # 🔺 Добавляем вершины оболочки (метка 3)
        for point in polygon:
            polygon_points.append([point[0], point[1], z, 3])

        # 🔗 Добавляем промежуточные точки по рёбрам
        for j in range(len(polygon)):
            p1 = polygon[j]
            p2 = polygon[(j + 1) % len(polygon)]
            num_steps = max(1, int(np.linalg.norm(p1 - p2) / voxel_size))
            for step in range(num_steps):
                interp = p1 + (p2 - p1) * (step / num_steps)
                polygon_points.append([interp[0], interp[1], z, 3])

        # 📌 Добавление к self.voxels
        polygon_points = np.array(polygon_points)
        self.voxels = np.vstack([self.voxels, polygon_points])


    def generate_all_layer_polygons(self, z_step=1.0, voxel_size=0.1):
        """
        Генерирует многоугольники для всех слоев дерева.

        :param z_step: шаг по высоте
        :param voxel_size: размер вокселя (для построения границ)
        """
        points = self.get_active_points()
        if points is None or points.shape[0] == 0:
            print("⚠ Ошибка: У дерева нет точек для построения многоугольников.")
            return

        z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])
        z_levels = np.arange(z_min, z_max, z_step)

        print(f"🛠 Генерация многоугольников для {len(z_levels)} слоев...")
        for z in z_levels:
            self.generate_layer_polygon(z, voxel_size)

        print("✅ Завершено построение многоугольников.")


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
        Оценивает симметричность дерева по слоям, используя радиальную метрику.
        Для каждого слоя вычисляется корреляция между распределением радиусов r(θ) и r(θ+π),
        а также эллиптичность слоя на основе ковариации точек. Результаты логируются.
        
        :param z_step: Высота слоя (шаг по координате Z).
        :param angle_step: Шаг угла в градусах для построения гистограммы r(θ).
        :return: Средний коэффициент симметрии дерева (диапазон 0-1).
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
            layer_count = slice_points.shape[0]
            if layer_count < 10:
                # Недостаточно точек для оценки симметрии этого слоя
                self.symmetry_scores_per_layer.append(None)
                print(f"📊 Слой (z={z:.2f} м): недостаточно точек для оценки симметрии")
                continue
            
            # Центр слоя (используем координаты ствола как центр)
            trunk_center = np.array([self.trunk_x, self.trunk_y])
            # Переводим точки слоя в полярные координаты относительно центра
            rel_coords = slice_points - trunk_center
            angles = (np.degrees(np.arctan2(rel_coords[:, 1], rel_coords[:, 0])) + 360) % 360
            radii = np.linalg.norm(rel_coords, axis=1)
            
            # Строим распределение r(θ) и r(θ+π) по угловым секторам
            half_angles = np.arange(0, 180, angle_step)
            radial_A = []
            radial_B = []
            for base_angle in half_angles:
                angle_a_start = base_angle
                angle_a_end = base_angle + angle_step
                angle_b_start = base_angle + 180
                angle_b_end = base_angle + 180 + angle_step
                # Маски для точек, попадающих в сектора [angle_a_start, angle_a_end) и [angle_b_start, angle_b_end)
                mask_a = (angles >= angle_a_start) & (angles < angle_a_end)
                if angle_b_end <= 360:
                    mask_b = (angles >= angle_b_start) & (angles < angle_b_end)
                else:
                    # Если сектор θ+π переходит через 360°
                    wrap_angle = angle_b_end - 360.0
                    mask_b = ((angles >= angle_b_start) & (angles < 360.0)) | ((angles >= 0.0) & (angles < wrap_angle))
                # Максимальный радиус в каждом секторе (0, если нет точек)
                r_a = np.max(radii[mask_a]) if np.any(mask_a) else 0.0
                r_b = np.max(radii[mask_b]) if np.any(mask_b) else 0.0
                radial_A.append(r_a)
                radial_B.append(r_b)
            radial_A = np.array(radial_A, dtype=float)
            radial_B = np.array(radial_B, dtype=float)
            
            # Вычисляем корреляцию между r(θ) и r(θ+π)
            if radial_A.size == 0 or radial_B.size == 0:
                corr = 0.0
            elif np.std(radial_A) < 1e-9 and np.std(radial_B) < 1e-9:
                # Если распределения константные (все радиусы одинаковы) – считаем корреляцию идеальной (1.0)
                corr = 1.0
            else:
                # Pearson correlation
                corr_matrix = np.corrcoef(radial_A, radial_B)
                corr = corr_matrix[0, 1] if corr_matrix.shape == (2, 2) else 0.0
                if np.isnan(corr):
                    corr = 0.0
            
            # Эллиптичность слоя: отношение меньшей и большей собственных величин ковариационной матрицы
            cov = np.cov(rel_coords.T)  # 2x2 ковариационная матрица по X,Y
            eigvals = np.linalg.eigvals(cov)


            eigvals = np.sort(np.real(eigvals))
            ellipticity = 0.0
            if eigvals[-1] > 1e-9:
                ellipticity = eigvals[0] / eigvals[-1]
            
            # Комбинированный показатель симметрии слоя (учитывает радиальную симметрию и форму)
            corr = max(0.0, corr)  # отрицательные значения корреляции трактуем как 0
            layer_symmetry = corr * ellipticity
            self.symmetry_scores_per_layer.append(layer_symmetry)
            
            # Логгируем метрики слоя
            print(f"📊 Слой на высоте z={z:.2f} м: корреляция = {corr:.2f}, эллиптичность = {ellipticity:.2f}, симметрия = {layer_symmetry:.2f}")
        
        # Средний коэффициент симметрии дерева (по всем слоям с достаточным числом точек)
        valid_scores = [s for s in self.symmetry_scores_per_layer if s is not None]
        self.symmetry_score = float(np.mean(valid_scores)) if valid_scores else 0.0
        print(f"🌟 Общий коэффициент симметрии дерева: {self.symmetry_score:.3f}")
        
        # Сохраняем послойные коэффициенты симметрии в CSV (добавляется новая колонка)
        try:
            tree_id = getattr(self, "tree_id", None)
            if tree_id is None and hasattr(self, "file_path"):
                tree_id = os.path.splitext(os.path.basename(self.file_path))[0].split("_")[-1]
            write_symmetry_to_file(tree_id, self.symmetry_scores_per_layer)
        except Exception as e:
            print(f"⚠️ Ошибка при записи симметрии в файл: {e}")
        
        return self.symmetry_score

    def _build_radial_profile(self, slice_points_xy: np.ndarray, center_xy: np.ndarray, K: int = 180):
        """Строит радиальный профиль R(θ): средний радиус по K угловым секторам."""
        if slice_points_xy.shape[0] == 0:
            return np.zeros(K, dtype=float), None

        V = slice_points_xy - center_xy  # (N,2)
        angles = (np.degrees(np.arctan2(V[:, 1], V[:, 0])) + 360.0) % 360.0
        radii = np.linalg.norm(V, axis=1)

        bin_w = 360.0 / K
        idx = np.floor(angles / bin_w).astype(int) % K

        R = np.zeros(K, dtype=float)
        C = np.zeros(K, dtype=int)
        for r, k in zip(radii, idx):
            R[k] += r
            C[k] += 1
        C = np.maximum(C, 1)
        R = R / C
        return R, angles


    def _optimal_reflection_plane(self, slice_points_xy: np.ndarray, center_xy: np.ndarray, K: int = 180):
        """
        Ищет угол φ* (в градусах) плоскости отражения, максимизирующей зеркальную корреляцию профиля R(θ).
        Возвращает (phi_star_deg, corr_star) с нормировкой corr_star в [0,1].
        """
        R, _ = self._build_radial_profile(slice_points_xy, center_xy, K=K)
        if not np.any(R):
            return 0.0, 0.0

        best_phi = 0.0
        best_corr = -1.0
        bin_w = 360.0 / K
        half = K // 2
        R_centered = R - np.mean(R)

        for s in range(K):  # φ = s * bin_w
            Rp = np.roll(R_centered, -s)
            A = Rp[:half]
            B = Rp[::-1][:half]

            stdA = A.std()
            stdB = B.std()
            if stdA < 1e-9 and stdB < 1e-9:
                corr = 1.0
            elif stdA < 1e-9 or stdB < 1e-9:
                corr = 0.0
            else:
                corr = float(np.corrcoef(A, B)[0, 1])
                if np.isnan(corr):
                    corr = 0.0

            if corr > best_corr:
                best_corr = corr
                best_phi = s * bin_w

        best_corr = max(0.0, min(1.0, best_corr))
        return best_phi, best_corr


    def _reflect_points_about_line(self, points_xy: np.ndarray, center_xy: np.ndarray, phi_deg: float):
        """
        Отражает 2D-точки относительно прямой через center_xy под углом phi_deg к оси X.
        Формула отражения: R = 2uu^T - I, где u = [cosφ, sinφ].
        """
        phi = np.radians(phi_deg)
        u = np.array([np.cos(phi), np.sin(phi)])
        R = 2.0 * np.outer(u, u) - np.eye(2)
        V = points_xy - center_xy
        V_ref = (R @ V.T).T
        return V_ref + center_xy




    def restore_symmetry(self, neighbor_trees=None, z_step=1.0, voxel_size=0.1,
                        balance_factor=0.8, aggressive_radius=0.5,
                        symmetry_threshold=0.9, generate_mirrored=True,
                        mirror_plane: str = 'vertical_x',  # 'vertical_x' | 'optimal' | 'fixed_angle'
                        fixed_angle_deg: float = 0.0,      # используется, если mirror_plane='fixed_angle'
                        optimal_bins: int = 180):          # K для поиска оптимальной плоскости
        """
        Восстанавливает симметрию кроны дерева.
        Новые точки зеркалируются и проходят фильтрацию по ConvexHull и локальной плотности,
        затем выполняется дедупликация по voxel_size.

        :param neighbor_trees: список соседних деревьев (для балансировки)
        :param z_step: толщина слоя по Z
        :param voxel_size: размер вокселя для дедупликации
        :param balance_factor: доля переноса при балансировке
        :param aggressive_radius: радиус агрессивного отбора у соседа
        :param symmetry_threshold: если симметрия слоя >= порога, зеркалирование не выполняется
        :param generate_mirrored: включить генерацию зеркальных точек
        :param mirror_plane: плоскость отражения:
            - 'vertical_x' (по умолчанию): отражение по вертикальной оси X через центр слоя (как раньше)
            - 'optimal': отражение относительно найденной оптимальной плоскости φ* для слоя
            - 'fixed_angle': отражение относительно прямой под углом fixed_angle_deg к оси X
        :param fixed_angle_deg: угол для режима 'fixed_angle'
        :param optimal_bins: число угловых секторов для оценки φ* (обычно 180)
        """
        # Вокселизация при необходимости
        if self.voxels is None:
            print("🔄 Вокселизация перед восстановлением...")
            self.voxelize_tree(voxel_size=voxel_size)

        if self.tree_top is None:
            print("⚠️ Ошибка: Макушка дерева не найдена.")
            return

        if not hasattr(self, "recovered_voxels"):
            self.recovered_voxels = np.empty((0, 4))

        # Убедимся в наличии столбца меток
        if self.voxels.shape[1] == 3:
            self.voxels = np.column_stack((self.voxels, np.zeros(self.voxels.shape[0])))

        z_min, z_max = np.min(self.voxels[:, 2]), np.max(self.voxels[:, 2])
        z_levels = np.arange(z_min, z_max, z_step)
        print(f"🛠 Восстановление симметрии ({len(z_levels)} слоев)...")

        generated_points = []
        theta_max = 15  # макс. угол случайного отклонения (°)

        for i, z in enumerate(z_levels):
            center_x, center_y = self.compute_layer_center(z)
            if center_x is None or center_y is None:
                continue
            layer_voxels = self.voxels[(self.voxels[:, 2] >= z) & (self.voxels[:, 2] < z + z_step)]
            if layer_voxels.shape[0] == 0:
                continue

            symmetry_factor = None
            if i < len(self.symmetry_scores_per_layer):
                symmetry_factor = self.symmetry_scores_per_layer[i]

            if symmetry_factor is not None:
                print(f"📊 Симметрия слоя {i} (z={z:.2f} м): {symmetry_factor:.2f}")
            else:
                print(f"📊 Симметрия слоя {i} (z={z:.2f} м): недостаточно точек")

            # Зеркалирование слоя при низкой симметрии
            if generate_mirrored and symmetry_factor is not None and symmetry_factor < symmetry_threshold:
                recovery_strength = 1.0 - symmetry_factor

                # Параметры «естественного» поворота (как было)
                t_norm = i / max(len(z_levels) - 1, 1)
                angle_offset = (2 * t_norm - 1) * theta_max + np.random.uniform(-3, 3)  # ±3°
                angle_rad = np.radians(angle_offset)
                rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                            [np.sin(angle_rad),  np.cos(angle_rad)]])

                center_xy = np.array([center_x, center_y])
                layer_xy = layer_voxels[:, :2]
                to_mirror = layer_voxels  # берём все точки слоя; дедуп позже

                # === ВЫБОР ПЛОСКОСТИ ОТРАЖЕНИЯ ===
                if mirror_plane == 'vertical_x':
                    # Текущий способ: отражение по вертикальной оси X через центр
                    mirrored_xy = np.column_stack([2 * center_x - to_mirror[:, 0], to_mirror[:, 1]])
                elif mirror_plane == 'fixed_angle':
                    mirrored_xy = self._reflect_points_about_line(to_mirror[:, :2], center_xy, fixed_angle_deg)
                elif mirror_plane == 'optimal':
                    phi_star, corr_star = self._optimal_reflection_plane(layer_xy, center_xy, K=optimal_bins)
                    # print(f"Layer z={z:.2f}: φ*={phi_star:.1f}°, corr={corr_star:.2f}")
                    mirrored_xy = self._reflect_points_about_line(to_mirror[:, :2], center_xy, phi_star)
                else:
                    # fallback — как по умолчанию
                    mirrored_xy = np.column_stack([2 * center_x - to_mirror[:, 0], to_mirror[:, 1]])

                # Небольшой поворот вокруг центра для реалистичности
                rotated_xy = (mirrored_xy - center_xy) @ rotation_matrix + center_xy

                # Формируем новые воксели (метка 2) с лёгким шумом по Z
                new_points_layer = []
                for p_old, p_xy in zip(to_mirror, rotated_xy):
                    if np.random.rand() < recovery_strength:
                        new_points_layer.append([
                            p_xy[0],
                            p_xy[1],
                            p_old[2] + np.random.uniform(-voxel_size/2, voxel_size/2),
                            2.0
                        ])

                # --- Фильтр Convex Hull (по исходным точкам слоя) ---
                if len(new_points_layer) > 0:
                    try:
                        hull = ConvexHull(layer_xy)
                        hull_polygon = layer_xy[hull.vertices, :]
                        hull_path = Path(hull_polygon, closed=True)
                        new_xy = np.array(new_points_layer)[:, :2]
                        inside_mask = hull_path.contains_points(new_xy, radius=1e-9)
                        new_points_layer = [pt for j, pt in enumerate(new_points_layer) if inside_mask[j]]
                    except Exception:
                        pass

                # --- Ограничение локальной плотности (5-NN, 10-й перцентиль) ---
                if len(new_points_layer) > 0:
                    orig_points_2d = layer_xy
                    orig_count = orig_points_2d.shape[0]
                    if orig_count >= 5:
                        tree = cKDTree(orig_points_2d)
                        k_neighbors = min(6, orig_count)
                        dists, _ = tree.query(orig_points_2d, k=k_neighbors)
                        fifth_neighbor = dists[:, k_neighbors - 1] if dists.ndim == 2 and dists.shape[1] >= k_neighbors else None
                        if fifth_neighbor is not None:
                            d_thr = np.percentile(fifth_neighbor, 10)
                            new_xy = np.array(new_points_layer)[:, :2]
                            nd, _ = tree.query(new_xy, k=min(5, orig_count))
                            if nd.ndim == 1:
                                nd = nd.reshape(1, -1)
                            keep = []
                            for dd in nd:
                                keep.append(False if dd.shape[0] >= 5 and dd[-1] <= d_thr else True)
                            new_points_layer = [pt for j, pt in enumerate(new_points_layer) if keep[j]]

                generated_points.extend(new_points_layer)

        # Добавляем сгенерированные точки
        if len(generated_points) > 0:
            generated_points = np.array(generated_points)
            self.voxels = np.vstack([self.voxels, generated_points])
            self.recovered_voxels = np.vstack([self.recovered_voxels, generated_points]) if self.recovered_voxels.size else generated_points

        # Балансировка с соседями
        if neighbor_trees:
            for neighbor in neighbor_trees:
                for z in z_levels:
                    self.balance_layer_with_neighbor(neighbor, z, z_step=z_step)

        # Дедупликация на сетке voxel_size
        if self.voxels is not None and self.voxels.shape[0] > 0:
            orig_points_mask = (self.voxels[:, 3] == 0)
            if np.any(orig_points_mask):
                origin_coords = np.min(self.voxels[orig_points_mask][:, 0:3], axis=0)
            else:
                origin_coords = np.min(self.voxels[:, 0:3], axis=0)

            indices = np.floor((self.voxels[:, 0:3] - origin_coords) / voxel_size).astype(int)

            voxel_dict = {}
            for point, idx in zip(self.voxels, indices):
                idx_key = (int(idx[0]), int(idx[1]), int(idx[2]))
                label_val = int(point[3]) if point.shape[0] > 3 else 0
                if idx_key in voxel_dict:
                    voxel_dict[idx_key] = min(voxel_dict[idx_key], label_val)
                else:
                    voxel_dict[idx_key] = label_val

            final_voxels = []
            for idx_key, label_val in voxel_dict.items():
                cx = origin_coords[0] + (idx_key[0] + 0.5) * voxel_size
                cy = origin_coords[1] + (idx_key[1] + 0.5) * voxel_size
                cz = origin_coords[2] + (idx_key[2] + 0.5) * voxel_size
                final_voxels.append([cx, cy, cz, float(label_val)])

            self.voxels = np.array(final_voxels, dtype=float)
            new_voxels_mask = (self.voxels[:, 3] == 2.0)
            self.recovered_voxels = self.voxels[new_voxels_mask] if np.any(new_voxels_mask) else np.empty((0, 4))

        print(f"✅ Завершено восстановление симметрии и балансировка.")




    def balance_layer_with_neighbor(self, neighbor, z, z_step=1.0):
        neighbor_layer_mask = (neighbor.voxels[:, 2] >= z) & (neighbor.voxels[:, 2] < z + z_step)
        neighbor_voxels = neighbor.voxels[neighbor_layer_mask]

        if neighbor_voxels.shape[0] == 0:
            return

        own_center = np.array(self.compute_layer_center(z))
        neighbor_center = np.array(neighbor.compute_layer_center(z))

        dists_to_own = np.linalg.norm(neighbor_voxels[:, :2] - own_center, axis=1)
        dists_to_neighbor = np.linalg.norm(neighbor_voxels[:, :2] - neighbor_center, axis=1)

        take_mask = dists_to_own < dists_to_neighbor
        points_to_take = neighbor_voxels[take_mask]

        if points_to_take.shape[0] > 0:
            if points_to_take.shape[1] == 3:
                points_to_take = np.column_stack([points_to_take, np.ones(points_to_take.shape[0])])
            else:
                points_to_take[:, 3] = 1

            if self.voxels.shape[1] == 3:
                self.voxels = np.column_stack([self.voxels, np.zeros(self.voxels.shape[0])])
            if neighbor.voxels.shape[1] == 3:
                neighbor.voxels = np.column_stack([neighbor.voxels, np.zeros(neighbor.voxels.shape[0])])

            global_indices = np.where(neighbor_layer_mask)[0][take_mask]
            neighbor.voxels = np.delete(neighbor.voxels, global_indices, axis=0)
            self.voxels = np.vstack([self.voxels, points_to_take])

            print(f"🧲 Агрессивно забрано {points_to_take.shape[0]} точек у соседа на слое z={z:.2f}")
        else:
            print(f"➖ Нет точек для захвата на слое z={z:.2f}")



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

def write_symmetry_to_file(tree_id, symmetry_scores, output_dir="symmetry_logs"):
    """
    Сохраняет послойные коэффициенты симметрии в CSV-файл для дерева с данным ID.
    Каждый вызов добавляет новую колонку в файл, без перезаписи предыдущих данных.
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"symmetry_tree_{tree_id}.csv")

    # Загружаем существующие строки, если файл есть
    existing_rows = []
    if os.path.exists(filename):
        with open(filename, 'r', newline='') as f:
            reader = csv.reader(f)
            existing_rows = list(reader)

    # Убедимся, что достаточно строк (по числу слоёв)
    max_len = max(len(existing_rows), len(symmetry_scores))
    while len(existing_rows) < max_len:
        existing_rows.append([])

    # Добавим новые значения послойно
    for i, score in enumerate(symmetry_scores):
        existing_rows[i].append(f"{score:.3f}" if score is not None else "")


    # Запишем обратно в файл
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        print(f"📁 Запись в файл {filename}")
        print("📏 Кол-во строк:", len(existing_rows))
        print("🧩 Новые значения:", symmetry_scores)

        writer.writerows(existing_rows)


# from scipy.spatial import KDTree

# def get_local_density(points, radius=0.5):
#     """Возвращает массив плотностей точек."""
#     if len(points) == 0:
#         return np.zeros(0)
#     tree = KDTree(points[:, :2])
#     densities = np.array([len(tree.query_ball_point(p[:2], radius)) for p in points])
#     return densities

# def should_transfer(dist_self, dist_neighbor, voxel_size):
#     """Решает, стоит ли передавать точку (если она на границе деревьев)."""
#     diff = abs(dist_self - dist_neighbor)
#     if diff < voxel_size * 1.5:  # Пограничная зона
#         return np.random.rand() < 0.7  # С вероятностью 70% передаем
#     return dist_neighbor < dist_self  # Если сосед явно ближе

# def check_neighborhood(tree, point, neighbor_tree, radius=0.3):
#     """Проверяет, окружена ли точка точками своего или соседнего дерева."""
#     tree_kdtree = KDTree(tree.voxels[:, :2])
#     neighbor_kdtree = KDTree(neighbor_tree.voxels[:, :2])

#     tree_neighbors = len(tree_kdtree.query_ball_point(point[:2], radius))
#     neighbor_neighbors = len(neighbor_kdtree.query_ball_point(point[:2], radius))

#     return tree_neighbors >= neighbor_neighbors  # Если у своего дерева больше соседей, оставляем точку
