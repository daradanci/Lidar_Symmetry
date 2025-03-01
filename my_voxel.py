import os
import pandas as pd
import open3d as o3d
import numpy as np
import matplotlib.cm as cm
import imageio
from PIL import Image, ImageDraw, ImageFont

from classes.PCD_TREE import PCD_TREE


def visualize_tree(tree, point_size=5.0, transform="xy_to_xz"):
    """
    Визуализирует одно дерево (PCD_TREE) с Open3D, окрашивая точки по высоте (Z) с чёрным фоном.
    Перед визуализацией выполняется переориентация облака точек.

    :param tree: объект PCD_TREE
    :param point_size: размер точек
    :param transform: тип трансформации перед визуализацией ("xy_to_xz", "xy_to_yz", "flip_xy", "flip_xz", "flip_yz")
    """
    # Проверка наличия точек для визуализации
    if tree.points is None or tree.points.shape[0] == 0:
        print("⚠ Ошибка: У дерева нет точек для визуализации.")
        return

    # Проверка наличия координат ствола
    if tree.trunk_x is None or tree.trunk_y is None:
        print("⚠ Ошибка: Не заданы координаты ствола для визуализации.")
        return

    # Создание окна визуализации
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Копирование точек перед трансформацией
    pcd = o3d.geometry.PointCloud()
    points = tree.points.copy()

    # ✅ Отметка центра ствола (будет отображена как красная точка)
    trunk_center = np.array([[tree.trunk_x, tree.trunk_y, np.min(points[:, 2])]])

    # Применение трансформации перед визуализацией
    if transform == "xy_to_xz":
        points = points[:, [0, 2, 1]]    # Замена Y ↔ Z
        trunk_center = trunk_center[:, [0, 2, 1]]
    elif transform == "xy_to_yz":
        points = points[:, [2, 1, 0]]    # Замена X ↔ Z
        trunk_center = trunk_center[:, [2, 1, 0]]
    elif transform == "flip_xy":
        points[:, :2] *= -1              # Отражение по XY
        trunk_center[:, :2] *= -1
    elif transform == "flip_xz":
        points[:, [0, 2]] *= -1          # Отражение по XZ
        trunk_center[:, [0, 2]] *= -1
    elif transform == "flip_yz":
        points[:, [1, 2]] *= -1          # Отражение по YZ
        trunk_center[:, [1, 2]] *= -1

    # Обновление точек в Open3D
    pcd.points = o3d.utility.Vector3dVector(points)

    # Окрашивание точек по высоте (Z)
    z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])
    norm_z = (points[:, 2] - z_min) / (z_max - z_min)  # Нормализация в диапазоне [0, 1]
    colormap = cm.get_cmap("plasma")
    colors = colormap(norm_z)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Добавление облака точек дерева в визуализацию
    vis.add_geometry(pcd)

    # ✅ Добавление точки ствола (красная точка)
    trunk_pcd = o3d.geometry.PointCloud()
    trunk_pcd.points = o3d.utility.Vector3dVector(trunk_center)
    trunk_pcd.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # Красный цвет
    vis.add_geometry(trunk_pcd)

    # Настройка чёрного фона и размеров точек
    render_opt = vis.get_render_option()
    render_opt.background_color = np.array([0, 0, 0])
    render_opt.point_size = point_size

    # Вывод коэффициента симметрии в консоль
    print(f"Tree | Symmetry Score: {tree.symmetry_score:.2f}")

    # Запуск визуализации
    vis.run()
    vis.destroy_window()



def create_tree_rotation_gif(tree, tree_id, gif_path="tree_rotation.gif", point_size=5.0, num_frames=36, transform="xy_to_xz", z_step=1.0):
    """
    Создаёт GIF-анимацию вращения дерева (PCD_TREE) вокруг оси Y, окрашивая слои чередующимися цветами и отображая центр ствола.

    :param tree: объект PCD_TREE
    :param tree_id: ID дерева (будет отображаться на GIF)
    :param gif_path: путь для сохранения GIF
    :param point_size: размер точек
    :param num_frames: количество кадров (чем больше, тем плавнее вращение)
    :param transform: тип трансформации перед визуализацией ("xy_to_xz", "xy_to_yz", "flip_xy", "flip_xz", "flip_yz")
    :param z_step: Высота слоя (по умолчанию 1.0 м)
    """
    # Получаем активное представление точек (воксели или исходные точки)
    points = tree.get_active_points()

    # Добавляем приписку _vox, если используется вокселизация
    if tree.voxels is not None:
        gif_path = gif_path.replace(".gif", "_vox.gif")

    # Проверка наличия точек
    if points is None or points.shape[0] == 0:
        print(f"⚠ Ошибка: У дерева {tree_id} нет точек для визуализации.")
        return

    # Проверка наличия координат ствола
    if tree.trunk_x is None or tree.trunk_y is None:
        print(f"⚠ Ошибка: Не заданы координаты ствола для дерева {tree_id}.")
        return

    # Настройка визуализации в Open3D
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)

    # Подготовка облака точек
    pcd = o3d.geometry.PointCloud()

    # Разделение дерева на слои по высоте
    z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])
    z_levels = np.arange(z_min, z_max, z_step)
    trunk_x, trunk_y = tree.trunk_x, tree.trunk_y

    # Две чередующиеся палитры
    colormap_pair = ["plasma", "viridis"]

    # Создание массива цветов
    colors = np.zeros((points.shape[0], 3))

    # Назначение цвета каждому слою
    for i, z in enumerate(z_levels):
        idx = np.where((points[:, 2] >= z) & (points[:, 2] < z + z_step))
        if len(idx[0]) == 0:
            continue

        # Расстояния от ствола до точек слоя
        distances = np.linalg.norm(points[idx][:, :2] - np.array([trunk_x, trunk_y]), axis=1)
        norm_distances = (distances - np.min(distances)) / (np.max(distances) - np.min(distances) + 1e-6)

        # Чередование цветовой палитры
        colormap = cm.get_cmap(colormap_pair[i % 2])
        colors[idx] = colormap(norm_distances)[:, :3]

    # Применение трансформации координат
    trunk_center = np.array([[trunk_x, trunk_y, np.min(points[:, 2])]])
    if transform == "xy_to_xz":
        points = points[:, [0, 2, 1]]
        trunk_center = trunk_center[:, [0, 2, 1]]
    elif transform == "xy_to_yz":
        points = points[:, [2, 1, 0]]
        trunk_center = trunk_center[:, [2, 1, 0]]
    elif transform == "flip_xy":
        points[:, :2] *= -1
        trunk_center[:, :2] *= -1
    elif transform == "flip_xz":
        points[:, [0, 2]] *= -1
        trunk_center[:, [0, 2]] *= -1
    elif transform == "flip_yz":
        points[:, [1, 2]] *= -1
        trunk_center[:, [1, 2]] *= -1

    # Обновление точек и цветов
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(pcd)

    # ✅ Отображение центра ствола (красная точка)
    trunk_pcd = o3d.geometry.PointCloud()
    trunk_pcd.points = o3d.utility.Vector3dVector(trunk_center)
    trunk_pcd.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # Красный цвет
    vis.add_geometry(trunk_pcd)

    # Настройка визуализации
    render_opt = vis.get_render_option()
    render_opt.background_color = np.array([0, 0, 0])
    render_opt.point_size = point_size

    # Настройка камеры
    center = np.mean(points, axis=0)
    view_control = vis.get_view_control()
    view_control.set_lookat(center.tolist())
    view_control.set_front([0, 0, -1])
    view_control.set_up([0, 1, 0])
    view_control.set_zoom(0.8)

    # Коэффициенты симметрии
    symmetry_score = getattr(tree, "symmetry_score", 0)
    symmetry_scores_per_layer = getattr(tree, "symmetry_scores_per_layer", [])
    symmetry_list_str = "\n".join([f"{s:.2f}" for s in symmetry_scores_per_layer])
    if len(symmetry_scores_per_layer) > 16:
        symmetry_list_str = "\n".join([f"{s:.2f}" for s in symmetry_scores_per_layer[:16]]) + "\n..."

    # Папка для временных изображений
    temp_dir = "temp_gif_frames"
    os.makedirs(temp_dir, exist_ok=True)

    frames = []

    # Вращение и создание кадров
    for i in range(num_frames):
        angle = (i / num_frames) * 360
        theta = np.radians(angle)
        rotation_matrix = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])

        # Вращение точек вокруг центра
        rotated_points = (points - center) @ rotation_matrix.T + center
        pcd.points = o3d.utility.Vector3dVector(rotated_points)

        # Вращение точки ствола
        rotated_trunk = (trunk_center - center) @ rotation_matrix.T + center
        trunk_pcd.points = o3d.utility.Vector3dVector(rotated_trunk)

        vis.update_geometry(pcd)
        vis.update_geometry(trunk_pcd)
        vis.poll_events()
        vis.update_renderer()

        # Сохранение кадра
        img_path = os.path.join(temp_dir, f"frame_{i:03d}.png")
        vis.capture_screen_image(img_path)

        # Добавление текста с помощью PIL
        img = Image.open(img_path)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()

        text = f"Tree ID: {tree_id}\nSymmetry: {symmetry_score:.2f}\n\nSymmetry Per Layer:\n{symmetry_list_str}"
        draw.text((10, 50), text, fill=(255, 255, 255), font=font)
        img.save(img_path)
        frames.append(img_path)

    vis.destroy_window()

    # Создание GIF
    images = [imageio.imread(frame) for frame in frames]
    imageio.mimsave(gif_path, images, duration=0.1)

    # Удаление временных изображений
    for frame in frames:
        os.remove(frame)
    os.rmdir(temp_dir)

    print(f"✅ GIF сохранён: {gif_path}")

def create_tree_rotation_gif_mono(tree, tree_id, gif_path="tree_rotation_mono.gif", point_size=5.0, num_frames=36, transform="xy_to_xz", z_step=1.0):
    """
    Создаёт GIF-анимацию вращения дерева (PCD_TREE) вокруг оси Y, окрашивая слои градиентом от ствола к краям и отображая центр ствола.

    :param tree: объект PCD_TREE
    :param tree_id: ID дерева (будет отображаться на GIF)
    :param gif_path: путь для сохранения GIF
    :param point_size: размер точек
    :param num_frames: количество кадров (чем больше, тем плавнее вращение)
    :param transform: тип трансформации перед визуализацией ("xy_to_xz", "xy_to_yz", "flip_xy", "flip_xz", "flip_yz")
    :param z_step: Высота слоя (по умолчанию 1.0 м)
    """
    # Получаем активное представление точек (воксели или исходные точки)
    points = tree.get_active_points()

    # Добавляем приписку _vox, если используется вокселизация
    if tree.voxels is not None:
        gif_path = gif_path.replace(".gif", "_vox.gif")

    # Проверка наличия точек
    if points is None or points.shape[0] == 0:
        print(f"⚠ Ошибка: У дерева {tree_id} нет точек для визуализации.")
        return

    # Проверка наличия координат ствола
    if tree.trunk_x is None or tree.trunk_y is None:
        print(f"⚠ Ошибка: Не заданы координаты ствола для дерева {tree_id}.")
        return

    # Настройка визуализации в Open3D
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)

    # Подготовка облака точек
    pcd = o3d.geometry.PointCloud()

    # Разделение дерева на слои по высоте
    z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])
    z_levels = np.arange(z_min, z_max, z_step)
    trunk_x, trunk_y = tree.trunk_x, tree.trunk_y

    # Создание массива цветов
    colors = np.zeros((points.shape[0], 3))

    # Назначение цвета каждому слою (однотонная палитра plasma)
    colormap = cm.get_cmap("plasma")
    for i, z in enumerate(z_levels):
        idx = np.where((points[:, 2] >= z) & (points[:, 2] < z + z_step))
        if len(idx[0]) == 0:
            continue

        # Расстояния от ствола до точек слоя
        distances = np.linalg.norm(points[idx][:, :2] - np.array([trunk_x, trunk_y]), axis=1)
        norm_distances = (distances - np.min(distances)) / (np.max(distances) - np.min(distances) + 1e-6)

        # Градиентный цвет от центра к краям слоя
        colors[idx] = colormap(norm_distances)[:, :3]

    # Применение трансформации координат
    trunk_center = np.array([[trunk_x, trunk_y, np.min(points[:, 2])]])
    if transform == "xy_to_xz":
        points = points[:, [0, 2, 1]]
        trunk_center = trunk_center[:, [0, 2, 1]]
    elif transform == "xy_to_yz":
        points = points[:, [2, 1, 0]]
        trunk_center = trunk_center[:, [2, 1, 0]]
    elif transform == "flip_xy":
        points[:, :2] *= -1
        trunk_center[:, :2] *= -1
    elif transform == "flip_xz":
        points[:, [0, 2]] *= -1
        trunk_center[:, [0, 2]] *= -1
    elif transform == "flip_yz":
        points[:, [1, 2]] *= -1
        trunk_center[:, [1, 2]] *= -1

    # Обновление точек и цветов
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(pcd)

    # ✅ Отображение центра ствола (красная точка)
    trunk_pcd = o3d.geometry.PointCloud()
    trunk_pcd.points = o3d.utility.Vector3dVector(trunk_center)
    trunk_pcd.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # Красный цвет
    vis.add_geometry(trunk_pcd)

    # Настройка визуализации
    render_opt = vis.get_render_option()
    render_opt.background_color = np.array([0, 0, 0])
    render_opt.point_size = point_size

    # Настройка камеры
    center = np.mean(points, axis=0)
    view_control = vis.get_view_control()
    view_control.set_lookat(center.tolist())
    view_control.set_front([0, 0, -1])
    view_control.set_up([0, 1, 0])
    view_control.set_zoom(0.8)

    # Коэффициенты симметрии
    symmetry_score = getattr(tree, "symmetry_score", 0)
    symmetry_scores_per_layer = getattr(tree, "symmetry_scores_per_layer", [])
    symmetry_list_str = "\n".join([f"{s:.2f}" for s in symmetry_scores_per_layer])
    if len(symmetry_scores_per_layer) > 16:
        symmetry_list_str = "\n".join([f"{s:.2f}" for s in symmetry_scores_per_layer[:16]]) + "\n..."

    # Папка для временных изображений
    temp_dir = "temp_gif_frames_mono"
    os.makedirs(temp_dir, exist_ok=True)

    frames = []

    # Вращение и создание кадров
    for i in range(num_frames):
        angle = (i / num_frames) * 360
        theta = np.radians(angle)
        rotation_matrix = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])

        # Вращение точек вокруг центра
        rotated_points = (points - center) @ rotation_matrix.T + center
        pcd.points = o3d.utility.Vector3dVector(rotated_points)

        # Вращение точки ствола
        rotated_trunk = (trunk_center - center) @ rotation_matrix.T + center
        trunk_pcd.points = o3d.utility.Vector3dVector(rotated_trunk)

        vis.update_geometry(pcd)
        vis.update_geometry(trunk_pcd)
        vis.poll_events()
        vis.update_renderer()

        # Сохранение кадра
        img_path = os.path.join(temp_dir, f"frame_{i:03d}.png")
        vis.capture_screen_image(img_path)

        # Добавление текста с помощью PIL
        img = Image.open(img_path)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()

        text = f"Tree ID: {tree_id}\nSymmetry: {symmetry_score:.2f}\n\nSymmetry Per Layer:\n{symmetry_list_str}"
        draw.text((10, 50), text, fill=(255, 255, 255), font=font)
        img.save(img_path)
        frames.append(img_path)

    vis.destroy_window()

    # Создание GIF
    images = [imageio.imread(frame) for frame in frames]
    imageio.mimsave(gif_path, images, duration=0.1)

    # Удаление временных изображений
    for frame in frames:
        os.remove(frame)
    os.rmdir(temp_dir)

    print(f"✅ GIF сохранён: {gif_path}")


def process_trees_in_range(
    start_id, end_id, data_folder, gif_folder, transform="xy_to_xz", 
    z_threshold=0.1, z_step=1.0, voxelize=True, voxel_size=0.1
):
    """
    Выполняется обработка деревьев в указанном диапазоне:
    - Загрузка облака точек дерева.
    - Определение центра ствола по первому слою.
    - (Опционально) Вокселизация точек.
    - Вычисление симметрии дерева.
    - Создание GIF-анимаций вращающегося дерева.

    :param start_id: начальный ID дерева (например, 1)
    :param end_id: конечный ID дерева (например, 10)
    :param data_folder: путь к папке с файлами деревьев
    :param gif_folder: путь к папке для сохранения GIF-анимаций
    :param transform: метод трансформации перед визуализацией ("xy_to_xz", "xy_to_yz" и т. д.)
    :param z_threshold: максимальная высота слоя для определения центра ствола
    :param z_step: высота слоя для расчёта симметрии и визуализации
    :param voxelize: включить ли вокселизацию (по умолчанию True)
    :param voxel_size: размер вокселя (по умолчанию 0.1)
    """
    os.makedirs(gif_folder, exist_ok=True)  # Создание папки для GIF, если она отсутствует

    print(f"🚀 Обработка деревьев с {start_id} по {end_id}...")

    for i in range(start_id, end_id + 1):
        file_name = f"tree_{i:04d}.pcd"  # Формат файла: tree_0001.pcd
        file_path = os.path.join(data_folder, file_name)

        # Проверка наличия файла
        if not os.path.exists(file_path):
            print(f"⚠ Файл {file_path} не найден, пропуск...")
            continue

        # Создание объекта дерева и загрузка точек
        tree = PCD_TREE()
        tree.open(file_path, verbose=True)
        tree.file_path = file_path

        # Проверка наличия точек
        if tree.points is None or tree.points.shape[0] == 0:
            print(f"⚠ Облако точек дерева {i} пустое, пропуск...")
            continue

        # Определение центра ствола на первом слое
        tree.set_trunk_center(z_threshold=z_threshold, min_points=10)
        if tree.trunk_x is None or tree.trunk_y is None:
            print(f"⚠ Не удалось определить центр ствола для дерева {i}, пропуск...")
            continue

        # (Опционально) Выполнение вокселизации
        if voxelize:
            tree.voxelize_tree(voxel_size=voxel_size)
            print(f"✅ Дерево {i} вокселизировано.")

        # Вычисление коэффициента симметрии
        symmetry_score = tree.measure_tree_symmetry(z_step=z_step)
        print(f"✅ Tree {i} | Symmetry Score: {symmetry_score:.2f}")

        # ✅ Создание GIF-анимаций с отображением центра ствола
        gif_path = os.path.join(gif_folder, f"tree_rotation_{i:04d}.gif")
        create_tree_rotation_gif(tree, tree_id=i, gif_path=gif_path, transform=transform, z_step=z_step)

        gif_path_mono = os.path.join(gif_folder, f"tree_rotation_mono_{i:04d}.gif")
        create_tree_rotation_gif_mono(tree, tree_id=i, gif_path=gif_path_mono, transform=transform, z_step=z_step)

    print("🎉 Обработка завершена!")

# Вызов с опцией вокселизации
process_trees_in_range(
    start_id=98, 
    end_id=99, 
    data_folder="D:\\data\\symmetry", 
    gif_folder="D:\\data\\gifs\\symmetry", 
    transform="xy_to_xz", 
    z_threshold=0.1, 
    z_step=1.0, 
    voxelize=True,  # Включение вокселизации
    voxel_size=0.25  # Размер вокселя
)



# from classes.VOXEL_TREE import VOXEL_TREE

# def process_voxel_trees_in_range(start_id, end_id, data_folder, voxel_size=0.1, z_step=1.0):
#     """
#     Обрабатывает деревья в указанном диапазоне, выполняет вокселизацию и вычисляет симметрию для каждого слоя.

#     :param start_id: начальный ID дерева.
#     :param end_id: конечный ID дерева.
#     :param data_folder: путь к папке с файлами деревьев.
#     :param voxel_size: размер вокселя.
#     :param z_step: высота слоя для анализа.
#     """
#     print(f"Обработка воксельных деревьев с {start_id} по {end_id}...")

#     for i in range(start_id, end_id + 1):
#         file_name = f"tree_{i:04d}.pcd"  # Пример имени файла: tree_0001.pcd
#         file_path = os.path.join(data_folder, file_name)

#         if not os.path.exists(file_path):
#             print(f"Файл {file_path} не найден, пропуск...")
#             continue

#         # Загрузка точек из файла
#         points = load_points_from_pcd(file_path)  # Функция для загрузки точек
#         if points is None:
#             print(f"⚠ Ошибка при загрузке файла: {file_path}")
#             continue

#         # Создание объекта VOXEL_TREE и вокселизация
#         voxel_tree = VOXEL_TREE(points=points, voxel_size=voxel_size)
#         voxel_tree.voxelize()

#         # Разделение на слои и вычисление симметрии
#         voxel_tree.calculate_symmetry(z_step=z_step)

#         # Визуализация вокселей
#         voxel_tree.visualize_voxels()
#         print(f"✅ Дерево {i} обработано.\n")

# def load_points_from_pcd(file_path):
#     """
#     Загружает точки из PCD-файла с использованием Open3D.

#     :param file_path: путь к PCD-файлу.
#     :return: массив координат точек или None в случае ошибки.
#     """
#     try:
#         pcd = o3d.io.read_point_cloud(file_path)
#         points = np.asarray(pcd.points)
#         return points
#     except Exception as e:
#         print(f"⚠ Ошибка при чтении файла {file_path}: {str(e)}")
#         return None


# # Запуск обработки воксельных деревьев
# process_voxel_trees_in_range(
#     start_id=99, 
#     end_id=99, 
#     data_folder="D:\\data\\symmetry", 
#     voxel_size=0.1, 
#     z_step=1.0,
# )



# def process_trees_in_range1(start_id, end_id, data_folder, gif_folder, transform="xy_to_xz", z_threshold=0.1, z_step=1.0, voxel_size=0.1):
#     import os
#     import numpy as np

#     os.makedirs(gif_folder, exist_ok=True)
#     print(f"🚀 Обработка деревьев с {start_id} по {end_id}...")

#     for i in range(start_id, end_id + 1):
#         file_name = f"tree_{i:04d}.pcd"
#         file_path = os.path.join(data_folder, file_name)

#         if not os.path.exists(file_path):
#             print(f"⚠ Файл {file_path} не найден, пропуск...")
#             continue

#         tree = PCD_TREE()
#         tree.open(file_path, verbose=True)
#         tree.file_path = file_path

#         if tree.points is None or tree.points.shape[0] == 0:
#             print(f"⚠ Облако точек дерева {i} пустое, пропуск...")
#             continue

#         tree.set_trunk_center(z_threshold=z_threshold, min_points=10)
#         symmetry_score = tree.measure_tree_symmetry(z_step=z_step)
#         print(f"✅ Tree {i} | Symmetry Score: {symmetry_score:.2f}")

#         gif_path = os.path.join(gif_folder, f"tree_rotation_{i:04d}.gif")
#         create_tree_rotation_gif(tree, tree_id=i, gif_path=gif_path, transform=transform, z_step=z_step)

#         gif_path_mono = os.path.join(gif_folder, f"tree_rotation_mono_{i:04d}.gif")
#         create_tree_rotation_gif_mono(tree, tree_id=i, gif_path=gif_path_mono, transform=transform, z_step=z_step)

#         voxel_tree = VOXEL_TREE(points=tree.points, voxel_size=voxel_size)
#         voxel_tree.voxelize()
#         voxel_tree.calculate_symmetry(z_step=z_step)
#         print(f"✅ Tree {i} (Voxels) | Symmetry Score: {np.mean(voxel_tree.symmetry_scores_per_layer):.2f}")

#         voxel_gif_path = os.path.join(gif_folder, f"voxel_tree_rotation_{i:04d}.gif")
#         voxel_tree.visualize_voxels(z_step=z_step, point_size=5.0)

#     print("🎉 Обработка завершена!")


# # ✅ Запуск обработки деревьев с включением вокселей
# process_trees_in_range1(
#     start_id=29,
#     end_id=29,
#     data_folder="D:\\data\\orion_exp3\\trees",
#     gif_folder="D:\\data\\gifs\\symmetry",
#     transform="xy_to_xz",
#     z_threshold=0.1,
#     z_step=1.0,
#     voxel_size=0.1
# )

