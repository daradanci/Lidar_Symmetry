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
    if tree.points is None or tree.points.shape[0] == 0:
        print("⚠ Ошибка: У дерева нет точек для визуализации.")
        return

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Создаётся объект Open3D для облака точек
    pcd = o3d.geometry.PointCloud()
    points = tree.points.copy()  # Копирование точек перед трансформацией

    # Применяется трансформация перед визуализацией
    if transform == "xy_to_xz":
        points = points[:, [0, 2, 1]]  # Замена Y ↔ Z
    elif transform == "xy_to_yz":
        points = points[:, [2, 1, 0]]  # Замена X ↔ Z
    elif transform == "flip_xy":
        points[:, :2] *= -1  # Отражение по XY
    elif transform == "flip_xz":
        points[:, [0, 2]] *= -1  # Отражение по XZ
    elif transform == "flip_yz":
        points[:, [1, 2]] *= -1  # Отражение по YZ

    # Обновление точек в Open3D
    pcd.points = o3d.utility.Vector3dVector(points)

    # Определяется градиент цвета в зависимости от высоты (координаты Z)
    z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])
    norm_z = (points[:, 2] - z_min) / (z_max - z_min)  # Нормализация Z в диапазоне [0, 1]
    
    # Используется цветовая палитра "plasma" (фиолетовый → голубой → жёлтый → красный)
    colormap = cm.get_cmap("plasma")  
    colors = colormap(norm_z)[:, :3]  # Извлечение RGB-канала без альфа-компонента
    
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Добавление дерева в сцену
    vis.add_geometry(pcd)

    # Настраивается чёрный фон
    render_opt = vis.get_render_option()
    render_opt.background_color = np.array([0, 0, 0])  # Чёрный цвет
    render_opt.point_size = point_size  # Размер точек

    # Выводится коэффициент симметрии в консоль
    print(f"Tree | Symmetry Score: {tree.symmetry_score:.2f}")

    # Запуск визуализации
    vis.run()
    vis.destroy_window()





def create_tree_rotation_gif_mono(tree, tree_id, gif_path="tree_rotation_mono.gif", point_size=5.0, num_frames=36, transform="xy_to_xz", z_step=1.0):
    """
    Создаёт GIF-анимацию вращения дерева (PCD_TREE) вокруг оси Y, окрашивая слои градиентом от ствола к краям.

    :param tree: объект PCD_TREE
    :param tree_id: ID дерева (будет отображаться на GIF)
    :param gif_path: путь для сохранения GIF
    :param point_size: размер точек
    :param num_frames: количество кадров (чем больше, тем плавнее вращение)
    :param transform: тип трансформации перед визуализацией ("xy_to_xz", "xy_to_yz", "flip_xy", "flip_xz", "flip_yz")
    :param z_step: Высота слоя (по умолчанию 1.0 м)
    """
    if tree.points is None or tree.points.shape[0] == 0:
        print(f"⚠ Ошибка: У дерева {tree_id} нет точек для визуализации.")
        return

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)

    # Создаётся объект Open3D для облака точек
    pcd = o3d.geometry.PointCloud()
    points = tree.points.copy()  

    # Разделение дерева на слои по высоте
    z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])
    z_levels = np.arange(z_min, z_max, z_step)

    # Определение координат ствола
    trunk_x, trunk_y = tree.trunk_x, tree.trunk_y

    # Создаётся массив цветов
    colors = np.zeros((points.shape[0], 3))  

    # Назначение цвета каждому слою
    colormap = cm.get_cmap("plasma")  
    for i, z in enumerate(z_levels):
        idx = np.where((points[:, 2] >= z) & (points[:, 2] < z + z_step))
        if len(idx[0]) == 0:
            continue

        # Определение расстояний от ствола до точек слоя
        distances = np.linalg.norm(points[idx][:, :2] - np.array([trunk_x, trunk_y]), axis=1)
        norm_distances = (distances - np.min(distances)) / (np.max(distances) - np.min(distances) + 1e-6)

        # Использование нормированных расстояний для градиентного окрашивания
        colors[idx] = colormap(norm_distances)[:, :3]

    # Применение трансформации перед визуализацией
    if transform == "xy_to_xz":
        points = points[:, [0, 2, 1]]
    elif transform == "xy_to_yz":
        points = points[:, [2, 1, 0]]
    elif transform == "flip_xy":
        points[:, :2] *= -1
    elif transform == "flip_xz":
        points[:, [0, 2]] *= -1
    elif transform == "flip_yz":
        points[:, [1, 2]] *= -1

    # Обновление точек и цветов в Open3D
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(pcd)

    # Настройка фона
    render_opt = vis.get_render_option()
    render_opt.background_color = np.array([0, 0, 0])  
    render_opt.point_size = point_size  

    # Определение центра дерева и настройка камеры
    center = np.mean(points, axis=0)
    view_control = vis.get_view_control()
    view_control.set_lookat(center.tolist())  
    view_control.set_front([0, 0, -1])  
    view_control.set_up([0, 1, 0])  
    view_control.set_zoom(0.8)  

    # Получение коэффициентов симметрии
    symmetry_score = getattr(tree, "symmetry_score", 0)
    symmetry_scores_per_layer = getattr(tree, "symmetry_scores_per_layer", [])

    # Преобразование массива коэффициентов в строку
    symmetry_list_str = "\n".join([f"{s:.2f}" for s in symmetry_scores_per_layer])

    # Ограничение количества строк (при большом числе слоёв)
    max_lines = 16
    if len(symmetry_scores_per_layer) > max_lines:
        symmetry_list_str = "\n".join([f"{s:.2f}" for s in symmetry_scores_per_layer[:max_lines]]) + "\n..."

    # Создание папки для временных изображений
    temp_dir = "temp_gif_frames_mono"
    os.makedirs(temp_dir, exist_ok=True)

    frames = []

    for i in range(num_frames):
        angle = (i / num_frames) * 360  

        # Матрица поворота вокруг оси Y
        theta = np.radians(angle)
        rotation_matrix = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])

        # Вращение точек вокруг центра дерева
        rotated_points = (points - center) @ rotation_matrix.T + center
        pcd.points = o3d.utility.Vector3dVector(rotated_points)

        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

        # Сохранение кадра
        img_path = os.path.join(temp_dir, f"frame_{i:03d}.png")
        vis.capture_screen_image(img_path)

        # Добавление текста с помощью PIL
        img = Image.open(img_path)
        draw = ImageDraw.Draw(img)

        # Загрузка шрифта
        try:
            font = ImageFont.truetype("arial.ttf", 20)  
        except IOError:
            font = ImageFont.load_default()

        # Формирование текста для гифки
        text = f"Tree ID: {tree_id}\nSymmetry: {symmetry_score:.2f}\n\nSymmetry Per Layer:\n{symmetry_list_str}"
        text_position = (10, 50)  
        draw.text(text_position, text, fill=(255, 255, 255), font=font)  

        img.save(img_path)  
        frames.append(img_path)

    vis.destroy_window()

    # Создание GIF-анимации
    images = [imageio.imread(frame) for frame in frames]
    imageio.mimsave(gif_path, images, duration=0.1)  

    # Удаление временных изображений
    for frame in frames:
        os.remove(frame)
    os.rmdir(temp_dir)

    print(f"✅ GIF сохранён: {gif_path}")






def create_tree_rotation_gif(tree, tree_id, gif_path="tree_rotation.gif", point_size=5.0, num_frames=36, transform="xy_to_xz", z_step=1.0):
    """
    Создаёт GIF-анимацию вращения дерева (PCD_TREE) вокруг оси Y, окрашивая слои чередующимися цветами.

    :param tree: объект PCD_TREE
    :param tree_id: ID дерева (будет отображаться на GIF)
    :param gif_path: путь для сохранения GIF
    :param point_size: размер точек
    :param num_frames: количество кадров (чем больше, тем плавнее вращение)
    :param transform: тип трансформации перед визуализацией ("xy_to_xz", "xy_to_yz", "flip_xy", "flip_xz", "flip_yz")
    :param z_step: Высота слоя (по умолчанию 1.0 м)
    """
    if tree.points is None or tree.points.shape[0] == 0:
        print(f"⚠ Ошибка: У дерева {tree_id} нет точек для визуализации.")
        return

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)

    # Создание объекта Open3D для облака точек
    pcd = o3d.geometry.PointCloud()
    points = tree.points.copy()

    # Разделение дерева на слои по высоте
    z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])
    z_levels = np.arange(z_min, z_max, z_step)

    # Получение координат ствола дерева
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

        # Определение расстояний от ствола до точек слоя
        distances = np.linalg.norm(points[idx][:, :2] - np.array([trunk_x, trunk_y]), axis=1)
        norm_distances = (distances - np.min(distances)) / (np.max(distances) - np.min(distances) + 1e-6)

        # Использование чередующейся цветовой палитры
        colormap = cm.get_cmap(colormap_pair[i % 2])

        # Использование нормированных расстояний для градиентного окрашивания
        colors[idx] = colormap(norm_distances)[:, :3]

    # Применение трансформации перед визуализацией (и к координатам, и к цветам)
    if transform == "xy_to_xz":
        points = points[:, [0, 2, 1]]
    elif transform == "xy_to_yz":
        points = points[:, [2, 1, 0]]
    elif transform == "flip_xy":
        points[:, :2] *= -1
    elif transform == "flip_xz":
        points[:, [0, 2]] *= -1
    elif transform == "flip_yz":
        points[:, [1, 2]] *= -1

    # Обновление точек и цветов в Open3D
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(pcd)

    # Настройка фона
    render_opt = vis.get_render_option()
    render_opt.background_color = np.array([0, 0, 0])
    render_opt.point_size = point_size

    # Определение центра дерева и настройка камеры
    center = np.mean(points, axis=0)
    view_control = vis.get_view_control()
    view_control.set_lookat(center.tolist())
    view_control.set_front([0, 0, -1])
    view_control.set_up([0, 1, 0])
    view_control.set_zoom(0.8)

    # Получение коэффициентов симметрии
    symmetry_score = getattr(tree, "symmetry_score", 0)
    symmetry_scores_per_layer = getattr(tree, "symmetry_scores_per_layer", [])

    # Преобразование массива коэффициентов в строку
    symmetry_list_str = "\n".join([f"{s:.2f}" for s in symmetry_scores_per_layer])

    # Ограничение количества строк (при большом числе слоёв)
    max_lines = 16
    if len(symmetry_scores_per_layer) > max_lines:
        symmetry_list_str = "\n".join([f"{s:.2f}" for s in symmetry_scores_per_layer[:max_lines]]) + "\n..."

    # Создание папки для временных изображений
    temp_dir = "temp_gif_frames"
    os.makedirs(temp_dir, exist_ok=True)

    frames = []

    for i in range(num_frames):
        angle = (i / num_frames) * 360

        # Матрица поворота вокруг оси Y
        theta = np.radians(angle)
        rotation_matrix = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])

        # Вращение точек вокруг центра дерева
        rotated_points = (points - center) @ rotation_matrix.T + center
        pcd.points = o3d.utility.Vector3dVector(rotated_points)

        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

        # Сохранение кадра
        img_path = os.path.join(temp_dir, f"frame_{i:03d}.png")
        vis.capture_screen_image(img_path)

        # Добавление текста с помощью PIL
        img = Image.open(img_path)
        draw = ImageDraw.Draw(img)

        # Загрузка шрифта
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()

        # Формирование текста для гифки
        text = f"Tree ID: {tree_id}\nSymmetry: {symmetry_score:.2f}\n\nSymmetry Per Layer:\n{symmetry_list_str}"
        text_position = (10, 50)
        draw.text(text_position, text, fill=(255, 255, 255), font=font)

        img.save(img_path)
        frames.append(img_path)

    vis.destroy_window()

    # Создание GIF-анимации
    images = [imageio.imread(frame) for frame in frames]
    imageio.mimsave(gif_path, images, duration=0.1)

    # Удаление временных изображений
    for frame in frames:
        os.remove(frame)
    os.rmdir(temp_dir)

    print(f"✅ GIF сохранён: {gif_path}")









def process_trees_in_range(start_id, end_id, data_folder, results_csv, gif_folder, transform="xy_to_xz"):
    """
    Выполняется обработка деревьев в указанном диапазоне, загрузка координат ствола из RESULTS.csv,
    вычисление симметрии и создание GIF-анимаций.

    :param start_id: начальный ID дерева (например, 1)
    :param end_id: конечный ID дерева (например, 10)
    :param data_folder: путь к папке с файлами деревьев
    :param results_csv: путь к файлу RESULTS.csv (содержит X_shift и Y_shift)
    :param gif_folder: путь к папке для сохранения GIF-анимаций
    :param transform: метод трансформации перед визуализацией ("xy_to_xz", "xy_to_yz" и т. д.)
    """
    os.makedirs(gif_folder, exist_ok=True)  # Создание папки для GIF, если она отсутствует

    # Чтение данных из RESULTS.csv
    df_results = pd.read_csv(results_csv, delimiter=";")

    print(f"Обработка деревьев с {start_id} по {end_id}...")

    for i in range(start_id, end_id + 1):
        file_name = f"tree_{i:04d}.pcd"  # Формат файла: tree_0001.pcd
        file_path = os.path.join(data_folder, file_name)

        if not os.path.exists(file_path):
            print(f"Файл {file_path} не найден, пропуск...")
            continue

        # Определение индекса строки в RESULTS.csv (нумерация начинается с 0, поэтому i-1)
        row_index = i - 1  

        if row_index >= len(df_results):
            print(f"Нет данных для дерева {i} в RESULTS.csv, пропуск...")
            continue

        # Получение координат ствола из таблицы
        trunk_x = float(df_results.iloc[row_index]["X_shift"].replace(",", "."))
        trunk_y = float(df_results.iloc[row_index]["Y_shift"].replace(",", "."))

        # Создание объекта дерева с координатами ствола
        tree = PCD_TREE(trunk_x=trunk_x, trunk_y=trunk_y)
        tree.open(file_path, verbose=True)
        tree.file_path = file_path  # Добавление пути к файлу в объект

        # Вычисление коэффициента симметрии
        symmetry_score = tree.measure_tree_symmetry()
        print(f"Tree {i} | Symmetry Score: {symmetry_score:.2f}")

        # Создание GIF-анимаций
        gif_path = os.path.join(gif_folder, f"tree_rotation_{i:04d}.gif")
        create_tree_rotation_gif(tree, tree_id=i, gif_path=gif_path, transform=transform, z_step=1.0)

        gif_path_mono = os.path.join(gif_folder, f"tree_rotation_mono_{i:04d}.gif")
        create_tree_rotation_gif_mono(tree, tree_id=i, gif_path=gif_path_mono, transform=transform, z_step=1.0)

    print("Обработка завершена!")

process_trees_in_range(
    start_id=21,  # Начальный ID дерева
    end_id=156,   # Конечный ID дерева
    data_folder="D:\\data\\orion_exp3\\trees",  # Папка с деревьями
    results_csv="D:\\data\\orion_exp3\\RESULTS.csv",  # Файл с координатами стволов
    gif_folder="D:\\data\\gifs",  # Папка для GIF-анимаций
    transform="xy_to_xz"  # Трансформация перед визуализацией
)
