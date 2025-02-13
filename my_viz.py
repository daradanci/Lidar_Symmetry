import os
import time
import sys
import tracemalloc
import laspy
import numpy as np
import open3d as o3d
from scipy.interpolate import griddata
from tkinter import Tk, filedialog, Toplevel, Button, Label, ttk
from pykrige.ok import OrdinaryKriging
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.spatial import cKDTree
from scipy.interpolate import griddata

def select_file():
    Tk().withdraw()  # Скрываем главное окно tkinter
    file_path = filedialog.askopenfilename(
        title="Select a Point Cloud File",
        filetypes=[("LAS files", "*.las"), ("All files", "*.*")]
    )
    if not file_path:
        raise FileNotFoundError("No file selected!")
    return file_path

def load_las_to_open3d(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    print(f"Loading .las file: {file_path}")
    with laspy.open(file_path) as las_file:
        las_data = las_file.read()
        points = np.vstack((las_data.x, las_data.y, las_data.z)).T

        # Создаем объект PointCloud
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        print(f"Loaded {points.shape[0]} points from .las file.")
        return point_cloud, points


def select_two_files():
    """
    Позволяет пользователю выбрать два файла:
    1. Исходный файл (полный набор данных).
    2. Интерполированный файл.
    """
    Tk().withdraw()  # Скрываем главное окно tkinter
    file_paths = filedialog.askopenfilenames(
        title="Select Original and Interpolated Files",
        filetypes=[("LAS files", "*.las"), ("All files", "*.*")]
    )
    if len(file_paths) != 2:
        raise FileNotFoundError("Two files must be selected: original and interpolated!")
    return file_paths


def interpolate_on_grid(points, grid_x, grid_y, method='linear'):
    """
    Интерполирует значения Z на заданной сетке X-Y.

    :param points: numpy.ndarray, исходные точки (Nx3).
    :param grid_x: numpy.ndarray, сетка по X.
    :param grid_y: numpy.ndarray, сетка по Y.
    :param method: str, метод интерполяции ('linear', 'nearest', 'cubic').
    :return: numpy.ndarray, интерполированные значения Z на сетке.
    """
    z_interp = griddata(points[:, :2], points[:, 2], (grid_x, grid_y), method=method)
    return z_interp

def compare_point_clouds(original_points, interpolated_points, grid_resolution=2.0):
    """
    Сравнивает два облака точек: исходное и интерполированное.
    Вычисляет метрики RMSE, MAE, и R^2.
    
    :param original_points: numpy.ndarray, исходные точки (Nx3).
    :param interpolated_points: numpy.ndarray, интерполированные точки (Mx3).
    :param grid_resolution: float, разрешение сетки для интерполяции.
    :return: dict, метрики сравнения.
    """
    # Создаем общую сетку
    x_min = min(original_points[:, 0].min(), interpolated_points[:, 0].min())
    x_max = max(original_points[:, 0].max(), interpolated_points[:, 0].max())
    y_min = min(original_points[:, 1].min(), interpolated_points[:, 1].min())
    y_max = max(original_points[:, 1].max(), interpolated_points[:, 1].max())

    grid_x, grid_y = np.meshgrid(
        np.arange(x_min, x_max, grid_resolution),
        np.arange(y_min, y_max, grid_resolution)
    )

    # Интерполируем оба набора данных на общую сетку
    true_grid_z = interpolate_on_grid(original_points, grid_x, grid_y, method='linear')
    pred_grid_z = interpolate_on_grid(interpolated_points, grid_x, grid_y, method='linear')

    # Убираем NaN для корректного расчета метрик
    mask = ~np.isnan(true_grid_z) & ~np.isnan(pred_grid_z)
    true_z = true_grid_z[mask]
    pred_z = pred_grid_z[mask]

    # Рассчитываем метрики
    metrics = {
        "RMSE": np.sqrt(mean_squared_error(true_z, pred_z)),
        "MAE": mean_absolute_error(true_z, pred_z),
        "R^2": r2_score(true_z, pred_z),
    }

    return metrics

def save_to_las(points, file_path):
    if points.size == 0:
        print("No points to save.")
        return
    
    print(f"Saving points to: {file_path}")
    header = laspy.LasHeader(point_format=3, version="1.2")
    new_las = laspy.LasData(header)
    new_las.x = points[:, 0]
    new_las.y = points[:, 1]
    new_las.z = points[:, 2]
    new_las.write(file_path)
    print(f"Saved {points.shape[0]} points to {file_path}")

def measure_performance(func, *args, **kwargs):
    tracemalloc.start()
    start_time = time.time()
    result = func(*args, **kwargs)
    elapsed_time = time.time() - start_time
    current, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed_time, peak_memory / 1024 / 1024  # Convert to MB

def interpolate_knn(points, grid_x, grid_y):
    return griddata(points[:, :2], points[:, 2], (grid_x, grid_y), method='nearest')

def interpolate_linear(points, grid_x, grid_y):
    return griddata(points[:, :2], points[:, 2], (grid_x, grid_y), method='linear')

def interpolate_spline(points, grid_x, grid_y, block_size=1):
    """
    Выполняет сплайновую интерполяцию методом разбиения сетки на блоки.

    :param points: numpy.ndarray, точки для интерполяции (Nx3).
    :param grid_x: numpy.ndarray, сетка по X.
    :param grid_y: numpy.ndarray, сетка по Y.
    :param block_size: int, размер блока для обработки за один раз.
    :return: numpy.ndarray, интерполированные значения Z на сетке.
    """
    from scipy.interpolate import Rbf
    # Создаем RBF-функцию для интерполяции
    rbf = Rbf(points[:, 0], points[:, 1], points[:, 2], function='thin_plate')

    # Инициализируем массив для результатов интерполяции
    interpolated_z = np.empty(grid_x.shape)
    interpolated_z.fill(np.nan)

    # Разбиваем сетку на блоки
    total_rows, total_cols = grid_x.shape
    for i in range(0, total_rows, block_size):
        for j in range(0, total_cols, block_size):
            # Определяем текущий блок
            row_end = min(i + block_size, total_rows)
            col_end = min(j + block_size, total_cols)

            x_block = grid_x[i:row_end, j:col_end]
            y_block = grid_y[i:row_end, j:col_end]

            # Выполняем интерполяцию для текущего блока
            print(f"Processing block: rows {i}-{row_end}, cols {j}-{col_end}")
            z_block = rbf(x_block.flatten(), y_block.flatten())

            # Заполняем результаты для текущего блока
            interpolated_z[i:row_end, j:col_end] = z_block.reshape(x_block.shape)

    return interpolated_z


def interpolate_kriging(points, grid_x, grid_y):
    kriging = OrdinaryKriging(points[:, 0], points[:, 1], points[:, 2], variogram_model='linear')
    z_interp, _ = kriging.execute("grid", grid_x[0], grid_y[:, 0])
    return z_interp

def interpolate_bayesian(points, grid_x, grid_y):
    # Подготовка данных
    X_train = points[:, :2]
    y_train = points[:, 2]

    # Ядро: Константный коэффициент * радиальная базисная функция
    kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))

    # Создаем регрессор
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-2)

    # Обучение
    gp.fit(X_train, y_train)

    # Предсказание
    X_pred = np.column_stack((grid_x.flatten(), grid_y.flatten()))
    y_pred, sigma = gp.predict(X_pred, return_std=True)

    # Возвращаем интерполированные значения
    return y_pred.reshape(grid_x.shape)

def interpolate_area(filtered_points, resolution=1.0, method='linear', progress_callback=None):
    print(f"Interpolating missing points using {method} method...")

    # Создаем регулярную сетку по X и Y
    x_min, x_max = filtered_points[:, 0].min(), filtered_points[:, 0].max()
    y_min, y_max = filtered_points[:, 1].min(), filtered_points[:, 1].max()
    grid_x, grid_y = np.meshgrid(
        np.arange(x_min, x_max, resolution),
        np.arange(y_min, y_max, resolution)
    )

    # Выбор алгоритма интерполяции
    if method == 'nearest':
        interpolated_z = interpolate_knn(filtered_points, grid_x, grid_y)
    elif method == 'linear':
        interpolated_z = interpolate_linear(filtered_points, grid_x, grid_y)
    elif method == 'spline':
        interpolated_z = interpolate_spline(filtered_points, grid_x, grid_y)
    elif method == 'kriging':
        interpolated_z = interpolate_kriging(filtered_points, grid_x, grid_y)
    elif method == 'bayesian':
        interpolated_z = interpolate_bayesian(filtered_points, grid_x, grid_y)
    else:
        raise ValueError(f"Unsupported interpolation method: {method}")

    # Преобразуем сетку обратно в точки
    print("Converting grid to points...")
    interpolated_points = np.column_stack((
        grid_x.flatten(),
        grid_y.flatten(),
        interpolated_z.flatten()
    ))

    # Удаляем точки с NaN
    print("Removing NaN values...")
    interpolated_points = interpolated_points[~np.isnan(interpolated_points).any(axis=1)]

    if progress_callback:
        progress_callback(100)  # Интерполяция завершена

    print(f"Interpolated {interpolated_points.shape[0]} new points.")
    return interpolated_points

def select_area(pcd, original_points):
    print("Opening point cloud visualization for point selection...")
    
    # Открываем окно для выбора точек
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window("Select Points (X and Y bounds)", width=1920, height=1080)  # Задаем размеры окна

    # Настраиваем окно на полноэкранный режим
    vis.get_render_option().background_color = np.array([0, 0, 0])  # Черный фон (опционально)
    ctr = vis.get_view_control()
    ctr.change_field_of_view(step=90.0)  # Максимальный угол обзора

    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

    selected_indices = vis.get_picked_points()
    if not selected_indices:
        print("No points selected!")
        return None, None
    
    selected_points = np.asarray(pcd.points)[selected_indices]
    print(f"Selected {len(selected_points)} points.")

    x_min, y_min = selected_points[:, 0].min(), selected_points[:, 1].min()
    x_max, y_max = selected_points[:, 0].max(), selected_points[:, 1].max()
    print(f"Selected X bounds: [{x_min}, {x_max}], Y bounds: [{y_min}, {y_max}]")

    filtered_points = original_points[
        (original_points[:, 0] >= x_min) & (original_points[:, 0] <= x_max) &
        (original_points[:, 1] >= y_min) & (original_points[:, 1] <= y_max)
    ]

    print(f"Points inside the selected X-Y area: {filtered_points.shape[0]}.")
    return selected_points, filtered_points


def interpolate_area_action_with_progress(filtered_points, output_directory, gui, method='linear'):
    progress_win = Toplevel(gui)
    progress_win.title("Interpolating Points")
    Label(progress_win, text=f"Interpolating missing points using {method} method, please wait...").pack(pady=10)

    progress_bar = ttk.Progressbar(progress_win, orient="horizontal", length=300, mode="determinate")
    progress_bar.pack(pady=20)
    progress_bar["value"] = 0

    def update_progress(value):
        progress_bar["value"] = value
        progress_win.update_idletasks()

    interpolated_points, elapsed_time, peak_memory = measure_performance(interpolate_area, filtered_points, method=method, progress_callback=update_progress)
    combined_points = np.vstack((filtered_points, interpolated_points))

    original_save_path = os.path.join(output_directory, "selected_area_original.las")
    save_to_las(filtered_points, original_save_path)

    interpolated_save_path = os.path.join(output_directory, f"interpolated_area_{method}.las")
    save_to_las(combined_points, interpolated_save_path)

    Label(progress_win, text=f"Interpolation complete!\nTime: {elapsed_time:.2f}s\nPeak Memory: {peak_memory:.2f} MB\nFiles saved.").pack(pady=10)
    Button(progress_win, text="Close", command=progress_win.destroy).pack(pady=5)


def simulate_missing_on_selected_area(original_points, selected_points, min_radius=10, max_radius=40):
    # Симулирует отсутствующие данные, создавая большие пробелы в местах выбранных пользователем точек.

    # :param original_points: numpy.ndarray, все точки облака (Nx3).
    # :param selected_points: numpy.ndarray, выбранные пользователем точки (Mx3).
    # :param min_radius: float, минимальный радиус области (увеличен).
    # :param max_radius: float, максимальный радиус области (увеличен).
    # :return: numpy.ndarray, облако точек с пропусками.
    remaining_points = original_points.copy()

    for center in selected_points:
        radius_x = np.random.uniform(min_radius, max_radius)
        radius_y = np.random.uniform(min_radius, max_radius)
        radius_z = np.random.uniform(min_radius / 2, max_radius / 2)

        print(f"Creating large missing region: center={center}, radii=({radius_x}, {radius_y}, {radius_z})")

        mask = (
            ((remaining_points[:, 0] - center[0]) / radius_x) ** 2 +
            ((remaining_points[:, 1] - center[1]) / radius_y) ** 2 +
            ((remaining_points[:, 2] - center[2]) / radius_z) ** 2
        ) <= 1

        remaining_points = remaining_points[~mask]

    print(f"Simulated large missing data: {original_points.shape[0] - remaining_points.shape[0]} points removed.")
    return remaining_points





def launch_gui(pcd, filtered_points, original_points, selected_points, output_directory):
    def save_area_action():
        save_path = os.path.join(output_directory, "selected_area.las")
        save_to_las(filtered_points, save_path)
        print(f"Selected area saved to {save_path}")
        gui.destroy()
        no_selection_gui(original_points, output_directory)

    def simulate_missing_on_selected_action():
        # Симулирует отсутствие данных, создавая пропуски только в местах выделенных точек.
        # После завершения предлагает открыть новый файл.
        print("Simulating missing data based on selected points...")
        modified_points = simulate_missing_on_selected_area(original_points, selected_points)

        simulated_save_path = os.path.join(output_directory, "simulated_missing_on_selected.las")
        save_to_las(modified_points, simulated_save_path)
        print(f"Simulated missing data saved to {simulated_save_path}")

        gui.destroy()
        no_selection_gui(original_points, output_directory)




    def interpolate_area_action_with_method(method):
        interpolate_area_action_with_progress(filtered_points, output_directory, gui, method=method)

    def delete_selected_area_action():
        print("Deleting selected area...")
        remaining_points = np.array([
            point for point in np.asarray(pcd.points)
            if point.tolist() not in filtered_points.tolist()
        ])
        print(f"Remaining points after deletion: {remaining_points.shape[0]}")
        remaining_save_path = os.path.join(output_directory, "remaining_points.las")
        save_to_las(remaining_points, remaining_save_path)
        gui.destroy()
        no_selection_gui(original_points, output_directory)

    def open_new_file():
        gui.destroy()
        main()

    def on_close():
        print("Application closed by the user.")
        sys.exit()

    gui = Toplevel()
    gui.title("Select Action")
    gui.protocol("WM_DELETE_WINDOW", on_close)

    Label(gui, text="Choose an action for the selected area:").pack(pady=10)
    Button(gui, text="Save Selected Area", command=save_area_action).pack(pady=5)
    Button(gui, text="Simulate Missing on Selected", command=simulate_missing_on_selected_action).pack(pady=5)
    Button(gui, text="Interpolate (Nearest)", command=lambda: interpolate_area_action_with_method('nearest')).pack(pady=5)
    Button(gui, text="Interpolate (Linear)", command=lambda: interpolate_area_action_with_method('linear')).pack(pady=5)
    Button(gui, text="Interpolate (Spline)", command=lambda: interpolate_area_action_with_method('spline')).pack(pady=5)
    Button(gui, text="Interpolate (Kriging)", command=lambda: interpolate_area_action_with_method('kriging')).pack(pady=5)
    Button(gui, text="Interpolate (Bayesian)", command=lambda: interpolate_area_action_with_method('bayesian')).pack(pady=5)
    Button(gui, text="Delete Selected Area", command=delete_selected_area_action).pack(pady=5)
    Button(gui, text="Open New File", command=open_new_file).pack(pady=10)

    gui.mainloop()






def main():
    try:
        # Выбор файла через интерфейс
        input_file = select_file()
        output_directory = "./test_data/"
        os.makedirs(output_directory, exist_ok=True)

        # Загрузка облака точек
        pcd, original_points = load_las_to_open3d(input_file)
        
        # Выбор области
        selected_points, filtered_points = select_area(pcd, original_points)
        
        if selected_points is None or filtered_points is None:
            print("No area selected.")
            # Открытие окна для работы с полным облаком точек
            no_selection_gui(original_points, output_directory)
            return

        # Запуск графического интерфейса для выбора режима
        launch_gui(pcd, filtered_points, original_points, selected_points, output_directory)

    except Exception as e:
        print(f"An error occurred: {e}")



def no_selection_gui(original_points, output_directory):
    # Открывает окно с кнопками для выбора действия для всего облака точек.
    gui = Toplevel()
    gui.title("No Selection")

    original_file_path = None
    interpolated_file_path = None

    def on_close():
        print("Application closed by the user.")
        sys.exit()

    gui.protocol("WM_DELETE_WINDOW", on_close)

    Label(gui, text="Choose an action for the entire point cloud:").pack(pady=10)

    def select_original_file():
        nonlocal original_file_path
        try:
            original_file_path = select_file()
            print(f"Original file selected: {original_file_path}")
            Label(gui, text=f"Original file: {os.path.basename(original_file_path)}").pack(pady=5)
        except Exception as e:
            print(f"An error occurred while selecting the original file: {e}")

    def select_interpolated_file():
        nonlocal interpolated_file_path
        try:
            interpolated_file_path = select_file()
            print(f"Interpolated file selected: {interpolated_file_path}")
            Label(gui, text=f"Interpolated file: {os.path.basename(interpolated_file_path)}").pack(pady=5)
        except Exception as e:
            print(f"An error occurred while selecting the interpolated file: {e}")

    def compare_two_files_action():
        # Вызывает процесс сравнения двух выбранных файлов и отображает результаты.
        if not original_file_path or not interpolated_file_path:
            print("Both files must be selected for comparison.")
            Label(gui, text="Please select both files before comparing.", fg="red").pack(pady=5)
            return

        try:
            # Загружаем данные
            _, original_points = load_las_to_open3d(original_file_path)
            _, interpolated_points = load_las_to_open3d(interpolated_file_path)

            # Сравниваем файлы
            metrics = compare_point_clouds(original_points, interpolated_points)

            # Отображаем метрики
            result_win = Toplevel(gui)
            result_win.title("Comparison Results")

            Label(result_win, text="Comparison Metrics:").pack(pady=10)
            for key, value in metrics.items():
                Label(result_win, text=f"{key}: {value:.4f}").pack(pady=5)

            Button(result_win, text="Close", command=result_win.destroy).pack(pady=10)
        except Exception as e:
            print(f"An error occurred while comparing files: {e}")

    Button(gui, text="Select Original File", command=select_original_file).pack(pady=5)
    Button(gui, text="Select Interpolated File", command=select_interpolated_file).pack(pady=5)
    Button(gui, text="Compare Two Files", command=compare_two_files_action).pack(pady=10)

    def simulate_missing_on_full_cloud():
        print("Simulating missing data on the entire point cloud...")
        modified_points = simulate_missing_on_selected_area(original_points, original_points)
        simulated_save_path = os.path.join(output_directory, "simulated_missing_full_cloud.las")
        save_to_las(modified_points, simulated_save_path)
        print(f"Simulated missing data saved to {simulated_save_path}")
        gui.destroy()

    def interpolate_full_cloud_with_method(method):
        print(f"Interpolating entire point cloud using {method}...")
        progress_win = Toplevel(gui)
        progress_win.title("Interpolating Entire Point Cloud")
        Label(progress_win, text=f"Interpolating using {method}, please wait...").pack(pady=10)

        progress_bar = ttk.Progressbar(progress_win, orient="horizontal", length=300, mode="determinate")
        progress_bar.pack(pady=20)
        progress_bar["value"] = 0

        def update_progress(value):
            progress_bar["value"] = value
            progress_win.update_idletasks()

        interpolated_points, elapsed_time, peak_memory = measure_performance(
            interpolate_area, original_points, method=method, progress_callback=update_progress
        )
        combined_points = np.vstack((original_points, interpolated_points))
        interpolated_save_path = os.path.join(output_directory, f"interpolated_full_cloud_{method}.las")
        save_to_las(combined_points, interpolated_save_path)
        print(f"Interpolation complete. Saved to {interpolated_save_path}")
        Label(progress_win, text=f"Time: {elapsed_time:.2f}s\nPeak Memory: {peak_memory:.2f} MB").pack(pady=10)
        Button(progress_win, text="Close", command=progress_win.destroy).pack(pady=5)

    Button(gui, text="Simulate Missing Data on Full Cloud", command=simulate_missing_on_full_cloud).pack(pady=5)
    Button(gui, text="Interpolate Full Cloud (Nearest)", command=lambda: interpolate_full_cloud_with_method('nearest')).pack(pady=5)
    Button(gui, text="Interpolate Full Cloud (Linear)", command=lambda: interpolate_full_cloud_with_method('linear')).pack(pady=5)
    Button(gui, text="Interpolate Full Cloud (Spline)", command=lambda: interpolate_full_cloud_with_method('spline')).pack(pady=5)
    Button(gui, text="Interpolate Full Cloud (Kriging)", command=lambda: interpolate_full_cloud_with_method('kriging')).pack(pady=5)
    Button(gui, text="Interpolate Full Cloud (Bayesian)", command=lambda: interpolate_full_cloud_with_method('bayesian')).pack(pady=5)
    Button(gui, text="Exit", command=on_close).pack(pady=10)

    gui.mainloop()




if __name__ == "__main__":
    main()
