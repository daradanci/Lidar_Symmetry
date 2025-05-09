import os
import csv
import numpy as np
from collections import defaultdict

def compute_symmetry_averages(logs_folder="symmetry_logs", output_file="symmetry_averages.csv"):
    """
    Считает среднее значение симметрии для каждой ячейки (i, j) из всех CSV-файлов в папке.

    :param logs_folder: Папка с файлами symmetry_tree_XXXX.csv
    :param output_file: Имя выходного CSV-файла со средними значениями
    """
    # Словарь вида: (row_idx, col_idx) -> [values...]
    values_by_position = defaultdict(list)

    # Читаем все CSV-файлы
    for filename in os.listdir(logs_folder):
        if not filename.endswith(".csv"):
            continue
        filepath = os.path.join(logs_folder, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row_idx, row in enumerate(reader):
                for col_idx, value in enumerate(row):
                    try:
                        values_by_position[(row_idx, col_idx)].append(float(value))
                    except ValueError:
                        pass  # Пропуск пустых или некорректных ячеек

    # Находим максимальное количество строк и столбцов
    max_row = max(k[0] for k in values_by_position) + 1
    max_col = max(k[1] for k in values_by_position) + 1

    # Формируем таблицу средних значений
    result = []
    for i in range(max_row):
        row = []
        for j in range(max_col):
            vals = values_by_position.get((i, j), [])
            if vals:
                row.append(f"{np.mean(vals):.3f}")
            else:
                row.append("")
        result.append(row)

    # Сохраняем результат
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(result)

    print(f"✅ Средние значения сохранены в {output_file}")

# Запуск
compute_symmetry_averages()
