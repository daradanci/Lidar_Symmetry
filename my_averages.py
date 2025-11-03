import os
import csv
import numpy as np
from collections import defaultdict

def compute_symmetry_averages(logs_folder="symmetry_logs", output_file="symmetry_averages.csv", reverse_rows=False, row_step=1):
    """
    Считает среднее значение симметрии для каждой ячейки (i, j) из всех CSV-файлов в папке.
    
    :param logs_folder: Папка с файлами symmetry_tree_XXXX.csv
    :param output_file: Имя выходного CSV-файла со средними значениями
    :param reverse_rows: Если True — строки будут записаны в обратном порядке
    :param row_step: Сколько строк объединять при усреднении (например, 3 — каждые 3 строки объединяются)
    """
    values_by_position = defaultdict(list)

    for filename in os.listdir(logs_folder):
        if not filename.endswith(".csv"):
            continue
        filepath = os.path.join(logs_folder, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            reader = list(csv.reader(f))
            for row_idx, row in enumerate(reader):
                for col_idx, value in enumerate(row):
                    try:
                        values_by_position[(row_idx, col_idx)].append(float(value))
                    except ValueError:
                        pass

    max_row = max(k[0] for k in values_by_position) + 1
    max_col = max(k[1] for k in values_by_position) + 1

    grouped_results = []

    for group_start in range(0, max_row, row_step):
        group_rows = []
        for offset in range(row_step):
            i = group_start + offset
            if i >= max_row:
                break
            row = []
            for j in range(max_col):
                vals = values_by_position.get((i, j), [])
                row.append(vals)
            group_rows.append(row)

        # Усредняем значения по группе
        averaged_row = []
        for j in range(max_col):
            collected = []
            for i in range(len(group_rows)):
                collected.extend(group_rows[i][j])
            if collected:
                averaged_row.append(f"{np.mean(collected):.3f}")
            else:
                averaged_row.append("")
        grouped_results.append(averaged_row)

    if reverse_rows:
        grouped_results = grouped_results[::-1]

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(grouped_results)

    print(f"✅ Средние значения сохранены в {output_file} (reverse_rows={reverse_rows}, row_step={row_step})")

# Пример запуска
compute_symmetry_averages(reverse_rows=False, row_step=3)
