import csv
import os
from classes.PCD_TREE import PCD_TREE, find_trunk_center

def save_symmetry_log(csv_path, group_num, tree_id, stage, score):
    header = ["Группа", "Дерево", "Этап", "Симметрия"]
    row = [group_num, tree_id, stage, f"{score:.3f}" if score is not None else ""]
    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)

def process_tree_groups(
    groups,
    data_folder,
    output_csv="symmetry_log.csv",
    voxel_size=0.15,
    z_step=1.0,
    z_threshold=0.1,
):
    if os.path.exists(output_csv):
        os.remove(output_csv)

    for group_num, group in enumerate(groups, start=1):
        print(f"\n📦 Группа {group_num}: деревья {group}")

        # Этап 1: ДО и после балансировки
        trees = []
        for i in group:
            file_path = os.path.join(data_folder, f"tree_{i:04d}.pcd")
            tree = PCD_TREE()
            tree.open(file_path, verbose=False)
            tree.file_path = file_path
            tree.set_trunk_center(z_threshold=z_threshold, min_points=10)
            tree.find_tree_top()
            tree.voxelize_tree(voxel_size=voxel_size)

            try:
                score_initial = tree.measure_tree_symmetry(z_step=z_step)
            except ValueError as e:
                if "trunk_x" in str(e):
                    print("⚠ Повторное определение центра ствола через find_trunk_center()...")
                    tree.find_tree_top()  # на всякий случай
                    center = find_trunk_center(tree.get_active_points(), z_threshold=z_threshold+2, min_points=1)
                    if center:
                        tree.trunk_x, tree.trunk_y = center
                        score_initial = tree.measure_tree_symmetry(z_step=z_step)
                    else:
                        print(f"❌ Не удалось восстановить координаты ствола для дерева {tree.file_path}")
                        score_initial = None
                else:
                    raise




            save_symmetry_log(output_csv, group_num, i, "До восстановления", score_initial)
            trees.append(tree)

        for tree in trees:
            neighbors = [t for t in trees if t is not tree]
            tree.restore_symmetry(
                neighbor_trees=neighbors,
                z_step=z_step,
                voxel_size=voxel_size,
                generate_mirrored=False
            )
            score_balanced = tree.measure_tree_symmetry(z_step=z_step)
            tree_id = int(tree.file_path[-8:-4])
            save_symmetry_log(output_csv, group_num, tree_id, "После балансировки", score_balanced)

        # Этап 2: После генерации (повторная загрузка)
        for i in group:
            file_path = os.path.join(data_folder, f"tree_{i:04d}.pcd")
            tree = PCD_TREE()
            tree.open(file_path, verbose=False)
            tree.file_path = file_path
            tree.set_trunk_center(z_threshold=z_threshold, min_points=10)

            # 🔁 Повторная попытка, если не удалось найти основание
            if tree.trunk_x is None or tree.trunk_y is None:
                print(f"⚠ Повторный поиск центра ствола для дерева {i}...")
                center = find_trunk_center(tree.get_active_points(), z_threshold=z_threshold+2, min_points=1)
                if center:
                    tree.trunk_x, tree.trunk_y = center
                    print(f"✅ Центр обновлён: X = {tree.trunk_x:.2f}, Y = {tree.trunk_y:.2f}")
                else:
                    print(f"❌ Не удалось определить центр ствола у дерева {i}, пропуск.")
                    continue

            tree.find_tree_top()
            tree.voxelize_tree(voxel_size=voxel_size)

            neighbors = []
            for j in group:
                if j != i:
                    neighbor = PCD_TREE()
                    neighbor_path = os.path.join(data_folder, f"tree_{j:04d}.pcd")
                    neighbor.open(neighbor_path, verbose=False)
                    neighbor.file_path = neighbor_path
                    neighbor.set_trunk_center(z_threshold=z_threshold, min_points=10)

                    if neighbor.trunk_x is None or neighbor.trunk_y is None:
                        print(f"⚠ Повторный поиск центра ствола у соседа {j}...")
                        center = find_trunk_center(neighbor.get_active_points(), z_threshold=z_threshold+2, min_points=1)
                        if center:
                            neighbor.trunk_x, neighbor.trunk_y = center
                        else:
                            print(f"❌ Пропуск соседа {j} — нет центра ствола.")
                            continue

                    neighbor.find_tree_top()
                    neighbor.voxelize_tree(voxel_size=voxel_size)
                    neighbors.append(neighbor)

            tree.restore_symmetry(
                neighbor_trees=neighbors,
                z_step=z_step,
                voxel_size=voxel_size,
                generate_mirrored=True
            )
            score_generated = tree.measure_tree_symmetry(z_step=z_step)
            save_symmetry_log(output_csv, group_num, i, "После генерации", score_generated)



        print(f"✅ Группа {group_num} обработана и залогирована.")


groups_to_process = [
    [98, 99],
    [13, 14, 15],
    [18, 19, 20],
    [23, 149],
    [37, 146],
    [44, 75],
    [48, 49, 50],
]

process_tree_groups(groups=groups_to_process, data_folder="D:/data/symmetry")
