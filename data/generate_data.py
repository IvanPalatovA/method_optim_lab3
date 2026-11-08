#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Генератор исходных данных для Лабораторной работы № 3:
"MILP-модель дискретного выбора: Выбор проектов при капитальном бюджете с зависимостями"

Студент: Палатов Иван Александрович
Группа: Б24-517 (Q = 1)
Порядковый номер в списке группы: N = 16
Год выполнения: 2026 (Y = 26)
Вариант: V = 1 + ((17*16 + 7*1 + 26) mod 20) = 1 + 5 = 6
Seed генерации: 10000 + 100*Q + N = 10116
Масштаб: 15 проектов, 5 зависимостей, 3 технологических кластера

Формирует:
  - data/input_data.json
  - data/projects_data.csv
  - data/dependencies.csv
"""

import os
import json
import numpy as np
import pandas as pd


def generate_variant_data(seed: int = 10116, data_dir: str = "data"):
    """
    Детерминированная генерация параметров варианта 6.
    """
    os.makedirs(data_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    n_projects = 15
    clusters_info = [
        {
            "cluster_id": "K1",
            "name": "Вычислительная инфраструктура и промышленный ИИ",
            "fixed_cost": 35.0,
            "fixed_labor": 12.0,
            "description": "Кластер высокопроизводительных серверов, шины данных и платформ машинного обучения"
        },
        {
            "cluster_id": "K2",
            "name": "Промышленная автоматизация, робототехника и энергосистемы",
            "fixed_cost": 40.0,
            "fixed_labor": 15.0,
            "description": "Стенды роботизированного контроля, полевые датчики и микрогрид-накопители"
        },
        {
            "cluster_id": "K3",
            "name": "Комплексная кибербезопасность и защищенные коммуникации",
            "fixed_cost": 30.0,
            "fixed_labor": 10.0,
            "description": "Аппаратный шлюз криптографической защиты, центр мониторинга угроз SOC"
        }
    ]

    project_names = [
        "P1: Единая шина интеграции данных АСУТП (Enterprise Data Bus)",
        "P2: Предиктивная аналитика оборудования на базе ML (Predictive AI)",
        "P3: Цифровые двойники турбогенераторов и насосов (Digital Twin)",
        "P4: Система предиктивной вибродиагностики (Vibration Diagnostics)",
        "P5: Роботизированный комплекс неразрушающего контроля (Robotic NDT)",
        "P6: Централизованный центр мониторинга кибербезопасности (SOC)",
        "P7: Квантово-криптографическая защита каналов связи (Quantum Security)",
        "P8: Мониторинг радиационной и экологической обстановки (Eco/Radiation)",
        "P9: Модульная микрогрид-система с BESS-накопителями (Microgrid & BESS)",
        "P10: Облачная диспетчеризация MES/ERP (Cloud MES/ERP Operations)",
        "P11: AR-ассистент полевого инженера-ремонтника (AR Field Operations)",
        "P12: Оптическая дефектоскопия сварных швов (Computer Vision Defect)",
        "P13: On-Premises высокопроизводительный кластер (HPC Cluster)",
        "P14: Защищенная гибридная облачная инфраструктура (Secure Hybrid Cloud)",
        "P15: Оптимизатор энергопотребления энергоблока (Energy Efficiency AI)"
    ]

    # Принадлежность к кластерам (0: K1, 1: K2, 2: K3)
    cluster_indices = [
        0, 0, 0, 0,  # P1, P2, P3, P4 -> K1
        1,          # P5 -> K2
        2, 2,       # P6, P7 -> K3
        1, 1,       # P8, P9 -> K2
        0,          # P10 -> K1
        1, 1,       # P11, P12 -> K2
        0, 0,       # P13, P14 -> K1
        1           # P15 -> K2
    ]

    # Фиксированные капитальные затраты c_fix (млн руб.): 15..54
    c_fix = rng.integers(15, 55, size=n_projects).astype(float)
    # Удельные переменные затраты c_var (млн руб./ед.): 1.50..3.99
    c_var = np.round(rng.uniform(1.5, 4.0, size=n_projects), 2)
    # Максимальный уровень внедрения U_i (ед.): 5..15
    u_max = rng.integers(5, 16, size=n_projects).astype(float)

    # Фиксированный дисконтированный доход NPV p_fix (млн руб.): 25..89
    p_fix = rng.integers(25, 90, size=n_projects).astype(float)
    # Удельный доход масштабирования p_var (млн руб./ед.): 3.00..8.49
    p_var = np.round(rng.uniform(3.0, 8.5, size=n_projects), 2)

    # Трудовые ресурсы w_fix (чел-мес): 6..24
    w_fix = rng.integers(6, 25, size=n_projects).astype(float)
    # Удельные трудозатраты w_var (чел-мес/ед.): 0.80..2.49
    w_var = np.round(rng.uniform(0.8, 2.5, size=n_projects), 2)

    # Сводный список проектов
    projects = []
    for i in range(n_projects):
        c_idx = cluster_indices[i]
        cl = clusters_info[c_idx]
        projects.append({
            "project_id": f"P{i+1}",
            "index": i,
            "name": project_names[i],
            "cluster_id": cl["cluster_id"],
            "cluster_name": cl["name"],
            "c_fix": float(c_fix[i]),
            "c_var": float(c_var[i]),
            "u_max": float(u_max[i]),
            "p_fix": float(p_fix[i]),
            "p_var": float(p_var[i]),
            "w_fix": float(w_fix[i]),
            "w_var": float(w_var[i]),
            "max_capital_cost": float(round(c_fix[i] + c_var[i] * u_max[i], 2)),
            "max_labor_cost": float(round(w_fix[i] + w_var[i] * u_max[i], 2)),
            "max_npv": float(round(p_fix[i] + p_var[i] * u_max[i], 2))
        })

    # 5 структурных зависимостей (Dependencies)
    dependencies = [
        {
            "dep_id": "D1",
            "type": "Precedence",
            "from_project": "P1",
            "to_project": "P2",
            "algebraic_constraint": "x2 - x1 <= 0",
            "description": "Предиктивный ИИ (P2) требует обязательного наличия шины данных (P1)"
        },
        {
            "dep_id": "D2",
            "type": "Precedence",
            "from_project": "P1",
            "to_project": "P3",
            "algebraic_constraint": "x3 - x1 <= 0",
            "description": "Цифровые двойники (P3) функционируют только на потоках шины данных (P1)"
        },
        {
            "dep_id": "D3",
            "type": "Precedence",
            "from_project": "P2",
            "to_project": "P4",
            "algebraic_constraint": "x4 - x2 <= 0",
            "description": "Глубокая вибродиагностика (P4) использует ML-модели ядра предиктивного ИИ (P2)"
        },
        {
            "dep_id": "D4",
            "type": "Mutual Exclusion",
            "from_project": "P13",
            "to_project": "P14",
            "algebraic_constraint": "x13 + x14 <= 1",
            "description": "Взаимоисключение: либо локальный вычислительный кластер (P13), либо гибридное облако (P14)"
        },
        {
            "dep_id": "D5",
            "type": "Co-requisite",
            "from_project": "P5",
            "to_project": "P12",
            "algebraic_constraint": "x12 - x5 <= 0",
            "description": "Оптическая дефектоскопия сварки (P12) монтируется на роботехнический комплекс НК (P5)"
        }
    ]

    total_max_capital = sum(p["max_capital_cost"] for p in projects) + sum(c["fixed_cost"] for c in clusters_info)
    total_max_labor = sum(p["max_labor_cost"] for p in projects) + sum(c["fixed_labor"] for c in clusters_info)

    # Параметры бюджетов по сценариям
    scenarios_params = {
        "Base": {
            "name": "Базовый сценарий (Плановый бюджет)",
            "budget": float(np.round(total_max_capital * 0.55, 0)),
            "labor": float(np.round(total_max_labor * 0.60, 0)),
            "cost_multiplier": 1.0,
            "description": "Штатное финансирование инвестиционной программы цифровизации"
        },
        "Deficit": {
            "name": "Дефицитный сценарий (Секвестр инвестиций -31%)",
            "budget": float(np.round(total_max_capital * 0.38, 0)),
            "labor": float(np.round(total_max_labor * 0.45, 0)),
            "cost_multiplier": 1.0,
            "description": "Сокращение капитальных вложений и дефицит квалифицированных специалистов"
        },
        "Stress": {
            "name": "Стрессовый сценарий (Кризис: бюджет -49%, инфляция затрат +20%)",
            "budget": float(np.round(total_max_capital * 0.28, 0)),
            "labor": float(np.round(total_max_labor * 0.35, 0)),
            "cost_multiplier": 1.2,
            "description": "Жесткие ресурсные лимиты в сочетании с 20% инфляционным ростом себестоимости оборудования"
        }
    }

    full_data = {
        "metadata": {
            "title": "Инвестиционная программа цифровизации и технологической модернизации",
            "discipline": "Методы оптимизации. Линейное программирование",
            "organization": "НИЯУ МИФИ",
            "student": "Палатов Иван Александрович",
            "group": "Б24-517",
            "variant": 6,
            "seed": seed,
            "n_projects": n_projects,
            "n_dependencies": len(dependencies),
            "n_clusters": len(clusters_info),
            "total_potential_capital": float(round(total_max_capital, 2)),
            "total_potential_labor": float(round(total_max_labor, 2))
        },
        "clusters": clusters_info,
        "projects": projects,
        "dependencies": dependencies,
        "scenarios": scenarios_params
    }

    # Сохраняем в JSON
    json_path = os.path.join(data_dir, "input_data.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_data, f, ensure_ascii=False, indent=2)

    # Сохраняем проекты в CSV
    df_proj = pd.DataFrame(projects)
    csv_proj_path = os.path.join(data_dir, "projects_data.csv")
    df_proj.to_csv(csv_proj_path, index=False, encoding="utf-8")

    # Сохраняем зависимости в CSV
    df_dep = pd.DataFrame(dependencies)
    csv_dep_path = os.path.join(data_dir, "dependencies.csv")
    df_dep.to_csv(csv_dep_path, index=False, encoding="utf-8")

    print(f"[OK] Данные успешно сгенерированы (seed={seed}):")
    print(f"     JSON : {json_path}")
    print(f"     CSV  : {csv_proj_path}")
    print(f"     CSV  : {csv_dep_path}")

    return full_data


if __name__ == "__main__":
    generate_variant_data()
