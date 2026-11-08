#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Главный скрипт выполнения Лабораторной работы № 3:
"MILP-модель дискретного выбора: Выбор проектов при капитальном бюджете с зависимостями (0/1 рюкзак+)"

Студент: Палатов Иван Александрович
Группа: Б24-517 (Q = 1)
Порядковый номер по списку: N = 16
Вариант: V = 1 + ((17*16 + 7*1 + 26) mod 20) = 1 + 5 = 6
Seed генерации: 10000 + 100*Q + N = 10116
Масштаб задачи: 15 проектов, 5 зависимостей, 3 технологических кластера

Запуск полного комплекса расчетов, экспериментов и всех заданий со звёздочкой (*):
  python3 main.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd

# Обеспечиваем корректный импорт модулей
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from data.generate_data import generate_variant_data
from src.model import ProjectPortfolioModel
from src.solver import HighsSolver
from src.verification import SolutionVerifier
from src.scenarios import ScenarioManager
from src.greedy import GreedyProjectSelector
from src.big_m_analysis import BigMAnalyzer
from src.formulations import FormulationComparator
from src.stochastic import StochasticPortfolioOptimizer
from src.benchmark import ScalingBenchmarkRunner
from src.visualization import PortfolioVisualizer


def main():
    total_start_time = time.perf_counter()

    print("\n" + "=" * 80)
    print("  МИФИ | КАФЕДРА КИБЕРНЕТИКИ | МЕТОДЫ ОПТИМИЗАЦИИ. ЛИНЕЙНОЕ ПРОГРАММИРОВАНИЕ")
    print("  ЛАБОРАТОРНАЯ РАБОТА № 3: MILP-МОДЕЛЬ ДИСКРЕТНОГО ВЫБОРА")
    print("=" * 80)
    print("  Студент         : Палатов Иван Александрович")
    print("  Учебная группа  : Б24-517 (индекс группы Q = 1)")
    print("  Номер по списку : N = 16")
    print("  Год выполнения  : 2026 (Y = 26)")
    print("  Номер варианта  : V = 1 + ((17*16 + 7*1 + 26) mod 20) = 1 + 5 = 6")
    print("  Генератор seed  : 10000 + 100*Q + N = 10116")
    print("  Тема варианта   : Выбор проектов при капитальном бюджете с зависимостями (0/1 рюкзак+)")
    print("  Масштаб задачи  : 15 проектов, 5 зависимостей, 3 технологических кластера")
    print("=" * 80)

    # 0. Создаем структуру каталогов
    os.makedirs("data", exist_ok=True)
    os.makedirs("output/plots", exist_ok=True)
    os.makedirs("interactive", exist_ok=True)

    # 1. Генерация / загрузка детерминированных данных
    print("\n[ЭТАП 1/9] Генерация детерминированных параметров варианта (seed=10116)...")
    data = generate_variant_data(seed=10116, data_dir="data")

    # 2. Построение математической модели и решение базового сценария
    print("\n[ЭТАП 2/9] Построение MILP-модели и непрерывной LP-релаксации...")
    model_builder = ProjectPortfolioModel(data)
    problem_base = model_builder.build_problem(formulation="strong", big_m_factor=1.0)

    solver = HighsSolver(presolve=True)

    print("  -> Запуск промышленного решателя HiGHS (Branch-and-Cut MILP)...")
    milp_res = solver.solve_milp(problem_base)
    print("  -> Запуск решателя HiGHS (Dual Simplex LP Relaxation)...")
    lp_res = solver.solve_lp_relaxation(problem_base)

    npv_milp = milp_res["npv"]
    npv_lp = lp_res["npv"]
    int_gap = ((npv_lp - npv_milp) / npv_milp * 100.0) if npv_milp > 0 else 0.0

    print(f"  [РЕЗУЛЬТАТ] MILP NPV (Оптимум) : {npv_milp:.4f} млн руб. (время: {milp_res['solve_time_sec']*1000:.2f} мс)")
    print(f"  [РЕЗУЛЬТАТ] LP NPV (Релаксация): {npv_lp:.4f} млн руб. (время: {lp_res['solve_time_sec']*1000:.2f} мс)")
    print(f"  [РЕЗУЛЬТАТ] Интегральный зазор : {int_gap:.2f}% (MIP Nodes: {milp_res['mip_nodes']})")

    # Сохраняем лог солвера
    with open("output/solver_log.txt", "w", encoding="utf-8") as f:
        f.write(milp_res["log"])
        f.write("\n\n")
        f.write(lp_res["log"])

    # Формируем таблицу базового решения
    n_p = model_builder.n
    x_sol = milp_res["x"][0:n_p]
    y_sol = milp_res["x"][n_p:2*n_p]
    z_sol = milp_res["x"][2*n_p:]

    sol_rows = []
    for i in range(n_p):
        p = data["projects"][i]
        is_sel = int(round(x_sol[i]))
        y_val = round(float(y_sol[i]), 3)
        cost_p = round(p["c_fix"] * is_sel + p["c_var"] * y_val, 2)
        labor_p = round(p["w_fix"] * is_sel + p["w_var"] * y_val, 2)
        npv_p = round(p["p_fix"] * is_sel + p["p_var"] * y_val, 2)
        sol_rows.append({
            "Project_ID": p["project_id"],
            "Name": p["name"],
            "Cluster": p["cluster_id"],
            "Selected_x": is_sel,
            "Activity_y": y_val,
            "Max_Activity_U": p["u_max"],
            "Capital_Cost_MRub": cost_p,
            "Labor_ManMonths": labor_p,
            "NPV_Contribution_MRub": npv_p
        })
    df_sol = pd.DataFrame(sol_rows)
    df_sol.to_csv("output/solution_base.csv", index=False, encoding="utf-8")

    # 3. Независимая проверка допустимости решения (NumPy)
    print("\n[ЭТАП 3/9] Независимая проверка решения через NumPy (допуск 1e-7)...")
    verifier = SolutionVerifier(data, tol=1e-7)
    is_valid, report_lines, metrics = verifier.verify_solution(
        x_vec=milp_res["x"],
        solver_npv=npv_milp,
        budget=problem_base["budget"],
        labor=problem_base["labor"],
        cost_multiplier=problem_base["cost_multiplier"],
        formulation="strong",
        big_m_factor=1.0
    )

    with open("output/verification_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    for line in report_lines:
        print("  " + line)

    if not is_valid:
        print("[ОШИБКА] Решение не прошло независимую проверку! Прерывание выполнения.")
        sys.exit(1)

    # 4. Сценарный анализ (Base, Deficit, Stress)
    print("\n[ЭТАП 4/9] Проведение сценарного анализа (Base, Deficit, Stress)...")
    sc_mgr = ScenarioManager(data)
    df_scenarios = sc_mgr.run_all_scenarios(output_dir="output")
    print(df_scenarios[["Scenario", "MILP_NPV_MRub", "LP_NPV_MRub", "Integrality_Gap_Pct", "Selected_Count", "Capital_Utilization_Pct"]].to_string(index=False))

    # 5. Сравнение с жадной эвристикой
    print("\n[ЭТАП 5/9] Запуск жадной эвристики и сравнение с оптимумом...")
    greedy = GreedyProjectSelector(data)
    res_greedy = greedy.solve(budget=problem_base["budget"], labor=problem_base["labor"])
    greedy_npv = res_greedy["npv"]
    greedy_gap = ((npv_milp - greedy_npv) / npv_milp * 100.0)

    print(f"  [ЖАДНАЯ ЭВРИСТИКА] NPV          : {greedy_npv:.2f} млн руб. (время: {res_greedy['solve_time_us']:.1f} мкс)")
    print(f"  [ТОЧНЫЙ MILP HiGHS] NPV        : {npv_milp:.2f} млн руб. (время: {milp_res['solve_time_sec']*1000:.2f} мс)")
    print(f"  [ПОТЕРИ КАЧЕСТВА] Greedy Gap   : {greedy_gap:.2f}%")
    print(f"  [ВЫБРАННЫЕ ПРОЕКТЫ ЖАДНОЙ]     : {', '.join(res_greedy['selected_projects'])}")

    # 6. Исследование влияния константы Big-M
    print("\n[ЭТАП 6/9] Численное исследование влияния константы Big-M...")
    big_m_analyzer = BigMAnalyzer(data)
    df_big_m = big_m_analyzer.run_analysis(output_dir="output")
    print(df_big_m[["BigM_Factor", "Avg_BigM_Value", "LP_NPV_MRub", "Integrality_Gap_Pct", "Fractionality_Metric"]].to_string(index=False))

    # 7. Сравнение сильной и слабой формулировок MILP
    print("\n[ЭТАП 7/9] Сравнение двух эквивалентных формулировок (Strong vs Weak)...")
    form_comp = FormulationComparator(data)
    df_form = form_comp.compare_formulations(output_dir="output")
    print(df_form[["Formulation_Key", "Num_Constraints", "LP_NPV_MRub", "MILP_NPV_MRub", "Integrality_Gap_Pct"]].to_string(index=False))

    # 8. Двухэтапная стохастическая модель (*)
    print("\n[ЭТАП 8/9] Решение двухэтапной стохастической задачи (задание со звёздочкой *)...")
    stoch_opt = StochasticPortfolioOptimizer(data)
    df_stoch = stoch_opt.run_stochastic_analysis(output_dir="output")
    print(df_stoch[["Metric", "Value_MRub", "Interpretation"]].to_string(index=False))

    # 9. Нагрузочный бенчмарк масштабируемости
    print("\n[ЭТАП 9/9] Вычислительный бенчмарк масштабируемости (N = 10..160 проектов)...")
    bench_runner = ScalingBenchmarkRunner(base_seed=10116)
    df_bench = bench_runner.run_benchmark(dimensions=[10, 15, 20, 30, 45, 60, 90, 120, 160], output_dir="output")
    print(df_bench[["N_Projects", "Total_Variables", "MILP_Time_ms", "LP_Time_ms", "Greedy_Time_ms", "Integrality_Gap_Pct", "Greedy_Gap_Pct"]].to_string(index=False))

    # 10. Генерация всех графиков
    print("\n[ВИЗУАЛИЗАЦИЯ] Построение 7 публикационных графиков (300 DPI)...")
    visualizer = PortfolioVisualizer(data, plots_dir="output/plots")
    visualizer.plot_project_selection(milp_res)
    visualizer.plot_dependency_dag(milp_res)
    visualizer.plot_scenarios_comparison(df_scenarios)
    visualizer.plot_big_m_impact(df_big_m)
    visualizer.plot_formulations_comparison(df_form)
    visualizer.plot_benchmark_scaling(df_bench)
    visualizer.plot_stochastic_analysis(df_stoch)
    print("  [OK] Все графики успешно сохранены в output/plots/")

    total_time = time.perf_counter() - total_start_time
    print("\n" + "=" * 80)
    print(f"  ВСЕ ЭТАПЫ ЛАБОРАТОРНОЙ РАБОТЫ № 3 УСПЕШНО ВЫПОЛНЕНЫ ЗА {total_time:.2f} с!")
    print("  Интерактивный веб-стенд: interactive/capital_budgeting_demo.html")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
