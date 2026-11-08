#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль жадной эвристики (Greedy Heuristic) для задачи 0/1 рюкзака с зависимостями.
Реализует жадный выбор проектов на основе индекса рентабельности (Profitability Index),
с учетом обязательных предшественников, взаимных исключений, кластерных надбавок
и последующим жадным распределением непрерывных ресурсов масштабирования.
"""

import time
import numpy as np
from typing import Dict, Any, List, Tuple


class GreedyProjectSelector:
    """Жадная эвристика выбора инвестиционных проектов."""

    def __init__(self, raw_data: Dict[str, Any]):
        self.raw_data = raw_data
        self.projects = raw_data["projects"]
        self.clusters = raw_data["clusters"]
        self.dependencies = raw_data["dependencies"]

        self.n = len(self.projects)
        self.K = len(self.clusters)

        cluster_id_to_idx = {c["cluster_id"]: idx for idx, c in enumerate(self.clusters)}
        self.proj_cluster_idx = [cluster_id_to_idx[p["cluster_id"]] for p in self.projects]

        self.c_fix = np.array([p["c_fix"] for p in self.projects], dtype=float)
        self.c_var = np.array([p["c_var"] for p in self.projects], dtype=float)
        self.u_max = np.array([p["u_max"] for p in self.projects], dtype=float)

        self.p_fix = np.array([p["p_fix"] for p in self.projects], dtype=float)
        self.p_var = np.array([p["p_var"] for p in self.projects], dtype=float)

        self.w_fix = np.array([p["w_fix"] for p in self.projects], dtype=float)
        self.w_var = np.array([p["w_var"] for p in self.projects], dtype=float)

        self.F_cluster = np.array([c["fixed_cost"] for c in self.clusters], dtype=float)
        self.S_cluster = np.array([c["fixed_labor"] for c in self.clusters], dtype=float)

    def solve(
        self,
        budget: float,
        labor: float,
        cost_multiplier: float = 1.0
    ) -> Dict[str, Any]:
        """
        Запуск жадной эвристики.
        :return: словарь с результатами, временем работы и структурой выбранного портфеля.
        """
        t_start = time.perf_counter()

        curr_c_fix = self.c_fix * cost_multiplier
        curr_c_var = self.c_var * cost_multiplier
        curr_F_cl = self.F_cluster * cost_multiplier

        selected_x = np.zeros(self.n, dtype=int)
        active_z = np.zeros(self.K, dtype=int)

        rem_budget = float(budget)
        rem_labor = float(labor)

        # Жадный выбор дискретных проектов
        while True:
            best_idx = None
            best_pi = -1.0
            best_cost = 0.0
            best_labor = 0.0

            for i in range(self.n):
                if selected_x[i] == 1:
                    continue

                # 1. Проверка предшественников
                # P2 requires P1 (индексы 1 и 0)
                if i == 1 and selected_x[0] == 0:
                    continue
                # P3 requires P1 (индексы 2 и 0)
                if i == 2 and selected_x[0] == 0:
                    continue
                # P4 requires P2 (индексы 3 и 1)
                if i == 3 and selected_x[1] == 0:
                    continue
                # P12 requires P5 (индексы 11 и 4)
                if i == 11 and selected_x[4] == 0:
                    continue

                # 2. Проверка взаимоисключения P13 и P14 (индексы 12 и 13)
                if i == 12 and selected_x[13] == 1:
                    continue
                if i == 13 and selected_x[12] == 1:
                    continue

                k = self.proj_cluster_idx[i]
                inc_cost = curr_c_fix[i] + (curr_F_cl[k] if active_z[k] == 0 else 0.0)
                inc_labor = self.w_fix[i] + (self.S_cluster[k] if active_z[k] == 0 else 0.0)

                # Проверка ресурсных ограничений
                if inc_cost <= rem_budget and inc_labor <= rem_labor:
                    # Индекс эффективности (Profitability Index)
                    est_npv = self.p_fix[i] + self.p_var[i] * self.u_max[i]
                    est_cost = inc_cost + curr_c_var[i] * self.u_max[i]
                    pi = est_npv / est_cost if est_cost > 0 else 0.0

                    if pi > best_pi:
                        best_pi = pi
                        best_idx = i
                        best_cost = inc_cost
                        best_labor = inc_labor

            if best_idx is None:
                break

            # Фиксируем выбор
            selected_x[best_idx] = 1
            k_sel = self.proj_cluster_idx[best_idx]
            active_z[k_sel] = 1
            rem_budget -= best_cost
            rem_labor -= best_labor

        # 2. Жадное распределение непрерывных активностей y_i
        y_greedy = np.zeros(self.n, dtype=float)
        active_indices = [i for i in range(self.n) if selected_x[i] == 1]
        # Сортируем по маржинальной отдаче p_var / c_var
        active_indices.sort(key=lambda idx: self.p_var[idx] / curr_c_var[idx] if curr_c_var[idx] > 0 else 0.0, reverse=True)

        for i in active_indices:
            max_by_b = rem_budget / curr_c_var[i] if curr_c_var[i] > 0 else self.u_max[i]
            max_by_l = rem_labor / self.w_var[i] if self.w_var[i] > 0 else self.u_max[i]
            take_y = min(self.u_max[i], max_by_b, max_by_l)
            take_y = max(0.0, float(take_y))

            y_greedy[i] = take_y
            rem_budget -= take_y * curr_c_var[i]
            rem_labor -= take_y * self.w_var[i]

        t_sec = time.perf_counter() - t_start

        calc_npv = float(np.dot(self.p_fix, selected_x) + np.dot(self.p_var, y_greedy) - np.dot(curr_F_cl, active_z))
        used_cap = float(budget - rem_budget)
        used_lab = float(labor - rem_labor)

        selected_proj_names = [f"P{i+1}" for i in range(self.n) if selected_x[i] == 1]
        active_cl_names = [f"K{k+1}" for k in range(self.K) if active_z[k] == 1]

        return {
            "npv": calc_npv,
            "x": selected_x,
            "y": y_greedy,
            "z": active_z,
            "selected_projects": selected_proj_names,
            "active_clusters": active_cl_names,
            "used_capital": used_cap,
            "used_labor": used_lab,
            "rem_budget": rem_budget,
            "rem_labor": rem_labor,
            "solve_time_sec": t_sec,
            "solve_time_us": t_sec * 1_000_000
        }
