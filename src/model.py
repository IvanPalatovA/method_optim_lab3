#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль формализации MILP-модели задачи выбора проектов при капитальном бюджете.
Поддерживает сильную (Strong) и слабую (Weak) формулировки связок кластеров,
настраиваемый параметр Big-M, непрерывные переменные масштабирования проектов
и структурные технологические зависимости (предшествование, взаимоисключение, сопряжение).
"""

import numpy as np
from typing import Dict, Any, List, Optional


class ProjectPortfolioModel:
    """Класс построения математической модели MILP для задачи выбора проектов."""

    def __init__(self, data: Dict[str, Any]):
        """
        Инициализация данными варианта.
        :param data: словарь структуры input_data.json
        """
        self.raw_data = data
        self.projects = data["projects"]
        self.clusters = data["clusters"]
        self.dependencies = data["dependencies"]
        self.scenarios_info = data["scenarios"]

        self.n = len(self.projects)
        self.K = len(self.clusters)

        # Сопоставление кластеров: project_index -> cluster_index
        cluster_id_to_idx = {c["cluster_id"]: idx for idx, c in enumerate(self.clusters)}
        self.project_to_cluster = [cluster_id_to_idx[p["cluster_id"]] for p in self.projects]

        # Базовые векторы параметров
        self.c_fix = np.array([p["c_fix"] for p in self.projects], dtype=float)
        self.c_var = np.array([p["c_var"] for p in self.projects], dtype=float)
        self.u_max = np.array([p["u_max"] for p in self.projects], dtype=float)

        self.p_fix = np.array([p["p_fix"] for p in self.projects], dtype=float)
        self.p_var = np.array([p["p_var"] for p in self.projects], dtype=float)

        self.w_fix = np.array([p["w_fix"] for p in self.projects], dtype=float)
        self.w_var = np.array([p["w_var"] for p in self.projects], dtype=float)

        self.F_cluster = np.array([c["fixed_cost"] for c in self.clusters], dtype=float)
        self.S_cluster = np.array([c["fixed_labor"] for c in self.clusters], dtype=float)

    def build_problem(
        self,
        budget: Optional[float] = None,
        labor: Optional[float] = None,
        cost_multiplier: float = 1.0,
        formulation: str = "strong",
        big_m_factor: float = 1.0,
        fixed_big_m: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Формирование матриц задачи для солвера HiGHS / SciPy.

        Структура вектора переменных (всего 2*n + K переменных):
          - x_0 .. x_{n-1}  : бинарные переменные выбора проектов (1 - выбран, 0 - отклонен)
          - y_0 .. y_{n-1}  : непрерывные переменные масштаба активности (0 <= y_i <= U_i)
          - z_0 .. z_{K-1}  : бинарные переменные активации кластерной инфраструктуры

        :param budget: лимит капитального бюджета (если None - берется базовый)
        :param labor: лимит трудовых ресурсов (чел-мес)
        :param cost_multiplier: множитель инфляции затрат
        :param formulation: 'strong' (сильная) или 'weak' (слабая)
        :param big_m_factor: множитель для константы Big-M (по умолчанию 1.0 -> M_i = U_i)
        :param fixed_big_m: явное задание единой константы Big-M для всех проектов
        :return: словарь с матрицами, векторами целей, границ и метаданными
        """
        if budget is None:
            budget = self.scenarios_info["Base"]["budget"]
        if labor is None:
            labor = self.scenarios_info["Base"]["labor"]

        n_vars = 2 * self.n + self.K

        # 1. Вектор констант Big-M
        if fixed_big_m is not None:
            M_vec = np.full(self.n, fixed_big_m, dtype=float)
        else:
            M_vec = self.u_max * big_m_factor

        # 2. Вектор целевой функции:
        # MAX sum(p_fix * x + p_var * y) - sum(F_cluster * z)
        # Для минимизации (стандарт солвера): MIN - Objective
        c_obj = np.zeros(n_vars, dtype=float)
        c_obj[0:self.n] = -self.p_fix
        c_obj[self.n:2*self.n] = -self.p_var
        c_obj[2*self.n:] = self.F_cluster * cost_multiplier

        # 3. Границы переменных
        lb = np.zeros(n_vars, dtype=float)
        ub = np.zeros(n_vars, dtype=float)
        ub[0:self.n] = 1.0                 # x_i in [0, 1]
        ub[self.n:2*self.n] = self.u_max   # y_i in [0, U_i]
        ub[2*self.n:] = 1.0                # z_k in [0, 1]

        # 4. Вектор целочисленности (1 - целая/бинарная, 0 - непрерывная)
        integrality = np.zeros(n_vars, dtype=int)
        integrality[0:self.n] = 1          # x бинарные
        integrality[2*self.n:] = 1         # z бинарные
        # y - непрерывные (integrality = 0)

        # 5. Ограничения неравенства A_ub @ v <= b_ub
        A_rows: List[np.ndarray] = []
        b_rows: List[float] = []
        constraint_names: List[str] = []

        curr_c_fix = self.c_fix * cost_multiplier
        curr_c_var = self.c_var * cost_multiplier
        curr_F_cl = self.F_cluster * cost_multiplier

        # 5.1. Ограничение на суммарный бюджет капитала
        row_b = np.zeros(n_vars, dtype=float)
        row_b[0:self.n] = curr_c_fix
        row_b[self.n:2*self.n] = curr_c_var
        row_b[2*self.n:] = curr_F_cl
        A_rows.append(row_b)
        b_rows.append(budget)
        constraint_names.append("Capital Budget Limit")

        # 5.2. Ограничение на фонд трудовых ресурсов
        row_l = np.zeros(n_vars, dtype=float)
        row_l[0:self.n] = self.w_fix
        row_l[self.n:2*self.n] = self.w_var
        row_l[2*self.n:] = self.S_cluster
        A_rows.append(row_l)
        b_rows.append(labor)
        constraint_names.append("Workforce Labor Limit")

        # 5.3. Ограничения-переключатели Big-M для непрерывных активностей:
        # y_i - M_i * x_i <= 0  (если x_i = 0, то y_i <= 0, т.е. y_i = 0)
        for i in range(self.n):
            row_m = np.zeros(n_vars, dtype=float)
            row_m[self.n + i] = 1.0
            row_m[i] = -M_vec[i]
            A_rows.append(row_m)
            b_rows.append(0.0)
            constraint_names.append(f"Big-M Switch: y_{i+1} <= M_{i+1} * x_{i+1}")

        # 5.4. Активация кластерной инфраструктуры
        if formulation.lower() == "strong":
            # Сильная формулировка: для каждого проекта x_i <= z_{k(i)}
            for i in range(self.n):
                k = self.project_to_cluster[i]
                row_cl = np.zeros(n_vars, dtype=float)
                row_cl[i] = 1.0
                row_cl[2 * self.n + k] = -1.0
                A_rows.append(row_cl)
                b_rows.append(0.0)
                constraint_names.append(f"Strong Cluster Link: x_{i+1} <= z_{k+1}")
        else:
            # Слабая формулировка: суммарное число проектов <= |C_k| * z_k
            for k in range(self.K):
                cluster_members = [i for i in range(self.n) if self.project_to_cluster[i] == k]
                M_k = len(cluster_members)
                row_wk = np.zeros(n_vars, dtype=float)
                for i in cluster_members:
                    row_wk[i] = 1.0
                row_wk[2 * self.n + k] = -float(M_k)
                A_rows.append(row_wk)
                b_rows.append(0.0)
                constraint_names.append(f"Weak Cluster Link: sum(x in C_{k+1}) <= {M_k} * z_{k+1}")

        # 5.5. Структурные технологические зависимости (5 зависимостей)
        # D1: P2 <= P1 (x_2 - x_1 <= 0)
        r_d1 = np.zeros(n_vars, dtype=float)
        r_d1[1] = 1.0
        r_d1[0] = -1.0
        A_rows.append(r_d1)
        b_rows.append(0.0)
        constraint_names.append("Dep D1 (Precedence): P2 <= P1")

        # D2: P3 <= P1 (x_3 - x_1 <= 0)
        r_d2 = np.zeros(n_vars, dtype=float)
        r_d2[2] = 1.0
        r_d2[0] = -1.0
        A_rows.append(r_d2)
        b_rows.append(0.0)
        constraint_names.append("Dep D2 (Precedence): P3 <= P1")

        # D3: P4 <= P2 (x_4 - x_2 <= 0)
        r_d3 = np.zeros(n_vars, dtype=float)
        r_d3[3] = 1.0
        r_d3[1] = -1.0
        A_rows.append(r_d3)
        b_rows.append(0.0)
        constraint_names.append("Dep D3 (Precedence): P4 <= P2")

        # D4: P13 + P14 <= 1 (Взаимоисключение архитектур)
        r_d4 = np.zeros(n_vars, dtype=float)
        r_d4[12] = 1.0
        r_d4[13] = 1.0
        A_rows.append(r_d4)
        b_rows.append(1.0)
        constraint_names.append("Dep D4 (Mutual Exclusion): P13 + P14 <= 1")

        # D5: P12 <= P5 (x_12 - x_5 <= 0)
        r_d5 = np.zeros(n_vars, dtype=float)
        r_d5[11] = 1.0
        r_d5[4] = -1.0
        A_rows.append(r_d5)
        b_rows.append(0.0)
        constraint_names.append("Dep D5 (Co-requisite): P12 <= P5")

        A_ub = np.array(A_rows, dtype=float)
        b_ub = np.array(b_rows, dtype=float)

        var_names = [f"x_{i+1}" for i in range(self.n)] + \
                    [f"y_{i+1}" for i in range(self.n)] + \
                    [f"z_{k+1}" for k in range(self.K)]

        return {
            "n_vars": n_vars,
            "c_obj": c_obj,
            "A_ub": A_ub,
            "b_ub": b_ub,
            "lb": lb,
            "ub": ub,
            "bounds": list(zip(lb, ub)),
            "integrality": integrality,
            "var_names": var_names,
            "constraint_names": constraint_names,
            "budget": budget,
            "labor": labor,
            "cost_multiplier": cost_multiplier,
            "formulation": formulation,
            "big_m_vector": M_vec,
            "model_metadata": {
                "n_projects": self.n,
                "n_clusters": self.K,
                "n_binary": self.n + self.K,
                "n_continuous": self.n,
                "n_constraints": len(b_rows)
            }
        }
