#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль независимой верификации решения (NumPy Verification).
Выполняет полную проверку допустимости найденного решения без обращения
к внутренним объектам солвера: проверка целочисленности бинарных переменных,
соблюдения ресурсных лимитов, связок Big-M, активации кластеров, всех 5 зависимостей
и независимый пересчет значения целевой функции с допуском 1e-7.
"""

import numpy as np
from typing import Dict, Any, Tuple, List


class SolutionVerifier:
    """Класс независимой проверки допустимости и точности решения."""

    def __init__(self, raw_data: Dict[str, Any], tol: float = 1e-7):
        self.raw_data = raw_data
        self.projects = raw_data["projects"]
        self.clusters = raw_data["clusters"]
        self.dependencies = raw_data["dependencies"]
        self.tol = tol

        self.n = len(self.projects)
        self.K = len(self.clusters)

        # Подготовка векторов коэффициентов
        self.c_fix = np.array([p["c_fix"] for p in self.projects], dtype=float)
        self.c_var = np.array([p["c_var"] for p in self.projects], dtype=float)
        self.u_max = np.array([p["u_max"] for p in self.projects], dtype=float)

        self.p_fix = np.array([p["p_fix"] for p in self.projects], dtype=float)
        self.p_var = np.array([p["p_var"] for p in self.projects], dtype=float)

        self.w_fix = np.array([p["w_fix"] for p in self.projects], dtype=float)
        self.w_var = np.array([p["w_var"] for p in self.projects], dtype=float)

        self.F_cluster = np.array([c["fixed_cost"] for c in self.clusters], dtype=float)
        self.S_cluster = np.array([c["fixed_labor"] for c in self.clusters], dtype=float)

        cluster_id_to_idx = {c["cluster_id"]: idx for idx, c in enumerate(self.clusters)}
        self.proj_cluster_idx = [cluster_id_to_idx[p["cluster_id"]] for p in self.projects]

    def verify_solution(
        self,
        x_vec: np.ndarray,
        solver_npv: float,
        budget: float,
        labor: float,
        cost_multiplier: float = 1.0,
        formulation: str = "strong",
        big_m_factor: float = 1.0
    ) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Полная проверка вектора решения x_vec.
        :return: (is_valid, report_lines, metrics_dict)
        """
        report = []
        is_valid = True

        x = x_vec[0:self.n]
        y = x_vec[self.n:2*self.n]
        z = x_vec[2*self.n:]

        report.append("=" * 78)
        report.append("  НЕЗАВИСИМАЯ ВЕРИФИКАЦИЯ РЕШЕНИЯ (NUMPY VERIFICATION)")
        report.append(f"  Формулировка: {formulation.upper()} | Допуск (tolerance): {self.tol}")
        report.append("=" * 78)

        # 1. Проверка бинарной целочисленности x и z
        x_int_violation = np.max(np.abs(x * (1.0 - x)))
        z_int_violation = np.max(np.abs(z * (1.0 - z)))
        int_ok = (x_int_violation <= self.tol) and (z_int_violation <= self.tol)
        if not int_ok:
            is_valid = False
        report.append(f"1. Целочисленность x in {{0,1}} : {'[OK]' if x_int_violation <= self.tol else '[FAIL]'} (макс. отклонение: {x_int_violation:.2e})")
        report.append(f"   Целочисленность z in {{0,1}} : {'[OK]' if z_int_violation <= self.tol else '[FAIL]'} (макс. отклонение: {z_int_violation:.2e})")

        # 2. Проверка неотрицательности и границ
        lb_violation = max(0.0, -np.min(x_vec))
        ub_y_violation = max(0.0, np.max(y - self.u_max))
        bounds_ok = (lb_violation <= self.tol) and (ub_y_violation <= self.tol)
        if not bounds_ok:
            is_valid = False
        report.append(f"2. Границы переменных        : {'[OK]' if bounds_ok else '[FAIL]'} (отрицательность: {lb_violation:.2e}, превышение U_i: {ub_y_violation:.2e})")

        # 3. Связка Big-M: y_i <= M_i * x_i
        M_vec = self.u_max * big_m_factor
        big_m_diff = y - M_vec * x
        max_big_m_viol = max(0.0, float(np.max(big_m_diff)))
        big_m_ok = max_big_m_viol <= self.tol
        if not big_m_ok:
            is_valid = False
        report.append(f"3. Ограничения Big-M (y <= Mx): {'[OK]' if big_m_ok else '[FAIL]'} (макс. нарушение: {max_big_m_viol:.2e})")

        # 4. Активация кластерной инфраструктуры
        if formulation.lower() == "strong":
            cl_diffs = [x[i] - z[self.proj_cluster_idx[i]] for i in range(self.n)]
            max_cl_viol = max(0.0, float(np.max(cl_diffs)))
        else:
            cl_diffs = []
            for k in range(self.K):
                members = [i for i in range(self.n) if self.proj_cluster_idx[i] == k]
                cl_diffs.append(np.sum(x[members]) - len(members) * z[k])
            max_cl_viol = max(0.0, float(np.max(cl_diffs)))
        cl_ok = max_cl_viol <= self.tol
        if not cl_ok:
            is_valid = False
        report.append(f"4. Активация кластеров       : {'[OK]' if cl_ok else '[FAIL]'} (макс. нарушение: {max_cl_viol:.2e})")

        # 5. Проверка 5 структурных зависимостей
        # D1: P2 <= P1
        v_d1 = max(0.0, float(x[1] - x[0]))
        # D2: P3 <= P1
        v_d2 = max(0.0, float(x[2] - x[0]))
        # D3: P4 <= P2
        v_d3 = max(0.0, float(x[3] - x[1]))
        # D4: P13 + P14 <= 1
        v_d4 = max(0.0, float(x[12] + x[13] - 1.0))
        # D5: P12 <= P5
        v_d5 = max(0.0, float(x[11] - x[4]))

        deps_ok = all(v <= self.tol for v in [v_d1, v_d2, v_d3, v_d4, v_d5])
        if not deps_ok:
            is_valid = False
        report.append(f"5. Структурные зависимости   : {'[OK]' if deps_ok else '[FAIL]'}")
        report.append(f"   - D1 (P2 <= P1, x2-x1<=0) : {'[OK]' if v_d1 <= self.tol else '[FAIL]'} (нарушение: {v_d1:.2e})")
        report.append(f"   - D2 (P3 <= P1, x3-x1<=0) : {'[OK]' if v_d2 <= self.tol else '[FAIL]'} (нарушение: {v_d2:.2e})")
        report.append(f"   - D3 (P4 <= P2, x4-x2<=0) : {'[OK]' if v_d3 <= self.tol else '[FAIL]'} (нарушение: {v_d3:.2e})")
        report.append(f"   - D4 (P13+P14<=1)         : {'[OK]' if v_d4 <= self.tol else '[FAIL]'} (нарушение: {v_d4:.2e})")
        report.append(f"   - D5 (P12 <= P5, x12-x5<=0): {'[OK]' if v_d5 <= self.tol else '[FAIL]'} (нарушение: {v_d5:.2e})")

        # 6. Бюджетные ограничения
        curr_c_fix = self.c_fix * cost_multiplier
        curr_c_var = self.c_var * cost_multiplier
        curr_F_cl = self.F_cluster * cost_multiplier

        calc_capital = float(np.dot(curr_c_fix, x) + np.dot(curr_c_var, y) + np.dot(curr_F_cl, z))
        capital_slack = budget - calc_capital
        capital_ok = capital_slack >= -self.tol
        if not capital_ok:
            is_valid = False
        report.append(f"6. Капитальный бюджет        : {'[OK]' if capital_ok else '[FAIL]'} (израсходовано: {calc_capital:.2f} из {budget:.2f} млн руб., запас: {capital_slack:.2f})")

        # 7. Трудовые ресурсы
        calc_labor = float(np.dot(self.w_fix, x) + np.dot(self.w_var, y) + np.dot(self.S_cluster, z))
        labor_slack = labor - calc_labor
        labor_ok = labor_slack >= -self.tol
        if not labor_ok:
            is_valid = False
        report.append(f"7. Трудовые ресурсы          : {'[OK]' if labor_ok else '[FAIL]'} (израсходовано: {calc_labor:.2f} из {labor:.2f} чел-мес, запас: {labor_slack:.2f})")

        # 8. Независимый пересчет целевой функции NPV
        calc_npv = float(np.dot(self.p_fix, x) + np.dot(self.p_var, y) - np.dot(curr_F_cl, z))
        npv_diff = abs(calc_npv - solver_npv)
        npv_ok = npv_diff <= self.tol
        if not npv_ok:
            is_valid = False
        report.append(f"8. Сверка целевой функции    : {'[OK]' if npv_ok else '[FAIL]'}")
        report.append(f"   - Расчет NumPy            : {calc_npv:.6f} млн руб.")
        report.append(f"   - Значение солвера HiGHS  : {solver_npv:.6f} млн руб.")
        report.append(f"   - Абсолютная разница      : {npv_diff:.2e} (порог: {self.tol})")

        report.append("-" * 78)
        report.append(f"ИТОГ ВЕРИФИКАЦИИ: {'РЕШЕНИЕ СТРОГО ДОПУСТИМО И СВЕРЕНО' if is_valid else 'ОБНАРУЖЕНЫ НАРУШЕНИЯ'}")
        report.append("=" * 78)

        metrics = {
            "is_valid": is_valid,
            "calc_npv": calc_npv,
            "solver_npv": solver_npv,
            "npv_diff": npv_diff,
            "calc_capital": calc_capital,
            "capital_slack": capital_slack,
            "calc_labor": calc_labor,
            "labor_slack": labor_slack,
            "x_int_violation": x_int_violation,
            "z_int_violation": z_int_violation,
            "max_big_m_viol": max_big_m_viol,
            "selected_projects_count": int(np.sum(x > 0.5)),
            "active_clusters_count": int(np.sum(z > 0.5))
        }

        return is_valid, report, metrics
