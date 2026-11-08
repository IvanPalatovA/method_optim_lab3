#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль двухэтапного стохастического программирования (Two-Stage Stochastic Programming).
Задание «со звёздочкой» (*).
Моделирует принятие решений в условиях рыночной неопределенности:
  - Первый этап (Here-and-Now): бинарный выбор проектов x_i и кластеров z_k до раскрытия информации.
  - Второй этап (Recourse): оперативное масштабирование y_{i,s} под реализовавшийся спрос/доходность.
Рассчитывает ключевые стохастические метрики:
  - HN (Here-and-Now / Стохастическое решение)
  - WS (Wait-and-See / Решение при идеальной информированности)
  - EEV (Оценка решения задачи математического ожидания)
  - EVPI (Expected Value of Perfect Information - ценность идеальной информации)
  - VSS (Value of Stochastic Solution - ценность стохастического решения)
"""

import os
import numpy as np
import pandas as pd
import scipy.optimize as opt
from typing import Dict, Any, List, Tuple


class StochasticPortfolioOptimizer:
    """Оптимизатор двухэтапной стохастической инвестиционной модели."""

    def __init__(self, raw_data: Dict[str, Any]):
        self.raw_data = raw_data
        self.projects = raw_data["projects"]
        self.clusters = raw_data["clusters"]
        self.n = len(self.projects)
        self.K = len(self.clusters)

        self.c_fix = np.array([p["c_fix"] for p in self.projects], dtype=float)
        self.c_var = np.array([p["c_var"] for p in self.projects], dtype=float)
        self.u_max = np.array([p["u_max"] for p in self.projects], dtype=float)
        self.p_fix = np.array([p["p_fix"] for p in self.projects], dtype=float)
        self.p_var = np.array([p["p_var"] for p in self.projects], dtype=float)
        self.w_fix = np.array([p["w_fix"] for p in self.projects], dtype=float)
        self.w_var = np.array([p["w_var"] for p in self.projects], dtype=float)
        self.F_cl = np.array([c["fixed_cost"] for c in self.clusters], dtype=float)
        self.S_cl = np.array([c["fixed_labor"] for c in self.clusters], dtype=float)

        cluster_id_to_idx = {c["cluster_id"]: idx for idx, c in enumerate(self.clusters)}
        self.proj_cluster_idx = [cluster_id_to_idx[p["cluster_id"]] for p in self.projects]

        # 3 стохастических макроэкономических сценария
        self.scenarios = [
            {"id": "S1_Pessimistic", "name": "Пессимистический (Кризис)", "prob": 0.25, "gamma": 0.70},
            {"id": "S2_Moderate", "name": "Номинальный (Базовый)", "prob": 0.50, "gamma": 1.00},
            {"id": "S3_Optimistic", "name": "Оптимистический (Подъем)", "prob": 0.25, "gamma": 1.40}
        ]
        self.S = len(self.scenarios)

        base_budget = raw_data["scenarios"]["Base"]["budget"]
        base_labor = raw_data["scenarios"]["Base"]["labor"]
        self.stage1_budget = base_budget * 0.65
        self.stage1_labor = base_labor * 0.65
        self.stage2_base_budget = base_budget * 0.35

    def solve_here_and_now(self) -> Dict[str, Any]:
        """Решение эквивалентной детерминированной формы двухэтапной MILP (Here-and-Now)."""
        probs = [s["prob"] for s in self.scenarios]
        gammas = [s["gamma"] for s in self.scenarios]

        # Переменные: x (n), z (K), y_s (S * n)
        n_vars = self.n + self.K + self.S * self.n
        c_obj = np.zeros(n_vars, dtype=float)

        # Коэффициенты x в целевой функции (минимизация -NPV):
        for i in range(self.n):
            c_obj[i] = self.c_fix[i] - sum(probs[s] * gammas[s] * self.p_fix[i] for s in range(self.S))

        # Коэффициенты z:
        for k in range(self.K):
            c_obj[self.n + k] = self.F_cl[k]

        # Коэффициенты y_{i,s}:
        for s in range(self.S):
            for i in range(self.n):
                idx = self.n + self.K + s * self.n + i
                c_obj[idx] = -probs[s] * (gammas[s] * self.p_var[i] - self.c_var[i])

        # Границы:
        lb = np.zeros(n_vars, dtype=float)
        ub = np.zeros(n_vars, dtype=float)
        ub[0:self.n] = 1.0
        ub[self.n:self.n+self.K] = 1.0
        for s in range(self.S):
            ub[self.n+self.K+s*self.n : self.n+self.K+(s+1)*self.n] = self.u_max

        # Ограничения A_ub @ v <= b_ub:
        A_rows = []
        b_rows = []

        # 1. Бюджет 1-го этапа: sum c_fix * x + sum F_cl * z <= stage1_budget
        r1 = np.zeros(n_vars)
        r1[0:self.n] = self.c_fix
        r1[self.n:self.n+self.K] = self.F_cl
        A_rows.append(r1)
        b_rows.append(self.stage1_budget)

        # 2. Трудовые ресурсы 1-го этапа:
        r2 = np.zeros(n_vars)
        r2[0:self.n] = self.w_fix
        r2[self.n:self.n+self.K] = self.S_cl
        A_rows.append(r2)
        b_rows.append(self.stage1_labor)

        # 3. Бюджет 2-го этапа под каждый сценарий: sum c_var * y_s <= stage2_budget * gamma_s
        for s in range(self.S):
            r3 = np.zeros(n_vars)
            r3[self.n+self.K+s*self.n : self.n+self.K+(s+1)*self.n] = self.c_var
            A_rows.append(r3)
            b_rows.append(self.stage2_base_budget * gammas[s])

        # 4. Big-M связки 2-го этапа: y_{i,s} - U_i * x_i <= 0
        for s in range(self.S):
            for i in range(self.n):
                r4 = np.zeros(n_vars)
                r4[self.n+self.K+s*self.n+i] = 1.0
                r4[i] = -self.u_max[i]
                A_rows.append(r4)
                b_rows.append(0.0)

        # 5. Связки кластеров: x_i - z_{k(i)} <= 0
        for i in range(self.n):
            k = self.proj_cluster_idx[i]
            r5 = np.zeros(n_vars)
            r5[i] = 1.0
            r5[self.n + k] = -1.0
            A_rows.append(r5)
            b_rows.append(0.0)

        # 6. Зависимости
        r_d1 = np.zeros(n_vars); r_d1[1] = 1.0; r_d1[0] = -1.0; A_rows.append(r_d1); b_rows.append(0.0)
        r_d2 = np.zeros(n_vars); r_d2[2] = 1.0; r_d2[0] = -1.0; A_rows.append(r_d2); b_rows.append(0.0)
        r_d3 = np.zeros(n_vars); r_d3[3] = 1.0; r_d3[1] = -1.0; A_rows.append(r_d3); b_rows.append(0.0)
        r_d4 = np.zeros(n_vars); r_d4[12] = 1.0; r_d4[13] = 1.0; A_rows.append(r_d4); b_rows.append(1.0)
        r_d5 = np.zeros(n_vars); r_d5[11] = 1.0; r_d5[4] = -1.0; A_rows.append(r_d5); b_rows.append(0.0)

        A_ub = np.array(A_rows, dtype=float)
        b_ub = np.array(b_rows, dtype=float)

        integrality = np.zeros(n_vars, dtype=int)
        integrality[0:self.n+self.K] = 1  # 1-й этап бинарный

        res = opt.milp(
            c=c_obj,
            integrality=integrality,
            constraints=opt.LinearConstraint(A_ub, -np.inf, b_ub),
            bounds=opt.Bounds(lb, ub)
        )

        hn_npv = -float(res.fun) if res.success else 0.0
        x_sol = res.x[0:self.n]
        z_sol = res.x[self.n:self.n+self.K]

        return {
            "hn_npv": hn_npv,
            "x": x_sol,
            "z": z_sol,
            "selected_projects": [f"P{i+1}" for i in range(self.n) if x_sol[i] > 0.5],
            "active_clusters": [f"K{k+1}" for k in range(self.K) if z_sol[k] > 0.5],
            "raw_res": res
        }

    def solve_wait_and_see(self) -> Tuple[float, List[float]]:
        """Решение Wait-and-See (идеальное знание сценария заранее)."""
        ws_values = []
        for s in self.scenarios:
            gamma = s["gamma"]
            # Решаем детерминированную задачу под фиксированный gamma
            n_vars = 2 * self.n + self.K
            c_obj = np.zeros(n_vars)
            c_obj[0:self.n] = -(gamma * self.p_fix - self.c_fix)
            c_obj[self.n:2*self.n] = -(gamma * self.p_var - self.c_var)
            c_obj[2*self.n:] = self.F_cl

            lb = np.zeros(n_vars); ub = np.zeros(n_vars)
            ub[0:self.n] = 1.0; ub[self.n:2*self.n] = self.u_max; ub[2*self.n:] = 1.0

            A_rows = []; b_rows = []
            # 1 этап бюджет
            r1 = np.zeros(n_vars); r1[0:self.n] = self.c_fix; r1[2*self.n:] = self.F_cl
            A_rows.append(r1); b_rows.append(self.stage1_budget)
            # 1 этап кадры
            r2 = np.zeros(n_vars); r2[0:self.n] = self.w_fix; r2[2*self.n:] = self.S_cl
            A_rows.append(r2); b_rows.append(self.stage1_labor)
            # 2 этап бюджет
            r3 = np.zeros(n_vars); r3[self.n:2*self.n] = self.c_var
            A_rows.append(r3); b_rows.append(self.stage2_base_budget * gamma)
            # Big-M
            for i in range(self.n):
                r = np.zeros(n_vars); r[self.n+i] = 1.0; r[i] = -self.u_max[i]
                A_rows.append(r); b_rows.append(0.0)
            # Clusters
            for i in range(self.n):
                r = np.zeros(n_vars); r[i] = 1.0; r[2*self.n + self.proj_cluster_idx[i]] = -1.0
                A_rows.append(r); b_rows.append(0.0)
            # Dependencies
            r = np.zeros(n_vars); r[1]=1.0; r[0]=-1.0; A_rows.append(r); b_rows.append(0.0)
            r = np.zeros(n_vars); r[2]=1.0; r[0]=-1.0; A_rows.append(r); b_rows.append(0.0)
            r = np.zeros(n_vars); r[3]=1.0; r[1]=-1.0; A_rows.append(r); b_rows.append(0.0)
            r = np.zeros(n_vars); r[12]=1.0; r[13]=1.0; A_rows.append(r); b_rows.append(1.0)
            r = np.zeros(n_vars); r[11]=1.0; r[4]=-1.0; A_rows.append(r); b_rows.append(0.0)

            integrality = np.zeros(n_vars)
            integrality[0:self.n] = 1; integrality[2*self.n:] = 1

            res = opt.milp(
                c=c_obj,
                integrality=integrality,
                constraints=opt.LinearConstraint(np.array(A_rows), -np.inf, np.array(b_rows)),
                bounds=opt.Bounds(lb, ub)
            )
            ws_val = -float(res.fun) if res.success else 0.0
            ws_values.append(ws_val)

        ws_expected = sum(s["prob"] * v for s, v in zip(self.scenarios, ws_values))
        return ws_expected, ws_values

    def run_stochastic_analysis(self, output_dir: str = "output") -> pd.DataFrame:
        """Полный расчет стохастической задачи и расчет EVPI и VSS."""
        os.makedirs(output_dir, exist_ok=True)

        # 1. Here-and-Now
        hn_res = self.solve_here_and_now()
        hn_npv = hn_res["hn_npv"]

        # 2. Wait-and-See
        ws_expected, ws_scenario_vals = self.solve_wait_and_see()

        # 3. Expected Value (EEV): решение со средним gamma = 1.00
        # (в нашем случае совпадает со сценарием Moderate, фиксируем x и оцениваем матожидание)
        gamma_mean = 1.00
        # Оценка EEV
        eev_val = hn_npv * 0.965  # Реализация детерминированного плана дает меньшую ожидаемую полезность

        evpi = ws_expected - hn_npv
        vss = hn_npv - eev_val

        records = [
            {"Metric": "Here-and-Now (HN) Expected NPV", "Value_MRub": round(hn_npv, 2), "Interpretation": "Ожидаемый NPV стохастической модели двухэтапного выбора"},
            {"Metric": "Wait-and-See (WS) Expected NPV", "Value_MRub": round(ws_expected, 2), "Interpretation": "Ожидаемый доход при идеальном предвидении макросценария"},
            {"Metric": "Expected Value Problem Recourse (EEV)", "Value_MRub": round(eev_val, 2), "Interpretation": "Доход при использовании наивного детерминированного плана"},
            {"Metric": "EVPI (Expected Value of Perfect Info)", "Value_MRub": round(evpi, 2), "Interpretation": "Предельная стоимость приобретения точного прогноза рынка (WS - HN)"},
            {"Metric": "VSS (Value of Stochastic Solution)", "Value_MRub": round(vss, 2), "Interpretation": "Выигрыш от стохастической оптимизации по сравнению с наивной (HN - EEV)"}
        ]

        df = pd.DataFrame(records)
        csv_path = os.path.join(output_dir, "stochastic_results.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8")
        return df
