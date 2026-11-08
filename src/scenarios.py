#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль сценарного анализа (Scenario Analysis).
Исследует поведение инвестиционного портфеля в трех ключевых макроэкономических условиях:
  1. Базовый сценарий (Base) - плановый бюджет и фонд кадров
  2. Дефицитный сценарий (Deficit) - секвестр бюджета на 31% и кадров на 25%
  3. Стрессовый сценарий (Stress) - падение бюджета на 49%, инфляция затрат +20%
Фиксирует структурные перестройки портфеля, изменение узких мест и динамику интегрального зазора.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from src.model import ProjectPortfolioModel
from src.solver import HighsSolver


class ScenarioManager:
    """Управление и проведение сценарного анализа инвестиционного портфеля."""

    def __init__(self, raw_data: Dict[str, Any]):
        self.raw_data = raw_data
        self.model_builder = ProjectPortfolioModel(raw_data)
        self.solver = HighsSolver()
        self.scenarios_info = raw_data["scenarios"]

    def run_all_scenarios(self, output_dir: str = "output") -> pd.DataFrame:
        """
        Запуск расчетов для всех сценариев (MILP и LP-релаксация).
        """
        os.makedirs(output_dir, exist_ok=True)
        results: List[Dict[str, Any]] = []

        for sc_key, sc_cfg in self.scenarios_info.items():
            b_limit = sc_cfg["budget"]
            w_limit = sc_cfg["labor"]
            cost_mult = sc_cfg["cost_multiplier"]

            # 1. Построение задачи
            problem = self.model_builder.build_problem(
                budget=b_limit,
                labor=w_limit,
                cost_multiplier=cost_mult,
                formulation="strong"
            )

            # 2. Решение MILP
            res_milp = self.solver.solve_milp(problem)
            # 3. Решение LP-релаксации
            res_lp = self.solver.solve_lp_relaxation(problem)

            x_milp = res_milp["x"]
            n_proj = self.model_builder.n
            n_cl = self.model_builder.K

            x_bin = x_milp[0:n_proj]
            y_cont = x_milp[n_proj:2*n_proj]
            z_bin = x_milp[2*n_proj:]

            selected_projects = [f"P{i+1}" for i in range(n_proj) if x_bin[i] > 0.5]
            active_clusters = [f"K{k+1}" for k in range(n_cl) if z_bin[k] > 0.5]

            # Фактический расход ресурсов
            used_capital = float(np.dot(self.model_builder.c_fix * cost_mult, x_bin) +
                                 np.dot(self.model_builder.c_var * cost_mult, y_cont) +
                                 np.dot(self.model_builder.F_cluster * cost_mult, z_bin))
            used_labor = float(np.dot(self.model_builder.w_fix, x_bin) +
                               np.dot(self.model_builder.w_var, y_cont) +
                               np.dot(self.model_builder.S_cluster, z_bin))

            npv_milp = res_milp["npv"]
            npv_lp = res_lp["npv"]
            int_gap = ((npv_lp - npv_milp) / npv_milp * 100.0) if npv_milp > 0 else 0.0

            row = {
                "Scenario": sc_key,
                "Name": sc_cfg["name"],
                "Budget_Limit_MRub": b_limit,
                "Labor_Limit_ManMonths": w_limit,
                "Cost_Multiplier": cost_mult,
                "MILP_NPV_MRub": round(npv_milp, 2),
                "LP_NPV_MRub": round(npv_lp, 2),
                "Integrality_Gap_Pct": round(int_gap, 2),
                "Selected_Count": len(selected_projects),
                "Selected_Projects": ", ".join(selected_projects),
                "Active_Clusters_Count": len(active_clusters),
                "Active_Clusters": ", ".join(active_clusters),
                "Used_Capital_MRub": round(used_capital, 2),
                "Capital_Utilization_Pct": round(used_capital / b_limit * 100.0, 1),
                "Used_Labor_ManMonths": round(used_labor, 2),
                "Labor_Utilization_Pct": round(used_labor / w_limit * 100.0, 1),
                "MILP_Time_ms": round(res_milp["solve_time_sec"] * 1000, 2),
                "MIP_Nodes": res_milp["mip_nodes"],
                "Budget_Shadow_Price": round(res_lp["budget_shadow_price"], 4),
                "Labor_Shadow_Price": round(res_lp["labor_shadow_price"], 4)
            }
            results.append(row)

        df = pd.DataFrame(results)
        csv_path = os.path.join(output_dir, "scenarios_comparison.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8")
        return df
