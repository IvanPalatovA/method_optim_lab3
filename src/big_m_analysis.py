#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль исследования константы Big-M (Big-M Sensitivity Analysis).
Исследует влияние величины Big-M на силу непрерывной LP-релаксации,
размер интегрального зазора (MIP gap), вычислительную устойчивость
и эффект «паразитного просачивания» (numerical leakage) при завышенных значениях M.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from src.model import ProjectPortfolioModel
from src.solver import HighsSolver


class BigMAnalyzer:
    """Анализ влияния константы Big-M на качество релаксации и скорость решения."""

    def __init__(self, raw_data: Dict[str, Any]):
        self.raw_data = raw_data
        self.model_builder = ProjectPortfolioModel(raw_data)
        self.solver = HighsSolver()

    def run_analysis(self, output_dir: str = "output") -> pd.DataFrame:
        """
        Проводит серию экспериментов с различными множителями Big-M.
        """
        os.makedirs(output_dir, exist_ok=True)

        factors = [1.0, 2.0, 5.0, 10.0, 50.0, 200.0, 1000.0, 50000.0]
        results: List[Dict[str, Any]] = []

        # Получаем базовый оптимум MILP при точном Big-M (factor = 1.0)
        base_problem = self.model_builder.build_problem(big_m_factor=1.0)
        base_milp = self.solver.solve_milp(base_problem)
        true_milp_npv = base_milp["npv"]

        for f in factors:
            # Для больших множителей используем масштабированный Big-M
            problem = self.model_builder.build_problem(big_m_factor=f)

            # Решаем LP-релаксацию
            res_lp = self.solver.solve_lp_relaxation(problem)
            lp_npv = res_lp["npv"] if res_lp["success"] else np.nan

            # Решаем MILP
            res_milp = self.solver.solve_milp(problem)
            milp_npv = res_milp["npv"] if res_milp["success"] else np.nan

            gap_pct = ((lp_npv - true_milp_npv) / true_milp_npv * 100.0) if true_milp_npv > 0 else 0.0

            # Среднее значение Big-M в модели
            avg_m = float(np.mean(problem["big_m_vector"]))

            # Суммарная дробность бинарных переменных в LP-релаксации
            x_lp = res_lp["x"]
            if x_lp is not None:
                n_proj = self.model_builder.n
                x_bin_lp = x_lp[0:n_proj]
                # Мера дробности: sum(4 * x * (1 - x))
                fractionality = float(np.sum(4.0 * x_bin_lp * (1.0 - x_bin_lp)))
                avg_x_val = float(np.mean(x_bin_lp))
            else:
                fractionality = np.nan
                avg_x_val = np.nan

            results.append({
                "BigM_Factor": f,
                "Avg_BigM_Value": round(avg_m, 1),
                "LP_NPV_MRub": round(lp_npv, 2),
                "MILP_NPV_MRub": round(milp_npv, 2),
                "Integrality_Gap_Pct": round(gap_pct, 2),
                "Fractionality_Metric": round(fractionality, 3),
                "Avg_Binary_X_LP": round(avg_x_val, 3),
                "LP_Simplex_Iters": res_lp["nit"],
                "LP_Time_ms": round(res_lp["solve_time_sec"] * 1000, 2),
                "MILP_Time_ms": round(res_milp["solve_time_sec"] * 1000, 2),
                "MIP_Nodes": res_milp["mip_nodes"]
            })

        df = pd.DataFrame(results)
        csv_path = os.path.join(output_dir, "big_m_comparison.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8")
        return df
