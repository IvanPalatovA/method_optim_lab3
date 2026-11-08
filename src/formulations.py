#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль сравнения двух эквивалентных формулировок MILP (Formulation Strength Comparison).
Сравнивает:
  1. Сильную формулировку (Strong formulation: дезагрегированные связки x_i <= z_k)
  2. Слабую формулировку (Weak formulation: агрегированные связки sum x_i <= |C_k| * z_k)
Демонстрирует теоретическое превосходство многогранника сильной формулировки:
P_Strong строго содержится в P_Weak, что обеспечивает более плотную верхнюю грань
и сокращает объем дерева ветвлений Branch-and-Bound.
"""

import os
import pandas as pd
from typing import Dict, Any, List
from src.model import ProjectPortfolioModel
from src.solver import HighsSolver


class FormulationComparator:
    """Сравнение силы математических формулировок MILP."""

    def __init__(self, raw_data: Dict[str, Any]):
        self.raw_data = raw_data
        self.model_builder = ProjectPortfolioModel(raw_data)
        self.solver = HighsSolver()

    def compare_formulations(self, output_dir: str = "output") -> pd.DataFrame:
        """
        Проводит сопоставительный анализ сильной и слабой постановок.
        """
        os.makedirs(output_dir, exist_ok=True)
        results: List[Dict[str, Any]] = []

        for form in ["strong", "weak"]:
            problem = self.model_builder.build_problem(formulation=form)

            res_milp = self.solver.solve_milp(problem)
            res_lp = self.solver.solve_lp_relaxation(problem)

            npv_milp = res_milp["npv"]
            npv_lp = res_lp["npv"]
            gap_pct = ((npv_lp - npv_milp) / npv_milp * 100.0) if npv_milp > 0 else 0.0

            results.append({
                "Formulation": "Strong (Дезагрегированная: x_i <= z_k)" if form == "strong" else "Weak (Агрегированная: sum x_i <= |C_k|*z_k)",
                "Formulation_Key": form,
                "Num_Constraints": problem["A_ub"].shape[0],
                "Num_Variables": problem["n_vars"],
                "LP_NPV_MRub": round(npv_lp, 2),
                "MILP_NPV_MRub": round(npv_milp, 2),
                "Integrality_Gap_Pct": round(gap_pct, 2),
                "LP_Simplex_Iters": res_lp["nit"],
                "LP_Time_ms": round(res_lp["solve_time_sec"] * 1000, 2),
                "MILP_Time_ms": round(res_milp["solve_time_sec"] * 1000, 2),
                "MIP_Nodes": res_milp["mip_nodes"],
                "HiGHS_Log_Snippet": res_milp["log"][:300].replace("\n", " | ")
            })

        df = pd.DataFrame(results)
        csv_path = os.path.join(output_dir, "formulations_comparison.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8")
        return df
