#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Интерфейс к промышленному решателю HiGHS через SciPy (scipy.optimize.milp и linprog).
Обеспечивает точное решение непрерывных LP-релаксаций и дискретных MILP-задач,
замер времени выполнения, сбор статистики ветвления (узлы, зазор) и логирование.
"""

import time
import numpy as np
import scipy.optimize as opt
from typing import Dict, Any, Optional


class HighsSolver:
    """Обертка над промышленным решателем HiGHS."""

    def __init__(self, presolve: bool = True, time_limit: float = 60.0):
        self.presolve = presolve
        self.time_limit = time_limit

    def solve_milp(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Решение задачи в смешанно-целочисленной постановке (MILP) через HiGHS.
        """
        c_obj = problem["c_obj"]
        A_ub = problem["A_ub"]
        b_ub = problem["b_ub"]
        lb = problem["lb"]
        ub = problem["ub"]
        integrality = problem["integrality"]
        var_names = problem["var_names"]

        constraints = [opt.LinearConstraint(A_ub, -np.inf, b_ub)]
        bounds = opt.Bounds(lb, ub)

        options = {
            "disp": False,
            "presolve": self.presolve,
            "time_limit": self.time_limit
        }

        t_start = time.perf_counter()
        res = opt.milp(
            c=c_obj,
            integrality=integrality,
            constraints=constraints,
            bounds=bounds,
            options=options
        )
        t_sec = time.perf_counter() - t_start

        success = res.success and (res.status == 0)
        # c_obj минимизировался (-NPV), поэтому действительный максимум = -res.fun
        npv_obj = -float(res.fun) if success else None

        mip_nodes = getattr(res, "mip_node_count", 0)
        mip_gap = getattr(res, "mip_gap", 0.0)
        mip_dual_bound = -float(getattr(res, "mip_dual_bound", 0.0)) if getattr(res, "mip_dual_bound", None) is not None else None

        log_lines = [
            "=" * 78,
            f"  HiGHS MILP SOLVER LOG | Formulation: {problem.get('formulation', 'strong').upper()}",
            "=" * 78,
            f"Status code      : {res.status}",
            f"Solver message   : {res.message}",
            f"Success flag     : {success}",
            f"Solve time       : {t_sec * 1000:.3f} ms ({t_sec:.6f} s)",
            f"Objective (NPV)  : {f'{npv_obj:.4f} млн руб.' if success else 'N/A'}",
            f"MIP Node count   : {mip_nodes}",
            f"MIP Dual bound   : {f'{mip_dual_bound:.4f}' if mip_dual_bound is not None else 'N/A'}",
            f"MIP Branch Gap   : {f'{mip_gap * 100:.4f}%' if mip_gap is not None else 'N/A'}",
            "-" * 78
        ]

        if success:
            n_proj = problem["model_metadata"]["n_projects"]
            n_cl = problem["model_metadata"]["n_clusters"]
            x_vals = res.x[0:n_proj]
            y_vals = res.x[n_proj:2*n_proj]
            z_vals = res.x[2*n_proj:]

            selected_projects = [f"P{i+1}" for i in range(n_proj) if x_vals[i] > 0.5]
            active_clusters = [f"K{k+1}" for k in range(n_cl) if z_vals[k] > 0.5]

            log_lines.append(f"Selected projects ({len(selected_projects)} of {n_proj}): {', '.join(selected_projects)}")
            log_lines.append(f"Active clusters   ({len(active_clusters)} of {n_cl}): {', '.join(active_clusters)}")
            log_lines.append(f"Active scaling vars: {int(np.sum(y_vals > 1e-4))} variables")

        log_lines.append("=" * 78)
        log_text = "\n".join(log_lines)

        return {
            "status": res.status,
            "success": success,
            "message": res.message,
            "npv": npv_obj,
            "fun": float(res.fun) if success else None,
            "x": res.x if success else None,
            "solve_time_sec": t_sec,
            "mip_nodes": mip_nodes,
            "mip_gap": mip_gap,
            "mip_dual_bound": mip_dual_bound,
            "log": log_text,
            "raw_res": res
        }

    def solve_lp_relaxation(self, problem: Dict[str, Any], method: str = "highs-ds") -> Dict[str, Any]:
        """
        Решение непрерывной LP-релаксации (без требований целочисленности) через HiGHS.
        """
        c_obj = problem["c_obj"]
        A_ub = problem["A_ub"]
        b_ub = problem["b_ub"]
        bounds = problem["bounds"]

        options = {
            "disp": False,
            "presolve": self.presolve,
            "time_limit": self.time_limit
        }

        t_start = time.perf_counter()
        res = opt.linprog(
            c=c_obj,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method=method,
            options=options
        )
        t_sec = time.perf_counter() - t_start

        success = res.success and (res.status == 0)
        npv_obj = -float(res.fun) if success else None

        # Двойственные переменные (теневые цены)
        ineq_marginals = np.array([])
        if hasattr(res, "ineqlin") and res.ineqlin is not None and hasattr(res.ineqlin, "marginals"):
            ineq_marginals = np.array(res.ineqlin.marginals)

        budget_shadow_price = float(ineq_marginals[0]) if len(ineq_marginals) > 0 else 0.0
        labor_shadow_price = float(ineq_marginals[1]) if len(ineq_marginals) > 1 else 0.0

        log_lines = [
            "=" * 78,
            f"  HiGHS LP RELAXATION LOG | Method: {method} | Presolve: {self.presolve}",
            "=" * 78,
            f"Status code      : {res.status}",
            f"Solver message   : {res.message}",
            f"Success flag     : {success}",
            f"Simplex iters    : {getattr(res, 'nit', 0)}",
            f"Solve time       : {t_sec * 1000:.3f} ms ({t_sec:.6f} s)",
            f"Objective (NPV)  : {f'{npv_obj:.4f} млн руб.' if success else 'N/A'}",
            f"Budget shadow pr : {budget_shadow_price:.4f} руб/руб",
            f"Labor shadow pr  : {labor_shadow_price:.4f} руб/чел-мес",
            "=" * 78
        ]
        log_text = "\n".join(log_lines)

        return {
            "status": res.status,
            "success": success,
            "message": res.message,
            "npv": npv_obj,
            "fun": float(res.fun) if success else None,
            "x": res.x if success else None,
            "nit": getattr(res, "nit", 0),
            "solve_time_sec": t_sec,
            "ineq_marginals": ineq_marginals,
            "budget_shadow_price": budget_shadow_price,
            "labor_shadow_price": labor_shadow_price,
            "log": log_text,
            "raw_res": res
        }
