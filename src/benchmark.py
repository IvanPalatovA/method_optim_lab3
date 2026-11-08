#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль нагрузочного тестирования и масштабируемости (Scaling Benchmark).
Исследует зависимость времени решения, числа узлов дерева Branch-and-Bound,
симплекс-итераций и зазоров (Integrality Gap, Greedy Gap) от размерности задачи:
от N = 10 проектов до N = 160 проектов (до 350+ переменных оптимизации).
"""

import os
import time
import numpy as np
import pandas as pd
import scipy.optimize as opt
from typing import Dict, Any, List


class ScalingBenchmarkRunner:
    """Проведение масштабируемых вычислительных экспериментов."""

    def __init__(self, base_seed: int = 10116):
        self.base_seed = base_seed

    def run_benchmark(
        self,
        dimensions: List[int] = [10, 15, 20, 30, 45, 60, 90, 120, 160],
        output_dir: str = "output"
    ) -> pd.DataFrame:
        """
        Запуск бенчмарка по списку размерностей N.
        """
        os.makedirs(output_dir, exist_ok=True)
        results: List[Dict[str, Any]] = []

        for n in dimensions:
            rng = np.random.default_rng(self.base_seed + n)
            K = max(2, n // 5)
            clusters = rng.integers(0, K, size=n)

            c_fix = rng.integers(15, 55, size=n).astype(float)
            c_var = np.round(rng.uniform(1.5, 4.0, size=n), 2)
            u_max = rng.integers(5, 16, size=n).astype(float)

            p_fix = rng.integers(25, 90, size=n).astype(float)
            p_var = np.round(rng.uniform(3.0, 8.5, size=n), 2)

            w_fix = rng.integers(6, 25, size=n).astype(float)
            w_var = np.round(rng.uniform(0.8, 2.5, size=n), 2)

            F_cl = rng.uniform(25.0, 45.0, size=K)
            S_cl = rng.uniform(8.0, 18.0, size=K)

            tot_cost = np.sum(c_fix + c_var * u_max) + np.sum(F_cl)
            tot_labor = np.sum(w_fix + w_var * u_max) + np.sum(S_cl)

            b_limit = float(np.round(tot_cost * 0.55, 0))
            w_limit = float(np.round(tot_labor * 0.60, 0))

            n_vars = 2 * n + K
            c_obj = np.zeros(n_vars)
            c_obj[0:n] = -p_fix
            c_obj[n:2*n] = -p_var
            c_obj[2*n:] = F_cl

            lb = np.zeros(n_vars); ub = np.zeros(n_vars)
            ub[0:n] = 1.0; ub[n:2*n] = u_max; ub[2*n:] = 1.0

            A_rows = []; b_rows = []
            # Budget
            r_b = np.zeros(n_vars); r_b[0:n] = c_fix; r_b[n:2*n] = c_var; r_b[2*n:] = F_cl
            A_rows.append(r_b); b_rows.append(b_limit)
            # Labor
            r_l = np.zeros(n_vars); r_l[0:n] = w_fix; r_l[n:2*n] = w_var; r_l[2*n:] = S_cl
            A_rows.append(r_l); b_rows.append(w_limit)
            # Big-M
            for i in range(n):
                r_m = np.zeros(n_vars); r_m[n+i] = 1.0; r_m[i] = -u_max[i]
                A_rows.append(r_m); b_rows.append(0.0)
            # Strong Cluster links
            for i in range(n):
                r_c = np.zeros(n_vars); r_c[i] = 1.0; r_c[2*n + clusters[i]] = -1.0
                A_rows.append(r_c); b_rows.append(0.0)
            # Precedence dependencies (цепочки предшествования)
            n_deps = max(1, n // 3)
            for d in range(n_deps):
                r_d = np.zeros(n_vars)
                r_d[d + 1] = 1.0
                r_d[d] = -1.0
                A_rows.append(r_d)
                b_rows.append(0.0)

            A_ub = np.array(A_rows, dtype=float)
            b_ub = np.array(b_rows, dtype=float)

            integrality = np.zeros(n_vars, dtype=int)
            integrality[0:n] = 1
            integrality[2*n:] = 1

            # 1. Запуск MILP HiGHS
            t0 = time.perf_counter()
            res_m = opt.milp(
                c=c_obj,
                integrality=integrality,
                constraints=opt.LinearConstraint(A_ub, -np.inf, b_ub),
                bounds=opt.Bounds(lb, ub)
            )
            t_milp = (time.perf_counter() - t0) * 1000.0

            # 2. Запуск непрерывной LP-релаксации
            t0 = time.perf_counter()
            res_lp = opt.linprog(
                c=c_obj,
                A_ub=A_ub,
                b_ub=b_ub,
                bounds=list(zip(lb, ub)),
                method="highs-ds"
            )
            t_lp = (time.perf_counter() - t0) * 1000.0

            # 3. Запуск жадной эвристики
            t0 = time.perf_counter()
            sel_x = np.zeros(n, dtype=int)
            act_z = np.zeros(K, dtype=int)
            rem_b = b_limit
            rem_w = w_limit

            for i in range(n):
                # Проверка цепочки: если d < n_deps и i <= n_deps
                if 1 <= i <= n_deps and sel_x[i-1] == 0:
                    continue
                k = clusters[i]
                inc_c = c_fix[i] + (F_cl[k] if act_z[k] == 0 else 0.0)
                inc_w = w_fix[i] + (S_cl[k] if act_z[k] == 0 else 0.0)
                if inc_c <= rem_b and inc_w <= rem_w:
                    sel_x[i] = 1
                    act_z[k] = 1
                    rem_b -= inc_c
                    rem_w -= inc_w

            # Жадные y
            y_gr = np.zeros(n)
            act_idx = [i for i in range(n) if sel_x[i] == 1]
            act_idx.sort(key=lambda idx: p_var[idx] / c_var[idx] if c_var[idx] > 0 else 0.0, reverse=True)
            for i in act_idx:
                take_y = min(u_max[i], rem_b / c_var[i] if c_var[i] > 0 else u_max[i], rem_w / w_var[i] if w_var[i] > 0 else u_max[i])
                take_y = max(0.0, float(take_y))
                y_gr[i] = take_y
                rem_b -= take_y * c_var[i]
                rem_w -= take_y * w_var[i]

            t_greedy = (time.perf_counter() - t0) * 1000.0

            npv_milp = -float(res_m.fun) if res_m.success else np.nan
            npv_lp = -float(res_lp.fun) if res_lp.success else np.nan
            npv_greedy = float(np.dot(p_fix, sel_x) + np.dot(p_var, y_gr) - np.dot(F_cl, act_z))

            int_gap = ((npv_lp - npv_milp) / npv_milp * 100.0) if npv_milp > 0 else 0.0
            greedy_gap = ((npv_milp - npv_greedy) / npv_milp * 100.0) if npv_milp > 0 else 0.0

            results.append({
                "N_Projects": n,
                "N_Clusters": K,
                "Total_Variables": n_vars,
                "Binary_Variables": n + K,
                "Continuous_Variables": n,
                "Num_Constraints": len(b_rows),
                "MILP_Time_ms": round(t_milp, 2),
                "LP_Time_ms": round(t_lp, 2),
                "Greedy_Time_ms": round(t_greedy, 3),
                "MIP_Nodes": getattr(res_m, "mip_node_count", 0),
                "Simplex_Iters": getattr(res_lp, "nit", 0),
                "MILP_NPV_MRub": round(npv_milp, 2),
                "LP_NPV_MRub": round(npv_lp, 2),
                "Greedy_NPV_MRub": round(npv_greedy, 2),
                "Integrality_Gap_Pct": round(int_gap, 2),
                "Greedy_Gap_Pct": round(greedy_gap, 2)
            })

        df = pd.DataFrame(results)
        csv_path = os.path.join(output_dir, "benchmark_results.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8")
        return df
