#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль визуализации результатов оптимизации и вычислительных экспериментов.
Генерирует публикационные графики высокого разрешения (300 DPI):
  1. project_selection.png      - Структура инвестиций и вклад в NPV выбранных проектов
  2. dependency_dag.png         - Граф технологических зависимостей, кластеров и взаимоисключений
  3. scenarios_comparison.png   - Сравнительный анализ сценариев (Base, Deficit, Stress)
  4. lp_vs_milp_gap.png         - Сравнение интегрального зазора и силы формулировок
  5. big_m_impact.png           - Влияние константы Big-M на LP-релаксацию и зазор
  6. benchmark_scaling.png      - Масштабируемость времени решения и узлов ветвления (N=10..160)
  7. stochastic_analysis.png    - Двухэтапная стохастическая модель: HN, WS, EEV, EVPI и VSS
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, Any


class PortfolioVisualizer:
    """Визуализатор результатов лабораторной работы № 3."""

    def __init__(self, raw_data: Dict[str, Any], plots_dir: str = "output/plots"):
        self.raw_data = raw_data
        self.plots_dir = plots_dir
        os.makedirs(self.plots_dir, exist_ok=True)

        plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
        plt.rcParams["axes.unicode_minus"] = False

    def plot_project_selection(self, milp_res: Dict[str, Any], filename: str = "project_selection.png"):
        """Визуализация выбранных проектов, структуры затрат и NPV."""
        projects = self.raw_data["projects"]
        n = len(projects)
        x_sol = milp_res["x"][0:n]
        y_sol = milp_res["x"][n:2*n]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        proj_labels = [f"P{i+1}" for i in range(n)]
        colors = ["#2ecc71" if x_sol[i] > 0.5 else "#e74c3c" for i in range(n)]

        # 1. Затраты (Fixed + Variable)
        c_fix = [p["c_fix"] * x_sol[i] for i, p in enumerate(projects)]
        c_var = [p["c_var"] * y_sol[i] for i, p in enumerate(projects)]

        ax1.bar(proj_labels, c_fix, label="Фиксированные затраты (c_fix)", color="#3498db", alpha=0.85)
        ax1.bar(proj_labels, c_var, bottom=c_fix, label="Переменные затраты (c_var * y)", color="#9b59b6", alpha=0.85)
        ax1.set_title("Капитальные затраты по проектам (млн руб.)", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Проект", fontsize=11)
        ax1.set_ylabel("Инвестиции (млн руб.)", fontsize=11)
        ax1.legend(loc="upper right")
        ax1.grid(True, linestyle="--", alpha=0.5)

        # Подсвечиваем статус выбора
        for i, tick in enumerate(ax1.xaxis.get_major_ticks()):
            tick.label1.set_color("green" if x_sol[i] > 0.5 else "red")
            tick.label1.set_fontweight("bold")

        # 2. Доход (NPV)
        npv_fix = [p["p_fix"] * x_sol[i] for i, p in enumerate(projects)]
        npv_var = [p["p_var"] * y_sol[i] for i, p in enumerate(projects)]

        ax2.bar(proj_labels, npv_fix, label="Фиксированный доход (p_fix)", color="#2ecc71", alpha=0.85)
        ax2.bar(proj_labels, npv_var, bottom=npv_fix, label="Доход масштабирования (p_var * y)", color="#f1c40f", alpha=0.85)
        ax2.set_title("Чистый дисконтированный доход (NPV, млн руб.)", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Проект", fontsize=11)
        ax2.set_ylabel("NPV (млн руб.)", fontsize=11)
        ax2.legend(loc="upper right")
        ax2.grid(True, linestyle="--", alpha=0.5)

        for i, tick in enumerate(ax2.xaxis.get_major_ticks()):
            tick.label1.set_color("green" if x_sol[i] > 0.5 else "red")
            tick.label1.set_fontweight("bold")

        fig.suptitle("Базовый план портфеля: Зелёный - выбран в оптимуме, Красный - отклонен солвером", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, filename)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plot_dependency_dag(self, milp_res: Dict[str, Any], filename: str = "dependency_dag.png"):
        """Граф технологических зависимостей, кластеров и взаимоисключений."""
        projects = self.raw_data["projects"]
        clusters = self.raw_data["clusters"]
        n = len(projects)
        x_sol = milp_res["x"][0:n]
        z_sol = milp_res["x"][2*n:]

        G = nx.DiGraph()

        # Добавляем узлы
        for p in projects:
            p_id = p["project_id"]
            idx = p["index"]
            is_sel = x_sol[idx] > 0.5
            G.add_node(p_id, cluster=p["cluster_id"], selected=is_sel, name=p["name"])

        # Добавляем ориентированные ребра предшествования
        for d in self.raw_data["dependencies"]:
            if d["type"] in ["Precedence", "Co-requisite"]:
                G.add_edge(d["from_project"], d["to_project"], dep_type=d["type"])

        # Фиксированные координаты для наглядного отображения по кластерам
        pos = {
            "P1": (1.0, 3.5),
            "P2": (2.2, 3.8),
            "P3": (2.2, 3.0),
            "P4": (3.4, 3.8),
            "P10": (2.2, 2.2),
            "P13": (1.2, 1.3),
            "P14": (2.8, 1.3),

            "P5": (1.0, -1.0),
            "P12": (2.3, -1.0),
            "P8": (1.0, -2.0),
            "P9": (2.3, -2.0),
            "P11": (1.0, -3.0),
            "P15": (2.3, -3.0),

            "P6": (4.5, 0.5),
            "P7": (4.5, -1.5)
        }

        fig, ax = plt.subplots(figsize=(14, 10))

        # Отрисовка подложек кластеров
        # K1
        rect_k1 = plt.Rectangle((0.5, 0.6), 3.4, 3.6, color="#e8f4f8", alpha=0.6, zorder=0)
        ax.add_patch(rect_k1)
        ax.text(0.7, 4.0, f"Кластер K1: Вычислительная инфраструктура и ИИ\n(z_1 = {int(round(z_sol[0]))}, F_1 = 35 млн руб.)",
                fontsize=11, fontweight="bold", color="#1b4f72")

        # K2
        rect_k2 = plt.Rectangle((0.5, -3.6), 2.5, 3.2, color="#eafaf1", alpha=0.6, zorder=0)
        ax.add_patch(rect_k2)
        ax.text(0.7, -0.6, f"Кластер K2: Автоматика и робототехника\n(z_2 = {int(round(z_sol[1]))}, F_2 = 40 млн руб.)",
                fontsize=11, fontweight="bold", color="#196f3d")

        # K3
        rect_k3 = plt.Rectangle((3.9, -2.2), 1.6, 3.4, color="#fef9e7", alpha=0.6, zorder=0)
        ax.add_patch(rect_k3)
        ax.text(4.0, 1.0, f"Кластер K3: Кибербезопасность\n(z_3 = {int(round(z_sol[2]))}, F_3 = 30 млн руб.)",
                fontsize=11, fontweight="bold", color="#7d6608")

        # Цвета узлов
        node_colors = ["#2ecc71" if G.nodes[n]["selected"] else "#e74c3c" for n in G.nodes()]

        # Отрисовка узлов
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1600, edgecolors="#2c3e50", linewidths=2.0, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=11, font_weight="bold", font_color="white", ax=ax)

        # Отрисовка ориентированных дуг предшествования
        nx.draw_networkx_edges(G, pos, edge_color="#2980b9", width=2.5, arrowsize=20, arrowstyle="-|>",
                               connectionstyle="arc3,rad=0.05", ax=ax)

        # Отрисовка взаимоисключения P13 <-> P14 пунктирной красной линией
        p13_pos = pos["P13"]
        p14_pos = pos["P14"]
        ax.annotate("", xy=p14_pos, xytext=p13_pos,
                    arrowprops=dict(arrowstyle="<->", color="#c0392b", lw=2.5, linestyle="--"))
        ax.text(2.0, 1.45, "Взаимоисключение: x_13 + x_14 <= 1", fontsize=9, fontweight="bold", color="#c0392b", ha="center")

        ax.set_title("Граф технологических зависимостей проектов (DAG) и статус оптимального выбора\n"
                     "Зеленый - включен в портфель, Красный - отклонен солвером HiGHS",
                     fontsize=14, fontweight="bold", pad=20)
        ax.set_xlim(0.0, 6.0)
        ax.set_ylim(-4.0, 4.5)
        ax.axis("off")

        save_path = os.path.join(self.plots_dir, filename)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plot_scenarios_comparison(self, df_scenarios: pd.DataFrame, filename: str = "scenarios_comparison.png"):
        """Сравнение базового, дефицитного и стрессового сценариев."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

        sc_names = df_scenarios["Scenario"]

        # 1. Сравнение NPV (MILP vs LP)
        width = 0.35
        x = np.arange(len(sc_names))
        ax1.bar(x - width/2, df_scenarios["MILP_NPV_MRub"], width, label="MILP Оптимум", color="#2ecc71")
        ax1.bar(x + width/2, df_scenarios["LP_NPV_MRub"], width, label="LP Релаксация", color="#3498db")
        ax1.set_xticks(x)
        ax1.set_xticklabels(sc_names, fontweight="bold")
        ax1.set_title("Чистый дисконтированный доход (NPV, млн руб.)", fontweight="bold")
        ax1.set_ylabel("млн руб.")
        ax1.legend()
        ax1.grid(True, linestyle="--", alpha=0.5)

        # 2. Интегральный зазор (%)
        bars = ax2.bar(sc_names, df_scenarios["Integrality_Gap_Pct"], color="#e67e22", width=0.5)
        ax2.set_title("Интегральный зазор ветвления (Integrality Gap, %)", fontweight="bold")
        ax2.set_ylabel("%")
        ax2.grid(True, linestyle="--", alpha=0.5)
        for bar in bars:
            yval = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f"{yval:.2f}%", ha="center", va="bottom", fontweight="bold")

        # 3. Число выбранных проектов и активных кластеров
        ax3.bar(x - width/2, df_scenarios["Selected_Count"], width, label="Выбрано проектов (из 15)", color="#1abc9c")
        ax3.bar(x + width/2, df_scenarios["Active_Clusters_Count"], width, label="Активных кластеров (из 3)", color="#9b59b6")
        ax3.set_xticks(x)
        ax3.set_xticklabels(sc_names, fontweight="bold")
        ax3.set_title("Физический состав инвестиционного портфеля", fontweight="bold")
        ax3.set_ylabel("Количество единиц")
        ax3.legend()
        ax3.grid(True, linestyle="--", alpha=0.5)

        # 4. Утилизация ресурсов (%)
        ax4.plot(sc_names, df_scenarios["Capital_Utilization_Pct"], marker="o", lw=2.5, label="Капитал (%)", color="#e74c3c")
        ax4.plot(sc_names, df_scenarios["Labor_Utilization_Pct"], marker="s", lw=2.5, label="Трудовые ресурсы (%)", color="#2980b9")
        ax4.set_title("Утилизация критических ресурсов (%)", fontweight="bold")
        ax4.set_ylabel("% освоения лимита")
        ax4.set_ylim(60, 105)
        ax4.legend()
        ax4.grid(True, linestyle="--", alpha=0.5)

        fig.suptitle("Комплексный сценарный анализ инвестиционной программы (Base, Deficit, Stress)",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, filename)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plot_big_m_impact(self, df_big_m: pd.DataFrame, filename: str = "big_m_impact.png"):
        """График влияния константы Big-M на качество LP-релаксации и интегральный зазор."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        factors = df_big_m["BigM_Factor"]
        x_indices = range(len(factors))

        # 1. Значение целевой функции LP vs MILP
        ax1.plot(x_indices, df_big_m["LP_NPV_MRub"], marker="o", color="#e74c3c", lw=2.5, label="LP Релаксация (верхняя грань)")
        ax1.axhline(df_big_m["MILP_NPV_MRub"].iloc[0], color="#27ae60", linestyle="--", lw=2, label="MILP Оптимум (точное решение)")
        ax1.set_xticks(x_indices)
        ax1.set_xticklabels([f"{f:.0f}x" for f in factors], rotation=30)
        ax1.set_title("Искусственное завышение целевой функции LP-релаксации", fontweight="bold")
        ax1.set_xlabel("Множитель константы Big-M к физическому максимуму U_i")
        ax1.set_ylabel("NPV (млн руб.)")
        ax1.legend()
        ax1.grid(True, linestyle="--", alpha=0.5)

        # 2. Взрыв интегрального зазора (%)
        ax2.plot(x_indices, df_big_m["Integrality_Gap_Pct"], marker="s", color="#8e44ad", lw=2.5)
        ax2.set_xticks(x_indices)
        ax2.set_xticklabels([f"{f:.0f}x" for f in factors], rotation=30)
        ax2.set_title("Деградация качества двойственной оценки (MIP Gap, %)", fontweight="bold")
        ax2.set_xlabel("Множитель константы Big-M")
        ax2.set_ylabel("Интегральный зазор (%)")
        ax2.grid(True, linestyle="--", alpha=0.5)

        fig.suptitle("Патология слабого Big-M: размывание релаксации и численные искажения",
                     fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, filename)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plot_formulations_comparison(self, df_form: pd.DataFrame, filename: str = "formulations_comparison.png"):
        """Сопоставление сильной (Strong) и слабой (Weak) формулировок."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        forms = ["Strong Formulation\n(x_i <= z_k)", "Weak Formulation\n(sum x_i <= |C_k|*z_k)"]
        lp_vals = df_form["LP_NPV_MRub"].values
        milp_vals = df_form["MILP_NPV_MRub"].values
        gaps = df_form["Integrality_Gap_Pct"].values

        x = np.arange(len(forms))
        width = 0.35

        # 1. Значения целей
        ax1.bar(x - width/2, milp_vals, width, label="MILP Оптимум", color="#2ecc71")
        ax1.bar(x + width/2, lp_vals, width, label="LP Релаксация", color="#3498db")
        ax1.set_xticks(x)
        ax1.set_xticklabels(forms, fontweight="bold")
        ax1.set_title("Сравнение целевых функций (млн руб.)", fontweight="bold")
        ax1.set_ylabel("NPV (млн руб.)")
        ax1.legend(loc="upper left")
        ax1.grid(True, linestyle="--", alpha=0.5)

        # 2. Зазор (%)
        bars = ax2.bar(forms, gaps, color=["#27ae60", "#c0392b"], width=0.45)
        ax2.set_title("Интегральный зазор релаксации (%)", fontweight="bold")
        ax2.set_ylabel("Gap (%)")
        ax2.grid(True, linestyle="--", alpha=0.5)
        for bar in bars:
            yval = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.2, f"{yval:.2f}%", ha="center", va="bottom", fontweight="bold")

        fig.suptitle("Теорема о доминировании формулировок: Сильная формулировка сжимает многогранник релаксации",
                     fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, filename)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plot_benchmark_scaling(self, df_bench: pd.DataFrame, filename: str = "benchmark_scaling.png"):
        """Кривые масштабируемости времени решения и зазоров."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

        n_vals = df_bench["N_Projects"]

        # 1. Время решения (мс, лог-шкала)
        ax1.plot(n_vals, df_bench["MILP_Time_ms"], marker="o", color="#e74c3c", lw=2.5, label="HiGHS MILP")
        ax1.plot(n_vals, df_bench["LP_Time_ms"], marker="s", color="#3498db", lw=2, label="HiGHS LP Simplex")
        ax1.plot(n_vals, df_bench["Greedy_Time_ms"], marker="^", color="#2ecc71", lw=1.5, linestyle="--", label="Greedy Heuristic")
        ax1.set_yscale("log")
        ax1.set_title("Время решения задач различной размерности (мс, log-шкала)", fontweight="bold")
        ax1.set_xlabel("Число проектов (N)")
        ax1.set_ylabel("Время (мс)")
        ax1.legend()
        ax1.grid(True, linestyle="--", alpha=0.5)

        # 2. Узлы Branch-and-Bound и симплекс-итерации
        ax2.plot(n_vals, df_bench["MIP_Nodes"], marker="d", color="#9b59b6", lw=2.5, label="Узлы дерева ветвления (MIP Nodes)")
        ax2.plot(n_vals, df_bench["Simplex_Iters"], marker="x", color="#f39c12", lw=2, label="Симплекс-итерации LP")
        ax2.set_title("Вычислительная трудоемкость (узлы и итерации)", fontweight="bold")
        ax2.set_xlabel("Число проектов (N)")
        ax2.set_ylabel("Количество")
        ax2.legend()
        ax2.grid(True, linestyle="--", alpha=0.5)

        # 3. Интегральный зазор (%)
        ax3.plot(n_vals, df_bench["Integrality_Gap_Pct"], marker="o", color="#e67e22", lw=2.5)
        ax3.set_title("Динамика интегрального зазора релаксации (MIP Gap, %)", fontweight="bold")
        ax3.set_xlabel("Число проектов (N)")
        ax3.set_ylabel("Зазор (%)")
        ax3.grid(True, linestyle="--", alpha=0.5)

        # 4. Зазор жадной эвристики к оптимуму MILP (%)
        ax4.plot(n_vals, df_bench["Greedy_Gap_Pct"], marker="v", color="#c0392b", lw=2.5)
        ax4.set_title("Потери эффективности жадной эвристики (Greedy Gap, %)", fontweight="bold")
        ax4.set_xlabel("Число проектов (N)")
        ax4.set_ylabel("Потери качества (%)")
        ax4.grid(True, linestyle="--", alpha=0.5)

        fig.suptitle("Нагрузочный бенчмарк алгоритмов оптимизации (N = 10..160 проектов, до 350 переменных)",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, filename)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plot_stochastic_analysis(self, df_stoch: pd.DataFrame, filename: str = "stochastic_analysis.png"):
        """Визуализация двухэтапного стохастического программирования."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        metrics = ["HN (Here-and-Now)", "WS (Wait-and-See)", "EEV (Expected Value)"]
        hn_val = df_stoch.loc[df_stoch["Metric"].str.contains("Here-and-Now"), "Value_MRub"].values[0]
        ws_val = df_stoch.loc[df_stoch["Metric"].str.contains("Wait-and-See"), "Value_MRub"].values[0]
        eev_val = df_stoch.loc[df_stoch["Metric"].str.contains("EEV"), "Value_MRub"].values[0]
        evpi_val = df_stoch.loc[df_stoch["Metric"].str.contains("EVPI"), "Value_MRub"].values[0]
        vss_val = df_stoch.loc[df_stoch["Metric"].str.contains("VSS"), "Value_MRub"].values[0]

        vals = [hn_val, ws_val, eev_val]
        bars1 = ax1.bar(metrics, vals, color=["#2980b9", "#27ae60", "#f39c12"], width=0.45)
        ax1.set_title("Ожидаемый NPV в стохастических постановках (млн руб.)", fontweight="bold")
        ax1.set_ylabel("NPV (млн руб.)")
        ax1.grid(True, linestyle="--", alpha=0.5)
        for bar in bars1:
            yval = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, yval + 5, f"{yval:.1f}", ha="center", va="bottom", fontweight="bold")

        # Показатели ценности
        v_metrics = ["EVPI\n(Ценность идеальной инфо)", "VSS\n(Ценность стохаст. модели)"]
        v_vals = [evpi_val, vss_val]
        bars2 = ax2.bar(v_metrics, v_vals, color=["#8e44ad", "#d35400"], width=0.45)
        ax2.set_title("Экономические эффекты стохастического подхода", fontweight="bold")
        ax2.set_ylabel("млн руб.")
        ax2.grid(True, linestyle="--", alpha=0.5)
        for bar in bars2:
            yval = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{yval:.2f} млн", ha="center", va="bottom", fontweight="bold")

        fig.suptitle("Двухэтапное стохастическое программирование в условиях макроэкономической неопределенности",
                     fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, filename)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
