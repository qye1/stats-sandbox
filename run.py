# запускает все симуляции, сохраняет таблицы и графики в results

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sims

ROOT = Path(__file__).parent
RES = ROOT / "results"
BLUE, ORANGE, GREY, GREEN, RED = "#2b6cb0", "#dd6b20", "#a0aec0", "#2f855a", "#c53030"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "figure.dpi": 110, "savefig.bbox": "tight"})
pct = lambda v, p: f"{v:.0%}"


def exp_coverage():
    rows = []
    for p in [0.02, 0.1, 0.5]:
        for n in [10, 20, 50, 100, 200, 500]:
            for meth, c in sims.coverage_proportion(p, n).items():
                rows.append({"p": p, "n": n, "method": meth, "coverage": c})
    prop = pd.DataFrame(rows)
    prop.to_csv(RES / "coverage_proportion.csv", index=False, encoding="utf-8-sig")
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    colors = {"Вальд": RED, "Уилсон": BLUE, "Агрести Каул": GREEN}
    for ax, p in zip(axs, [0.02, 0.1, 0.5]):
        for meth, col in colors.items():
            d = prop[(prop["p"] == p) & (prop["method"] == meth)]
            ax.plot(d["n"], d["coverage"], marker="o", ms=4, color=col, label=meth)
        ax.axhline(0.95, color=GREY, ls="--", lw=1)
        ax.set_xscale("log")
        ax.set(title=f"доля p = {p:.0%}", xlabel="размер выборки n")
        ax.yaxis.set_major_formatter(pct)
    axs[0].set_ylabel("реальное покрытие 95% интервала")
    axs[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.savefig(RES / "01_coverage_proportion.png")

    ln = []
    for n in [10, 30, 100, 300, 1000]:
        r = sims.coverage_mean_lognormal(n, boot=n <= 300, reps=10000 if n > 300 else 6000)
        for meth, c in r.items():
            ln.append({"n": n, "method": meth, "coverage": c})
    ln = pd.DataFrame(ln)
    ln.to_csv(RES / "coverage_lognormal.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for meth, col in [("t интервал", BLUE), ("бутстреп", ORANGE)]:
        d = ln[ln["method"] == meth]
        ax.plot(d["n"], d["coverage"], marker="o", color=col, label=meth)
    ax.axhline(0.95, color=GREY, ls="--", lw=1)
    ax.set_xscale("log")
    ax.yaxis.set_major_formatter(pct)
    ax.set(xlabel="размер выборки n", ylabel="реальное покрытие", title="Среднее чека с тяжелым хвостом: интервал недобирает")
    ax.legend(frameon=False)
    fig.savefig(RES / "02_coverage_skewed.png")
    return prop, ln


def exp_peeking():
    n_total = 2000
    cum0 = sims.peeking_paths(n_total, 20000, 0.0, seed=1)
    cum1 = sims.peeking_paths(n_total, 20000, 0.1, seed=11)
    rows = []
    for k in [1, 2, 3, 5, 10, 20, 50]:
        looks = [n_total * i // k for i in range(1, k + 1)]
        fpr, _ = sims.evaluate_rule(cum0, looks, np.full(k, 1.96))
        rows.append({"looks": k, "naive_fpr": fpr})
    fpr_tbl = pd.DataFrame(rows)
    looks = [200 * i for i in range(1, 11)]
    k = len(looks)
    rules = {"один взгляд в конце": (looks[-1:], np.array([1.96])),
             "подглядывание без поправки": (looks, np.full(k, 1.96)),
             "граница Покока": (looks, sims.calibrate_boundary(looks, "pocock")),
             "граница О'Брайена Флеминга": (looks, sims.calibrate_boundary(looks, "obf"))}
    out = []
    for name, (lk, b) in rules.items():
        f, n_null = sims.evaluate_rule(cum0, lk, b)
        pw, n_alt = sims.evaluate_rule(cum1, lk, b)
        out.append({"rule": name, "false_positive_rate": f, "power": pw, "mean_n_if_effect": n_alt, "mean_n_if_no_effect": n_null})
    tbl = pd.DataFrame(out)
    tbl.to_csv(RES / "peeking_rules.csv", index=False, encoding="utf-8-sig")
    fpr_tbl.to_csv(RES / "peeking_fpr_by_looks.csv", index=False)

    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    axs[0].plot(fpr_tbl["looks"], fpr_tbl["naive_fpr"], marker="o", color=RED)
    axs[0].axhline(0.05, color=GREY, ls="--", lw=1)
    axs[0].set_xscale("log")
    axs[0].set_xticks([1, 2, 5, 10, 20, 50], ["1", "2", "5", "10", "20", "50"])
    axs[0].yaxis.set_major_formatter(pct)
    axs[0].set(xlabel="сколько раз заглянули в результат", ylabel="ложные срабатывания", title="Чем чаще смотришь,\nтем чаще ошибаешься")
    names = ["один\nвзгляд", "без\nпоправки", "Покок", "О'Брайен\nФлеминг"]
    cols = [GREY, RED, BLUE, BLUE]
    axs[1].bar(names, tbl["false_positive_rate"], color=cols)
    axs[1].axhline(0.05, color="black", ls="--", lw=1)
    axs[1].yaxis.set_major_formatter(pct)
    axs[1].set_title("Эффекта нет:\nложные срабатывания")
    axs[2].bar(names, tbl["power"], color=cols)
    for i, v in enumerate(tbl["mean_n_if_effect"]):
        axs[2].text(i, tbl["power"].iloc[i] + 0.015, f"n≈{v:.0f}", ha="center", fontsize=8)
    axs[2].set_ylim(0, 1.05)
    axs[2].yaxis.set_major_formatter(pct)
    axs[2].set_title("Эффект есть: мощность\nи средний размер выборки")
    fig.savefig(RES / "03_peeking.png")
    return fpr_tbl, tbl


def exp_multiple():
    mt = sims.multiple_testing()
    null = sims.global_null_fwer()
    mt.round(4).to_csv(RES / "multiple_testing.csv", index=False, encoding="utf-8-sig")
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    x = np.arange(len(mt))
    axs[0].bar(x - 0.2, mt["fwer"], 0.4, color=GREY, label="хотя бы одна ложная находка")
    axs[0].bar(x + 0.2, mt["fdr"], 0.4, color=BLUE, label="доля ложных среди находок")
    axs[0].set_xticks(x, [m.replace(" ", "\n", 1) for m in mt["method"]], fontsize=8)
    axs[0].axhline(0.05, color="black", ls="--", lw=1)
    axs[0].yaxis.set_major_formatter(pct)
    axs[0].legend(frameon=False, fontsize=8)
    axs[0].set_title("Ошибки: 100 проверок, 10 настоящих эффектов")
    axs[1].bar(x, mt["power"], color=ORANGE)
    axs[1].set_xticks(x, [m.replace(" ", "\n", 1) for m in mt["method"]], fontsize=8)
    axs[1].yaxis.set_major_formatter(pct)
    axs[1].set_title("Сколько настоящих эффектов нашли")
    fig.savefig(RES / "04_multiple_testing.png")
    return mt, null


def exp_simpson():
    s = sims.simpson()
    pd.DataFrame(s).T.round(4).to_csv(RES / "simpson.csv", encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(s)
    x = np.arange(2)
    ax.bar(x - 0.2, [s[n]["naive"] for n in names], 0.4, color=RED, label="простое сравнение B и A")
    ax.bar(x + 0.2, [s[n]["stratified"] for n in names], 0.4, color=BLUE, label="внутри сегментов, потом среднее")
    ax.axhline(0.02, color="black", ls="--", lw=1)
    ax.text(1.45, 0.022, "настоящий эффект +2 п.п.", ha="right", fontsize=8)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x, names)
    ax.yaxis.set_major_formatter(lambda v, p: f"{v * 100:.0f} п.п.")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_ylim(top=0.035)
    ax.set_title("Парадокс Симпсона, если вариант B показывали неслучайно")
    fig.savefig(RES / "05_simpson.png")
    return s


def exp_cuped():
    rows = [dict(rho=r, **sims.cuped(r)) for r in [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]]
    c = pd.DataFrame(rows)
    c.round(4).to_csv(RES / "cuped.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.2, 4))
    rr = np.linspace(0, 0.95, 50)
    ax.plot(rr, 1 - rr ** 2, color=GREY, label="теория: 1 минус корреляция в квадрате")
    ax.scatter(c["rho"], c["var_ratio"], color=BLUE, zorder=3, label="симуляция")
    ax.set(xlabel="корреляция метрики с ее значением до эксперимента", ylabel="дисперсия после CUPED к исходной",
           title="CUPED: чем сильнее связь с прошлым, тем меньше шум")
    ax.legend(frameon=False)
    fig.savefig(RES / "06_cuped.png")
    return c


def exp_cluster():
    rows = []
    for icc in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]:
        rows.append(dict(icc=icc, **sims.clustered_fpr(icc)))
    c = pd.DataFrame(rows)
    c.round(4).to_csv(RES / "clustered_fpr.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.2, 4))
    ax.plot(c["icc"], c["theory_naive"], color=GREY, label="теория по эффекту дизайна")
    ax.plot(c["icc"], c["naive_fpr"], marker="o", color=RED, label="тест по сессиям")
    ax.plot(c["icc"], c["user_level_fpr"], marker="o", color=BLUE, label="тест по людям")
    ax.axhline(0.05, color="black", ls="--", lw=1)
    ax.yaxis.set_major_formatter(pct)
    ax.set(xlabel="внутриклассовая корреляция (насколько сессии одного человека похожи)", ylabel="ложные срабатывания",
           title="Сессии одного человека не независимы")
    ax.legend(frameon=False)
    fig.savefig(RES / "07_clustered.png")
    return c


def main():
    RES.mkdir(exist_ok=True)
    prop, ln = exp_coverage()
    print("покрытие доли p=2%, n=50:", prop[(prop.p == 0.02) & (prop.n == 50)].set_index("method")["coverage"].round(3).to_dict())
    print("покрытие среднего логнормали n=30:", ln[ln.n == 30].set_index("method")["coverage"].round(3).to_dict())
    fpr, rules = exp_peeking()
    print(fpr.round(3).to_string(index=False))
    print(rules.round(3).to_string(index=False))
    mt, null = exp_multiple()
    print(mt.round(3).to_string(index=False))
    print("20 проверок без эффекта, хотя бы одна находка:", {k: round(v, 3) for k, v in null.items()})
    s = exp_simpson()
    print({k: {a: round(b, 4) for a, b in v.items()} for k, v in s.items()})
    print(exp_cuped().round(3).to_string(index=False))
    print(exp_cluster().round(3).to_string(index=False))
    plt.close("all")
    print("готово, файлы в", RES)


if __name__ == "__main__":
    main()
