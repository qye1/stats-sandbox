# симуляции Монте Карло по статистике, каждая функция отвечает на один вопрос: насколько можно верить методу

import numpy as np
import pandas as pd
from scipy import stats

Z95 = stats.norm.ppf(0.975)


# ---------- 1. покрытие доверительных интервалов ----------

def prop_intervals(x, n, z=Z95):
    ph = x / n
    wald_h = z * np.sqrt(ph * (1 - ph) / n)
    wald = (ph - wald_h, ph + wald_h)
    c = (ph + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
    h = z * np.sqrt(ph * (1 - ph) / n + z ** 2 / (4 * n ** 2)) / (1 + z ** 2 / n)
    wilson = (c - h, c + h)
    nt = n + z ** 2
    pt = (x + z ** 2 / 2) / nt
    ht = z * np.sqrt(pt * (1 - pt) / nt)
    ac = (pt - ht, pt + ht)
    return {"Вальд": wald, "Уилсон": wilson, "Агрести Каул": ac}


def coverage_proportion(p, n, reps=40000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.binomial(n, p, size=reps)
    out = {}
    for name, (lo, hi) in prop_intervals(x, n).items():
        out[name] = float(((lo <= p) & (p <= hi)).mean())
    return out


def coverage_mean_lognormal(n, sigma=1.5, reps=10000, boot=False, n_boot=400, seed=0):
    # настоящее среднее логнормального распределения известно: exp(sigma^2 / 2)
    rng = np.random.default_rng(seed)
    truth = np.exp(sigma ** 2 / 2)
    x = rng.lognormal(0, sigma, size=(reps, n))
    m = x.mean(axis=1)
    se = x.std(axis=1, ddof=1) / np.sqrt(n)
    t = stats.t.ppf(0.975, n - 1)
    res = {"t интервал": float(((m - t * se <= truth) & (truth <= m + t * se)).mean())}
    if boot:
        r = min(reps, 1500)
        cov = 0
        for i in range(r):
            bm = x[i][rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
            lo, hi = np.quantile(bm, [0.025, 0.975])
            cov += lo <= truth <= hi
        res["бутстреп"] = cov / r
    return res


# ---------- 2. подглядывание в A/B тест ----------

def peeking_paths(n_total=2000, reps=20000, effect=0.0, seed=1):
    # разность двух групп с единичной дисперсией на человека: накопленная сумма по парам даст z на любом шаге
    rng = np.random.default_rng(seed)
    d = rng.normal(effect, np.sqrt(2.0), size=(reps, n_total))
    return np.cumsum(d, axis=1)


def z_at_looks(cum, looks):
    looks = np.asarray(looks)
    return cum[:, looks - 1] / np.sqrt(2.0 * looks)


def calibrate_boundary(looks, kind="pocock", alpha=0.05, reps=40000, n_total=2000, seed=2):
    # границу подбираем так, чтобы ложных срабатываний за все заглядывания было ровно alpha
    cum = peeking_paths(n_total, reps, 0.0, seed)
    z = np.abs(z_at_looks(cum, looks))
    k = len(looks)
    shape = np.ones(k) if kind == "pocock" else np.sqrt(k / (np.arange(k) + 1))
    c = float(np.quantile((z / shape).max(axis=1), 1 - alpha))
    return c * shape


def evaluate_rule(cum, looks, bounds):
    z = np.abs(z_at_looks(cum, looks))
    hit = z > bounds
    stopped = hit.any(axis=1)
    first = np.where(stopped, hit.argmax(axis=1), len(looks) - 1)
    n_used = np.asarray(looks)[first]
    return float(stopped.mean()), float(n_used.mean())


# ---------- 3. множественные проверки ----------

def adjust(p, method):
    # возвращает булевую матрицу отвержений при уровне 0,05, p имеет форму (повторы, m)
    m = p.shape[1]
    a = 0.05
    if method == "без поправки":
        return p <= a
    if method == "Бонферрони":
        return p <= a / m
    order = np.argsort(p, axis=1)
    ps = np.take_along_axis(p, order, axis=1)
    k = np.arange(1, m + 1)
    if method == "Холм":
        ok = ps <= a / (m - k + 1)
        # Холм останавливается на первом непройденном пороге
        passed = np.cumprod(ok, axis=1).astype(bool)
    elif method == "Бенджамини Хохберг":
        ok = ps <= a * k / m
        last = np.where(ok.any(axis=1), m - 1 - np.argmax(ok[:, ::-1], axis=1), -1)
        passed = k[None, :] - 1 <= last[:, None]
    else:
        raise ValueError(method)
    rej = np.zeros_like(passed)
    np.put_along_axis(rej, order, passed, axis=1)
    return rej


def multiple_testing(m=100, n_true=10, mu=3.5, reps=20000, seed=3):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(reps, m))
    is_true = np.zeros(m, bool)
    is_true[:n_true] = True
    z[:, is_true] += mu
    p = 2 * stats.norm.sf(np.abs(z))
    rows = []
    for meth in ["без поправки", "Бонферрони", "Холм", "Бенджамини Хохберг"]:
        rej = adjust(p, meth)
        false = rej[:, ~is_true].sum(axis=1)
        tp = rej[:, is_true].sum(axis=1)
        total = rej.sum(axis=1)
        fdp = np.where(total > 0, false / np.maximum(total, 1), 0.0)
        rows.append({"method": meth, "fwer": float((false > 0).mean()), "fdr": float(fdp.mean()),
                     "power": float(tp.mean() / n_true), "false_per_run": float(false.mean())})
    return pd.DataFrame(rows)


def global_null_fwer(m=20, reps=40000, seed=4):
    rng = np.random.default_rng(seed)
    p = 2 * stats.norm.sf(np.abs(rng.normal(size=(reps, m))))
    return {meth: float(adjust(p, meth).any(axis=1).mean()) for meth in ["без поправки", "Бонферрони", "Холм", "Бенджамини Хохберг"]}


# ---------- 4. парадокс Симпсона ----------

def simpson(n=400000, effect=0.02, seed=5):
    # два сегмента: активные покупают часто, редкие гости почти не покупают
    # вариант B в наблюдательных данных чаще показывали редким гостям
    rng = np.random.default_rng(seed)
    heavy = rng.random(n) < 0.5
    base = np.where(heavy, 0.30, 0.05)

    def build(p_b_heavy, p_b_light):
        pb = np.where(heavy, p_b_heavy, p_b_light)
        b = rng.random(n) < pb
        y = rng.random(n) < base + effect * b
        return pd.DataFrame({"heavy": heavy, "b": b, "y": y})

    out = {}
    for name, d in [("наблюдательные данные", build(0.2, 0.8)), ("случайное распределение", build(0.5, 0.5))]:
        naive = d.loc[d["b"], "y"].mean() - d.loc[~d["b"], "y"].mean()
        seg = d.groupby(["heavy", "b"])["y"].mean().unstack()
        within = seg[True] - seg[False]
        w = d["heavy"].value_counts(normalize=True).reindex(within.index)
        out[name] = {"naive": float(naive), "stratified": float((within * w).sum()),
                     "heavy": float(within.loc[True]), "light": float(within.loc[False])}
    return out


# ---------- 5. CUPED ----------

def cuped(rho, n=500, reps=4000, effect=0.0, seed=6):
    rng = np.random.default_rng(seed)
    pre = rng.normal(size=(reps, 2, n))
    noise = rng.normal(size=(reps, 2, n))
    y = rho * pre + np.sqrt(1 - rho ** 2) * noise
    y[:, 1, :] += effect
    ctrl, trt = y[:, 0, :], y[:, 1, :]
    raw = trt.mean(axis=1) - ctrl.mean(axis=1)
    x_all = pre.reshape(reps, -1)
    y_all = y.reshape(reps, -1)
    xc = x_all - x_all.mean(axis=1, keepdims=True)
    theta = (xc * (y_all - y_all.mean(axis=1, keepdims=True))).sum(axis=1) / (xc ** 2).sum(axis=1)
    adj = y - theta[:, None, None] * (pre - x_all.mean(axis=1)[:, None, None])
    cu = adj[:, 1, :].mean(axis=1) - adj[:, 0, :].mean(axis=1)
    return {"var_ratio": float(cu.var() / raw.var()), "theory": 1 - rho ** 2, "sd_raw": float(raw.std()), "sd_cuped": float(cu.std()),
            "bias": float(cu.mean() - effect)}


# ---------- 6. зависимые наблюдения ----------

def clustered_fpr(icc, users=200, sessions=8, reps=6000, seed=7):
    # у каждого человека свой уровень, поэтому его сессии не независимы
    rng = np.random.default_rng(seed)
    u = rng.normal(0, np.sqrt(icc), size=(reps, 2, users, 1))
    e = rng.normal(0, np.sqrt(1 - icc), size=(reps, 2, users, sessions))
    y = u + e
    a, b = y[:, 0].reshape(reps, -1), y[:, 1].reshape(reps, -1)
    # неверный тест: каждая сессия считается отдельным наблюдением
    se = np.sqrt(a.var(axis=1, ddof=1) / a.shape[1] + b.var(axis=1, ddof=1) / b.shape[1])
    z_bad = (b.mean(axis=1) - a.mean(axis=1)) / se
    # верный тест: сначала среднее по человеку
    ua, ub = y[:, 0].mean(axis=2), y[:, 1].mean(axis=2)
    se2 = np.sqrt(ua.var(axis=1, ddof=1) / users + ub.var(axis=1, ddof=1) / users)
    z_ok = (ub.mean(axis=1) - ua.mean(axis=1)) / se2
    crit = Z95
    de = 1 + (sessions - 1) * icc
    return {"naive_fpr": float((np.abs(z_bad) > crit).mean()), "user_level_fpr": float((np.abs(z_ok) > crit).mean()),
            "theory_naive": float(2 * stats.norm.sf(crit / np.sqrt(de))), "design_effect": de}
