import numpy as np
import pandas as pd
from scipy import stats

import sims


def test_wald_fails_for_rare_events_and_wilson_does_not():
    c = sims.coverage_proportion(0.02, 50, reps=20000)
    assert c["Вальд"] < 0.75 and c["Уилсон"] > 0.9


def test_all_proportion_intervals_agree_for_large_balanced_sample():
    c = sims.coverage_proportion(0.5, 500, reps=20000)
    assert all(abs(v - 0.95) < 0.02 for v in c.values())


def test_wilson_matches_scipy_reference():
    ref = stats.binomtest(37, 120).proportion_ci(confidence_level=0.95, method="wilson")
    lo, hi = sims.prop_intervals(np.array([37.0]), 120)["Уилсон"]
    assert abs(lo[0] - ref.low) < 1e-9 and abs(hi[0] - ref.high) < 1e-9


def test_t_interval_is_fine_for_normal_like_and_bad_for_heavy_tail_small_n():
    small = sims.coverage_mean_lognormal(20, sigma=1.5, reps=8000)["t интервал"]
    big = sims.coverage_mean_lognormal(1000, sigma=0.3, reps=4000)["t интервал"]
    assert small < 0.88 and abs(big - 0.95) < 0.015


def test_naive_peeking_inflates_false_positives_and_one_look_does_not():
    cum = sims.peeking_paths(1000, 8000, 0.0, seed=1)
    one, _ = sims.evaluate_rule(cum, [1000], np.array([1.96]))
    looks = list(range(100, 1001, 100))
    ten, _ = sims.evaluate_rule(cum, looks, np.full(10, 1.96))
    assert abs(one - 0.05) < 0.012 and ten > 0.14


def test_calibrated_boundaries_hold_alpha_on_fresh_data():
    looks = list(range(200, 2001, 200))
    fresh = sims.peeking_paths(2000, 20000, 0.0, seed=99)
    for kind in ["pocock", "obf"]:
        b = sims.calibrate_boundary(looks, kind, reps=20000, seed=5)
        fpr, _ = sims.evaluate_rule(fresh, looks, b)
        assert abs(fpr - 0.05) < 0.01, kind
    pocock = sims.calibrate_boundary(looks, "pocock", reps=20000, seed=5)
    obf = sims.calibrate_boundary(looks, "obf", reps=20000, seed=5)
    assert np.allclose(pocock, pocock[0]) and (np.diff(obf) < 0).all() and obf[0] > pocock[0] > obf[-1]


def test_z_statistic_equals_two_sample_formula():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=500), rng.normal(size=500)
    cum = np.cumsum(b - a)[None, :]
    z = sims.z_at_looks(cum, [500])[0, 0]
    # при известной дисперсии 1 это обычный z критерий для двух выборок
    assert abs(z - (b.mean() - a.mean()) / np.sqrt(2 / 500)) < 1e-9


def test_bonferroni_and_holm_control_fwer_under_global_null():
    r = sims.global_null_fwer(m=20, reps=20000)
    assert r["без поправки"] > 0.55
    for k in ["Бонферрони", "Холм", "Бенджамини Хохберг"]:
        assert r[k] < 0.065, k


def test_bh_controls_fdr_and_beats_bonferroni_on_power():
    t = sims.multiple_testing(reps=6000).set_index("method")
    assert t.loc["Бенджамини Хохберг", "fdr"] < 0.06
    assert t.loc["Бенджамини Хохберг", "power"] > t.loc["Бонферрони", "power"] + 0.1
    assert t.loc["Холм", "power"] >= t.loc["Бонферрони", "power"] - 0.01


def test_adjust_known_small_example():
    p = np.array([[0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216]])
    assert sims.adjust(p, "Бенджамини Хохберг").sum() == 2
    assert sims.adjust(p, "Бонферрони").sum() == 1
    assert sims.adjust(p, "Холм").sum() == 1


def test_simpson_reverses_sign_in_observational_but_not_randomized():
    s = sims.simpson(n=200000)
    obs, rnd = s["наблюдательные данные"], s["случайное распределение"]
    assert obs["naive"] < -0.05 and obs["stratified"] > 0.01
    assert abs(rnd["naive"] - 0.02) < 0.01 and abs(rnd["stratified"] - 0.02) < 0.01


def test_cuped_variance_matches_one_minus_rho_squared_and_is_unbiased():
    for rho in [0.0, 0.5, 0.9]:
        r = sims.cuped(rho, n=300, reps=3000)
        assert abs(r["var_ratio"] - (1 - rho ** 2)) < 0.06, rho
        assert abs(r["bias"]) < 0.01
    # эффект сохраняется после поправки
    assert abs(sims.cuped(0.7, effect=0.2, reps=2000)["bias"]) < 0.01


def test_clustered_data_inflate_session_level_test_but_not_user_level():
    r = sims.clustered_fpr(0.3, reps=4000)
    assert r["naive_fpr"] > 0.2 and abs(r["naive_fpr"] - r["theory_naive"]) < 0.04
    assert abs(r["user_level_fpr"] - 0.05) < 0.012
    z = sims.clustered_fpr(0.0, reps=4000)
    assert abs(z["naive_fpr"] - 0.05) < 0.012


def test_holm_stops_at_first_failure():
    # первый порог 0,0125 не пройден, поэтому следующие, даже подходящие под свои пороги, не отвергаются
    p = np.array([[0.02, 0.021, 0.022, 0.0245]])
    assert sims.adjust(p, "Холм").sum() == 0
    # Бенджамини Хохберг тут отвергает все четыре: последний p меньше 0,05
    assert sims.adjust(p, "Бенджамини Хохберг").sum() == 4
