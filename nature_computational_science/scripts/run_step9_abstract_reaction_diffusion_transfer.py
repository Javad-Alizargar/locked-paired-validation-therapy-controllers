from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[2]
NCS = ROOT / "nature_computational_science"
OUT_DATA = NCS / "data" / "abstract_reaction_diffusion"
OUT_FIG = NCS / "figures"
OUT_NOTES = NCS / "notes"
EPS = 1e-12

SEEDS = [20260721, 20260722]
N_DRAWS = 8
GRID_N = 25
DT = 0.04
T_END = 24.0
N_STEPS = int(T_END / DT)
SOURCE_WIDTH = 0.22

SCENARIOS = [
    {"scenario": f"delay_{delay}s_noise_{int(noise * 100)}pct", "delay_s": float(delay), "noise_fraction": float(noise)}
    for delay in [0, 1.2, 2.4, 4.8]
    for noise in [0.0, 0.10]
]


def auc_score(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype=float)
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    vals = np.concatenate([pos, neg])
    order = np.argsort(vals)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(vals) + 1)
    _, inv, counts = np.unique(vals, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        ranks = (np.bincount(inv, weights=ranks) / counts)[inv]
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def logistic_fit(x_train: np.ndarray, y_train: np.ndarray, l2: float = 0.04) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.nanmean(x_train, axis=0)
    sd = np.nanstd(x_train, axis=0) + EPS
    x = np.nan_to_num((x_train - mu) / sd)
    x = np.c_[np.ones(len(x)), x]
    y = y_train.astype(float)
    n_pos = max(y.sum(), 1.0)
    n_neg = max((1 - y).sum(), 1.0)
    weights = np.where(y > 0, len(y) / (2 * n_pos), len(y) / (2 * n_neg))

    def loss(w: np.ndarray) -> float:
        z = np.clip(x @ w, -40, 40)
        return float(np.sum(weights * (np.logaddexp(0, z) - y * z)) + l2 * np.sum(w[1:] ** 2))

    def grad(w: np.ndarray) -> np.ndarray:
        z = np.clip(x @ w, -40, 40)
        p = 1 / (1 + np.exp(-z))
        g = x.T @ (weights * (p - y))
        g[1:] += 2 * l2 * w[1:]
        return g

    result = minimize(loss, np.zeros(x.shape[1]), jac=grad, method="L-BFGS-B", options={"maxiter": 250})
    return result.x, mu, sd


def logistic_predict(x_test: np.ndarray, w: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    x = np.nan_to_num((x_test - mu) / (sd + EPS))
    x = np.c_[np.ones(len(x)), x]
    z = np.clip(x @ w, -40, 40)
    return 1 / (1 + np.exp(-z))


def latin_draws(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = []
    for i in range(N_DRAWS):
        base.append(
            {
                "draw_id": i,
                "geometry_scale": float(rng.uniform(0.75, 1.45)),
                "recovery_rate": float(10 ** rng.uniform(np.log10(0.035), np.log10(0.18))),
                "dose_capacity": float(10 ** rng.uniform(np.log10(0.10), np.log10(2.50))),
                "drive_coupling": float(rng.uniform(0.70, 1.45)),
                "resource_consumption": float(rng.uniform(0.55, 1.45)),
            }
        )
    base = pd.DataFrame(base)
    transforms = {
        "baseline": lambda d: d,
        "large_geometry": lambda d: d.assign(geometry_scale=np.clip(d["geometry_scale"] * 1.35, 0.75, 2.10)),
        "low_capacity": lambda d: d.assign(dose_capacity=np.clip(d["dose_capacity"] * 0.42, 0.035, 2.50)),
        "slow_recovery": lambda d: d.assign(recovery_rate=np.clip(d["recovery_rate"] * 0.45, 0.012, 0.18)),
    }
    rows = []
    for family, transform in transforms.items():
        d = transform(base.copy())
        d["model_family"] = family
        rows.append(d)
    out = pd.concat(rows, ignore_index=True)
    out["atlas_uncertainty_id"] = np.arange(len(out))
    return out


def laplacian_neumann(y: np.ndarray, dx: float) -> np.ndarray:
    padded = np.r_[y[1], y, y[-2]]
    return (padded[:-2] - 2 * padded[1:-1] + padded[2:]) / (dx * dx)


def choose_feedback(state: dict, max_u: float = 1.0) -> float:
    if state["max_activation"] > 1.10:
        return 0.0
    if state["mean_resource"] < 0.28 and state["product_fraction"] < 0.70:
        return 0.0
    if state["max_activation"] > 0.92:
        return 0.15 if state["product_fraction"] < 0.78 else 0.0
    if state["product_fraction"] >= 0.78:
        return 0.18 if state["warm_fraction"] < 0.52 else 0.0
    if state["max_activation"] < 0.70:
        return max_u
    if state["warm_fraction"] < 0.52:
        return 0.62 * max_u
    if state["mean_resource"] < 0.40:
        return 0.22 * max_u
    return 0.40 * max_u


def predict_state(history: list[dict], delay_steps: int, lookahead_steps: int = 6) -> dict:
    latest = history[max(0, len(history) - 1 - delay_steps)].copy()
    if len(history) < delay_steps + 4:
        return latest
    idx2 = max(0, len(history) - 1 - delay_steps)
    idx1 = max(0, idx2 - 3)
    dt = max((idx2 - idx1) * DT, DT)
    horizon = (delay_steps + lookahead_steps) * DT
    for key, lo, hi in [
        ("max_activation", -0.08, 0.10),
        ("mean_resource", -0.08, 0.08),
        ("product_fraction", 0.0, 0.08),
        ("warm_fraction", -0.08, 0.08),
    ]:
        slope = np.clip((history[idx2][key] - history[idx1][key]) / dt, lo, hi)
        latest[key] = latest[key] + slope * horizon
    latest["mean_resource"] = float(np.clip(latest["mean_resource"], 0.0, 1.2))
    latest["product_fraction"] = float(np.clip(latest["product_fraction"], 0.0, 1.0))
    latest["warm_fraction"] = float(np.clip(latest["warm_fraction"], 0.0, 1.0))
    latest["max_activation"] = float(latest["max_activation"] + 0.04)
    return latest


def simulate(draw: pd.Series, scenario: dict, controller: str, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    length = float(draw["geometry_scale"])
    x = np.linspace(0, length, GRID_N)
    dx = x[1] - x[0]
    center = 0.5 * length
    source = np.exp(-0.5 * ((x - center) / (SOURCE_WIDTH * length)) ** 2)
    source = source / source.max()
    d_a = 0.018
    d_o = 0.035
    cooling = 0.19 / (length**2)
    recovery = float(draw["recovery_rate"])
    capacity = float(draw["dose_capacity"])
    drive = float(draw["drive_coupling"])
    consumption = float(draw["resource_consumption"])
    delay_steps = int(round(float(scenario["delay_s"]) / DT))
    noise = float(scenario["noise_fraction"])

    activation = np.zeros(GRID_N)
    resource = 0.75 + 0.25 * (x / length)
    product = np.zeros(GRID_N)
    history = []
    light_on = 0.0
    energy = 0.0
    max_command = 0.0

    for step in range(N_STEPS):
        state = {
            "max_activation": float(activation.max()),
            "warm_fraction": float(np.mean((activation >= 0.58) & (activation <= 1.08))),
            "mean_resource": float(resource.mean()),
            "product_fraction": float(np.mean(product >= 0.62)),
        }
        history.append(state)
        if controller == "fixed_reference":
            u = 0.82 if (step % 12) < 3 else 0.0
        else:
            if controller == "predictive_feedback":
                measured = predict_state(history, delay_steps)
            else:
                measured = history[max(0, len(history) - 1 - delay_steps)].copy()
            if noise > 0:
                measured["max_activation"] += rng.normal(0, 0.08 * noise / 0.10)
                measured["mean_resource"] = float(np.clip(measured["mean_resource"] * (1 + rng.normal(0, noise)), 0.0, 1.2))
                measured["product_fraction"] = float(np.clip(measured["product_fraction"] + rng.normal(0, 0.08 * noise / 0.10), 0.0, 1.0))
                measured["warm_fraction"] = float(np.clip(measured["warm_fraction"] + rng.normal(0, 0.08 * noise / 0.10), 0.0, 1.0))
            u = choose_feedback(measured, max_u=1.0)
        max_command = max(max_command, u)
        if u > 1e-9:
            light_on += DT
            energy += u * DT

        production = capacity * u * resource * source
        activation += DT * (d_a * laplacian_neumann(activation, dx) - cooling * activation + drive * u * source)
        resource += DT * (d_o * laplacian_neumann(resource, dx) + recovery * (1 - resource) - consumption * production)
        product += DT * production
        activation = np.clip(activation, 0, 2.0)
        resource = np.clip(resource, 0.02, 1.2)

    warm_fraction = float(np.mean((activation >= 0.58) & (activation <= 1.08)))
    overheat_fraction = float(np.mean(activation > 1.18))
    product_fraction = float(np.mean(product >= 0.62))
    resource_failure = float(np.mean((resource < 0.18) & (product < 0.62)))
    chance_success = bool(warm_fraction >= 0.50 and product_fraction >= 0.50 and overheat_fraction <= 0.05 and resource_failure <= 0.35)
    duration = N_STEPS * DT
    average_u = energy / duration
    recovery_time = 1 / (recovery + EPS)
    diffusion_time = length * length / d_a
    observation_quality = np.exp(-float(scenario["delay_s"]) / (0.35 * diffusion_time + EPS)) / (1 + 6 * noise**2)
    dose_capacity_number = capacity * np.sqrt(recovery / 0.08) / (length**1.15 + EPS)
    drive_number = average_u * drive * length * length / (recovery + EPS)
    reaction_number = average_u * capacity / (recovery + EPS)
    reaction_drive_balance = reaction_number / (drive_number + EPS)
    return {
        "controller": controller,
        "model_family": draw["model_family"],
        "atlas_uncertainty_id": int(draw["atlas_uncertainty_id"]),
        "draw_id": int(draw["draw_id"]),
        "scenario": scenario["scenario"],
        "delay_s": float(scenario["delay_s"]),
        "noise_fraction": noise,
        "geometry_scale": length,
        "recovery_rate": recovery,
        "dose_capacity": capacity,
        "drive_coupling": drive,
        "resource_consumption": consumption,
        "protocol_duration_s": duration,
        "light_on_time_s": light_on,
        "delivered_energy": energy,
        "average_intensity": average_u,
        "duty_proxy": light_on / duration,
        "max_commanded_intensity": max_command,
        "max_activation": float(activation.max()),
        "warm_fraction": warm_fraction,
        "overheat_fraction": overheat_fraction,
        "product_fraction": product_fraction,
        "resource_failure_fraction": resource_failure,
        "chance_success": chance_success,
        "recovery_time_s": recovery_time,
        "diffusion_time_s": diffusion_time,
        "observation_quality": observation_quality,
        "dose_capacity_number": dose_capacity_number,
        "drive_number": drive_number,
        "reaction_number": reaction_number,
        "reaction_drive_balance": reaction_drive_balance,
        "log_dose_capacity_number": float(np.log10(dose_capacity_number + EPS)),
        "log_reaction_number": float(np.log10(reaction_number + EPS)),
        "log_reaction_drive_balance": float(np.log10(reaction_drive_balance + EPS)),
        "log_diffusion_time_s": float(np.log10(diffusion_time + EPS)),
    }


def run_atlas() -> pd.DataFrame:
    rows = []
    total = len(SEEDS) * N_DRAWS * 4 * len(SCENARIOS) * 3
    done = 0
    for seed in SEEDS:
        draws = latin_draws(seed)
        for _, draw in draws.iterrows():
            for scenario_index, scenario in enumerate(SCENARIOS):
                for controller_index, controller in enumerate(["fixed_reference", "feedback", "predictive_feedback"]):
                    run_seed = seed + int(draw["atlas_uncertainty_id"]) * 1009 + scenario_index * 131 + controller_index * 917
                    row = simulate(draw, scenario, controller, run_seed)
                    row["validation_seed"] = f"abstract_{seed}"
                    rows.append(row)
                    done += 1
                    if done % 240 == 0:
                        print(f"processed {done}/{total} abstract simulations", flush=True)
    return pd.DataFrame(rows)


def evaluate_boundary(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["log_reaction_number*observation_quality"] = df["log_reaction_number"] * df["observation_quality"]
    features = ["log_dose_capacity_number", "log_reaction_number*observation_quality", "log_reaction_drive_balance"]
    details = []
    preds = []
    for train_seed in sorted(df["validation_seed"].unique()):
        test_seed = [s for s in sorted(df["validation_seed"].unique()) if s != train_seed][0]
        train = df["validation_seed"] == train_seed
        test = df["validation_seed"] == test_seed
        w, mu, sd = logistic_fit(df.loc[train, features].to_numpy(float), df.loc[train, "chance_success"].astype(int).to_numpy())
        p = logistic_predict(df.loc[test, features].to_numpy(float), w, mu, sd)
        test_df = df.loc[test].copy()
        test_df["predicted_probability"] = p
        test_df["trained_on_seed"] = train_seed
        preds.append(test_df)
        details.append({"trained_on_seed": train_seed, "tested_on_seed": test_seed, "model_family": "all", "auc": auc_score(test_df["chance_success"].astype(int), p), "success_rate": float(test_df["chance_success"].mean())})
        for family, g in test_df.groupby("model_family"):
            details.append({"trained_on_seed": train_seed, "tested_on_seed": test_seed, "model_family": family, "auc": auc_score(g["chance_success"].astype(int), g["predicted_probability"]), "success_rate": float(g["chance_success"].mean())})
    pred = pd.concat(preds, ignore_index=True)
    q = pred["predicted_probability"].quantile([0.33, 0.67]).to_numpy()
    pred["controllability_class"] = pd.cut(pred["predicted_probability"], [-np.inf, q[0], q[1], np.inf], labels=["low", "transitional", "high"])
    return pd.DataFrame(details), pred


def paired_reclassification(pred: pd.DataFrame) -> pd.DataFrame:
    class_map = {"low": 0, "transitional": 1, "high": 2}
    pred["class_index"] = pred["controllability_class"].map(class_map).astype(int)
    keys = ["validation_seed", "model_family", "atlas_uncertainty_id", "scenario"]
    full = pred[pred["controller"] == "feedback"][keys + ["chance_success", "predicted_probability", "class_index", "controllability_class"]].rename(
        columns={k: f"feedback_{k}" for k in ["chance_success", "predicted_probability", "class_index", "controllability_class"]}
    )
    predictive = pred[pred["controller"] == "predictive_feedback"][keys + ["chance_success", "predicted_probability", "class_index", "controllability_class"]].rename(
        columns={k: f"predictive_{k}" for k in ["chance_success", "predicted_probability", "class_index", "controllability_class"]}
    )
    pair = full.merge(predictive, on=keys, how="inner", validate="one_to_one")
    pair["class_shift"] = pair["predictive_class_index"] - pair["feedback_class_index"]
    pair["success_uplift"] = pair["predictive_chance_success"].astype(int) - pair["feedback_chance_success"].astype(int)
    return pair


def make_figure(details: pd.DataFrame, pred: pd.DataFrame, pair: pd.DataFrame, out_path: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    ax = axes[0, 0]
    h = details[details["model_family"] != "all"].copy()
    h["group"] = h["tested_on_seed"] + " / " + h["model_family"]
    ax.barh(h["group"], h["auc"], color="#2f7f4f")
    ax.axvline(0.70, color="black", linestyle="--", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("AUC")
    ax.set_title("A. Abstract model transfer")

    ax = axes[0, 1]
    classes = pred.groupby("controllability_class", observed=False).agg(observed_success=("chance_success", "mean"), predicted=("predicted_probability", "mean")).reset_index()
    ax.bar(classes["controllability_class"].astype(str), classes["observed_success"], color=["#c44e52", "#ddcc77", "#2f7f4f"])
    ax.plot(classes["controllability_class"].astype(str), classes["predicted"], color="black", marker="o")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Success probability")
    ax.set_title("B. Abstract controllability classes")

    ax = axes[1, 0]
    pivot = pred.pivot_table(index="model_family", columns="controller", values="chance_success", aggfunc="mean")
    pivot[["fixed_reference", "feedback", "predictive_feedback"]].plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Success probability")
    ax.set_title("C. Controllers in abstract model")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1, 1]
    class_uplift = pair.groupby("feedback_controllability_class", observed=False).agg(success_uplift=("success_uplift", "mean"), class_shift=("class_shift", "mean")).reset_index()
    ax.bar(class_uplift["feedback_controllability_class"].astype(str), class_uplift["success_uplift"], color="#2f7f4f")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Predictive paired success uplift")
    ax.set_title("D. Predictive recovery by class")

    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_NOTES.mkdir(parents=True, exist_ok=True)
    outcomes = run_atlas()
    outcomes.to_csv(OUT_DATA / "abstract_reaction_diffusion_outcomes.csv", index=False)
    details, pred = evaluate_boundary(outcomes)
    details.to_csv(OUT_DATA / "abstract_boundary_cross_seed_validation.csv", index=False)
    pred.to_csv(OUT_DATA / "abstract_boundary_predictions.csv", index=False)
    pair = paired_reclassification(pred)
    pair.to_csv(OUT_DATA / "abstract_predictive_reclassification.csv", index=False)
    classes = pred.groupby("controllability_class", observed=False).agg(n=("chance_success", "size"), observed_success=("chance_success", "mean"), mean_predicted=("predicted_probability", "mean")).reset_index()
    controller_summary = outcomes.groupby(["controller", "model_family"], as_index=False).agg(success=("chance_success", "mean"), overheat=("overheat_fraction", "mean"), product=("product_fraction", "mean"))
    classes.to_csv(OUT_DATA / "abstract_controllability_classes.csv", index=False)
    controller_summary.to_csv(OUT_DATA / "abstract_controller_summary.csv", index=False)
    metadata = {
        "description": "Dimensionless abstract activator-resource-product reaction-diffusion benchmark.",
        "seeds": SEEDS,
        "features": ["log_dose_capacity_number", "log_reaction_number*observation_quality", "log_reaction_drive_balance"],
        "validation": "Leave-one-abstract-seed-out.",
    }
    (OUT_DATA / "abstract_reaction_diffusion_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    make_figure(details, pred, pair, OUT_FIG / "figure_9_abstract_reaction_diffusion_transfer.png")

    fam = details[details["model_family"] != "all"]
    min_auc = float(fam["auc"].min())
    median_auc = float(fam["auc"].median())
    class_monotonic = classes["observed_success"].is_monotonic_increasing
    pair_low = pair[pair["feedback_controllability_class"].astype(str) == "low"]
    low_uplift = float(pair_low["success_uplift"].mean()) if len(pair_low) else float("nan")
    decision = (
        "CONTINUE: class structure transfers to the abstract reaction-diffusion benchmark."
        if min_auc >= 0.65 and class_monotonic
        else "PIVOT: abstract transfer is not strong enough; focus on original-system predictive recovery."
    )
    lines = [
        "# Step 9 Abstract Reaction-Diffusion Transfer",
        "",
        "Question: does the seed-stable controllability class structure transfer beyond the nanoparticle/spheroid model?",
        "",
        "## Model",
        "",
        "A dimensionless one-dimensional activator-resource-product reaction-diffusion system was simulated with delayed/noisy feedback. Success required spatial activation in a target window, product accumulation, low overactivation, and limited resource-depletion failure.",
        "",
        "## Cross-Seed Boundary Validation",
        "",
        "```",
        details.round(3).to_string(index=False),
        "```",
        "",
        "## Controllability Classes",
        "",
        "```",
        classes.round(3).to_string(index=False),
        "```",
        "",
        "## Controller Summary",
        "",
        "```",
        controller_summary.round(3).to_string(index=False),
        "```",
        "",
        "## Step-Gate Decision",
        "",
        f"Minimum abstract seed-family AUC: {min_auc:.3f}.",
        f"Median abstract seed-family AUC: {median_auc:.3f}.",
        f"Low-class predictive paired success uplift: {low_uplift:.3f}.",
        decision,
    ]
    (OUT_NOTES / "step9_abstract_reaction_diffusion_transfer_result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote abstract reaction-diffusion transfer outputs")
    print(details.round(3).to_string(index=False))
    print(classes.round(3).to_string(index=False))
    print(decision)


if __name__ == "__main__":
    main()
