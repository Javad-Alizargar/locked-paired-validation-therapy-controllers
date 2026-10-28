#!/usr/bin/env python3
"""First-pass reproducible magneto-plasmonic spheroid model.

This is a screening-level model, not the final manuscript model. Its purpose is to
turn the evidence spine into executable phase diagrams while keeping all assumptions
visible and easy to replace.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "simulations"
FIG = ROOT / "figures" / "first_model"


@dataclass(frozen=True)
class EvidenceDefaults:
    # Evidence spine: 10.1021/acsomega.5c05623
    wavelength_min_nm: float = 400.0
    wavelength_max_nm: float = 1500.0
    nanorod_length_min_nm: float = 40.0
    nanorod_length_max_nm: float = 100.0
    nanorod_width_min_nm: float = 10.0
    nanorod_width_max_nm: float = 30.0
    medium_refractive_index: float = 1.33
    optical_constants_source: str = "Rakic 1998, DOI 10.1364/AO.37.005271; OpenAlex raw DOI record saved"

    # Evidence spine: 10.1038/s41598-025-10967-2
    nir1_min_nm: float = 750.0
    nir1_max_nm: float = 950.0
    nir2_min_nm: float = 1000.0
    nir2_max_nm: float = 1200.0
    benchmark_laser_w_cm2: float = 1.0
    benchmark_irradiation_s: float = 300.0
    volume_fraction_min: float = 0.001
    volume_fraction_max: float = 0.005

    # Evidence spine: 10.1002/cnm.70074
    sar_min_mw_per_kg: float = 0.1
    sar_max_mw_per_kg: float = 2.0
    perfusion_min_s: float = 0.0
    perfusion_max_s: float = 0.036

    # Evidence spine: 10.1016/j.jphotobiol.2016.09.031
    oxygen_initial_um: float = 40.0
    oxygen_beta_um: float = 11.9
    oxygen_delta_um: float = 33.0
    sigma_per_um: float = 1.8e-5
    xi_cm2_mw_s: float = 55e-3
    oxygen_perfusion_um_s: float = 1.7
    singlet_oxygen_threshold_mm: float = 0.67


@dataclass(frozen=True)
class ModelConfig:
    spheroid_radius_m: float = 500e-6
    radial_nodes: int = 96
    duration_s: float = 300.0
    dt_s: float = 10.0
    ros_dt_s: float = 0.5
    pulse_period_s: float = 60.0
    body_temperature_c: float = 37.0
    tissue_density_kg_m3: float = 1060.0
    tissue_cp_j_kg_k: float = 3600.0
    tissue_k_w_m_k: float = 0.50
    blood_temperature_c: float = 37.0
    blood_cp_j_kg_k: float = 3617.0
    blood_density_kg_m3: float = 1070.0
    optical_heat_coeff_w_m3: float = 7.0e8
    nanoparticle_surface_bias: float = 0.30
    external_cooling_s: float = 0.0
    photosensitizer_um_at_reference_phi: float = 1.0
    reference_volume_fraction: float = 0.003


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)


def plasmon_peak_nm(length_nm: float, width_nm: float) -> float:
    """Phenomenological nanorod peak used only for screening.

    The evidence spine supports geometry sweeps and Mie-Gans-style absorption formulas,
    but a full dielectric-function implementation is still a later step. This relation
    makes aspect-ratio trends explicit and bounded in the extracted wavelength range.
    """
    aspect_ratio = max(length_nm / width_nm, 1.0)
    return float(np.clip(520.0 + 95.0 * (aspect_ratio - 1.0), 520.0, 1200.0))


def relative_absorption(wavelength_nm: float, peak_nm: float, bandwidth_nm: float) -> float:
    return 1.0 / (1.0 + ((wavelength_nm - peak_nm) / bandwidth_nm) ** 2)


def gold_dielectric_rakic(wavelength_nm: np.ndarray | float) -> np.ndarray:
    """Gold dielectric function from Rakic-style Drude-Lorentz parameters.

    Energy is in eV. The parameters are a compact, code-level representation of
    Rakic et al. 1998 optical constants for Au and should be checked against the
    source table during manuscript finalization.
    """
    wl = np.asarray(wavelength_nm, dtype=float)
    energy = 1239.841984 / wl
    eps_inf = 1.53
    plasma = 9.03
    f = np.array([0.760, 0.024, 0.010, 0.071, 0.601, 4.384])
    gamma = np.array([0.053, 0.241, 0.345, 0.870, 2.494, 2.214])
    omega = np.array([0.000, 0.415, 0.830, 2.969, 4.304, 13.32])
    eps = eps_inf - f[0] * plasma**2 / (energy * (energy + 1j * gamma[0]))
    for j in range(1, len(f)):
        eps += f[j] * plasma**2 / (omega[j] ** 2 - energy**2 - 1j * gamma[j] * energy)
    return eps


def depolarization_long_axis(length_nm: float, width_nm: float) -> float:
    a = max(length_nm / 2.0, width_nm / 2.0)
    b = min(width_nm / 2.0, a * 0.999999)
    eccentricity = math.sqrt(max(1.0 - (b / a) ** 2, 1e-12))
    return ((1.0 - eccentricity**2) / eccentricity**2) * (
        -1.0 + (1.0 / (2.0 * eccentricity)) * math.log((1.0 + eccentricity) / (1.0 - eccentricity))
    )


def gans_absorption_cross_section_m2(
    wavelength_nm: np.ndarray | float,
    length_nm: float,
    width_nm: float,
    medium_refractive_index: float,
) -> np.ndarray:
    """Quasi-static Gans absorption cross-section for a prolate gold nanorod.

    The expression follows the ellipsoidal polarizability form extracted in the
    evidence spine. It returns absorption-like cross-section by taking k Im(alpha).
    """
    wl_m = np.asarray(wavelength_nm, dtype=float) * 1e-9
    eps_gold = gold_dielectric_rakic(wavelength_nm)
    eps_m = medium_refractive_index**2
    a = length_nm * 0.5e-9
    b = width_nm * 0.5e-9
    c = b
    volume = 4.0 / 3.0 * math.pi * a * b * c
    l1 = depolarization_long_axis(length_nm, width_nm)
    alpha = volume * (eps_gold - eps_m) / (eps_m + l1 * (eps_gold - eps_m))
    k = 2.0 * math.pi * medium_refractive_index / wl_m
    c_abs = k * np.imag(alpha)
    return np.maximum(c_abs, 0.0)


def relative_gans_absorption(
    wavelength_nm: float,
    length_nm: float,
    width_nm: float,
    medium_refractive_index: float,
) -> float:
    sweep = np.linspace(400.0, 1500.0, 300)
    spectrum = gans_absorption_cross_section_m2(sweep, length_nm, width_nm, medium_refractive_index)
    target = gans_absorption_cross_section_m2(np.array([wavelength_nm]), length_nm, width_nm, medium_refractive_index)[0]
    max_spectrum = float(np.max(spectrum))
    return float(np.clip(target / max_spectrum, 0.0, 1.0)) if max_spectrum > 0 else 0.0


def gans_peak_nm(length_nm: float, width_nm: float, medium_refractive_index: float) -> float:
    sweep = np.linspace(400.0, 1500.0, 300)
    spectrum = gans_absorption_cross_section_m2(sweep, length_nm, width_nm, medium_refractive_index)
    return float(sweep[int(np.argmax(spectrum))])


def radial_laplacian(values: np.ndarray, r: np.ndarray, dr: float, boundary_value: float) -> np.ndarray:
    lap = np.zeros_like(values)
    lap[0] = 6.0 * (values[1] - values[0]) / dr**2
    rr = r[1:-1]
    lap[1:-1] = (
        (values[2:] - 2.0 * values[1:-1] + values[:-2]) / dr**2
        + 2.0 / rr * (values[2:] - values[:-2]) / (2.0 * dr)
    )
    ghost = boundary_value
    lap[-1] = (ghost - 2.0 * values[-1] + values[-2]) / dr**2 + 2.0 / r[-1] * (ghost - values[-2]) / (2.0 * dr)
    return lap


def build_pennes_backward_euler_tridiagonal(
    r: np.ndarray,
    dr: float,
    dt_s: float,
    alpha_m2_s: float,
    perfusion_loss_s: float,
    external_cooling_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build tridiagonal coefficients for implicit radial Pennes heat stepping.

    The state variable is temperature rise above blood/body temperature. The outer
    boundary is zero-flux plus optional uniform cooling, representing an embedded
    spheroid/tumor region rather than a tissue surface clamped to 37 C.
    """
    n = len(r)
    lower = np.zeros(n)
    diag = np.ones(n)
    upper = np.zeros(n)
    sink = perfusion_loss_s + external_cooling_s
    # Center node, spherical symmetry: laplacian = 6 * (T_1 - T_0) / dr^2
    diag[0] = 1.0 + dt_s * (6.0 * alpha_m2_s / dr**2 + sink)
    upper[0] = -dt_s * 6.0 * alpha_m2_s / dr**2
    for i in range(1, n - 1):
        lower_l = 1.0 / dr**2 - 1.0 / (r[i] * dr)
        diag_l = -2.0 / dr**2
        upper_l = 1.0 / dr**2 + 1.0 / (r[i] * dr)
        lower[i] = -dt_s * alpha_m2_s * lower_l
        diag[i] = 1.0 - dt_s * (alpha_m2_s * diag_l - sink)
        upper[i] = -dt_s * alpha_m2_s * upper_l
    # Outer node, zero-flux approximation: laplacian = 2 * (T_{n-2} - T_{n-1}) / dr^2
    lower[-1] = -dt_s * 2.0 * alpha_m2_s / dr**2
    diag[-1] = 1.0 + dt_s * (2.0 * alpha_m2_s / dr**2 + sink)
    return lower, diag, upper


def solve_tridiagonal(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Thomas algorithm for a tridiagonal linear system."""
    n = len(diag)
    c_prime = np.zeros(n)
    d_prime = np.zeros(n)
    c_prime[0] = upper[0] / diag[0]
    d_prime[0] = rhs[0] / diag[0]
    for i in range(1, n):
        denom = diag[i] - lower[i] * c_prime[i - 1]
        c_prime[i] = upper[i] / denom if i < n - 1 else 0.0
        d_prime[i] = (rhs[i] - lower[i] * d_prime[i - 1]) / denom
    solution = np.zeros(n)
    solution[-1] = d_prime[-1]
    for i in range(n - 2, -1, -1):
        solution[i] = d_prime[i] - c_prime[i] * solution[i + 1]
    return solution


def solve_pennes_temperature(
    q_total_w_m3: np.ndarray,
    r: np.ndarray,
    dr: float,
    perfusion_loss_s: float,
    cfg: ModelConfig,
    duration_s: float,
) -> np.ndarray:
    alpha = cfg.tissue_k_w_m_k / (cfg.tissue_density_kg_m3 * cfg.tissue_cp_j_kg_k)
    source_k_s = q_total_w_m3 / (cfg.tissue_density_kg_m3 * cfg.tissue_cp_j_kg_k)
    lower, diag, upper = build_pennes_backward_euler_tridiagonal(
        r,
        dr,
        cfg.dt_s,
        alpha,
        perfusion_loss_s,
        cfg.external_cooling_s,
    )
    rise = np.zeros_like(r)
    steps = int(round(duration_s / cfg.dt_s))
    rhs_source = cfg.dt_s * source_k_s
    for _ in range(steps):
        rise = solve_tridiagonal(lower, diag, upper, rise + rhs_source)
    return cfg.body_temperature_c + rise


def nanoparticle_profile(r: np.ndarray, radius: float, surface_bias: float) -> np.ndarray:
    core = np.ones_like(r)
    shell = np.exp(-((radius - r) / (surface_bias * radius)) ** 2)
    profile = 0.35 * core + 0.65 * shell
    return profile / profile.mean()


def optical_fluence_profile(r: np.ndarray, radius: float, wavelength_nm: float, intensity_w_cm2: float) -> np.ndarray:
    # Screening approximation: deeper penetration in NIR windows than visible.
    if 750.0 <= wavelength_nm <= 950.0:
        penetration = 900e-6
    elif 1000.0 <= wavelength_nm <= 1200.0:
        penetration = 700e-6
    else:
        penetration = 350e-6
    depth = radius - r
    return intensity_w_cm2 * 1e4 * np.exp(-depth / penetration)


def simulate_case(
    wavelength_nm: float,
    length_nm: float,
    width_nm: float,
    volume_fraction: float,
    sar_mw_per_kg: float,
    intensity_w_cm2: float,
    perfusion_s: float,
    cfg: ModelConfig,
    ev: EvidenceDefaults,
    duty_cycle: float = 1.0,
) -> dict:
    duty_cycle = float(np.clip(duty_cycle, 1e-6, 1.0))
    protocol_duration_s = cfg.duration_s / duty_cycle
    radius = cfg.spheroid_radius_m
    r = np.linspace(0.0, radius, cfg.radial_nodes)
    dr = r[1] - r[0]
    peak_nm = gans_peak_nm(length_nm, width_nm, ev.medium_refractive_index)
    absorption = relative_gans_absorption(wavelength_nm, length_nm, width_nm, ev.medium_refractive_index)
    np_profile = nanoparticle_profile(r, radius, cfg.nanoparticle_surface_bias)
    fluence = optical_fluence_profile(r, radius, wavelength_nm, intensity_w_cm2)
    q_opt = cfg.optical_heat_coeff_w_m3 * volume_fraction * absorption * (fluence / 1e4) * np_profile
    q_mag = cfg.tissue_density_kg_m3 * volume_fraction * (sar_mw_per_kg * 1e6) * np_profile
    q_total = duty_cycle * (q_opt + q_mag)
    perfusion_term = perfusion_s * cfg.blood_density_kg_m3 * cfg.blood_cp_j_kg_k / (cfg.tissue_density_kg_m3 * cfg.tissue_cp_j_kg_k)
    temperature = solve_pennes_temperature(q_total, r, dr, perfusion_term, cfg, protocol_duration_s)
    oxygen_um = ev.oxygen_initial_um * (0.45 + 0.55 * (r / radius) ** 1.6)
    oxygen_um = np.maximum(oxygen_um, 2.0)
    s0_um = cfg.photosensitizer_um_at_reference_phi * (volume_fraction / cfg.reference_volume_fraction) * np_profile
    ros_dose_um = np.zeros_like(r)
    fluence_mw_cm2 = fluence * 0.1
    ros_steps = int(round(protocol_duration_s / cfg.ros_dt_s))
    for step in range(ros_steps):
        time_s = step * cfg.ros_dt_s
        pulse_on = (time_s % cfg.pulse_period_s) < (duty_cycle * cfg.pulse_period_s)
        if pulse_on:
            oxygen_factor = oxygen_um / (oxygen_um + ev.oxygen_beta_um)
            generation_um_s = ev.xi_cm2_mw_s * fluence_mw_cm2 * s0_um * oxygen_factor
            ros_dose_um += cfg.ros_dt_s * generation_um_s
            oxygen_um -= cfg.ros_dt_s * generation_um_s
            # Photobleaching from the macroscopic model. The delta term weakens bleaching at low S0.
            s0_um -= cfg.ros_dt_s * ev.sigma_per_um * generation_um_s * s0_um / (s0_um + ev.oxygen_delta_um)
            s0_um = np.maximum(s0_um, 0.0)
        oxygen_um += cfg.ros_dt_s * ev.oxygen_perfusion_um_s * (1.0 - oxygen_um / ev.oxygen_initial_um)
        oxygen_um = np.clip(oxygen_um, 0.5, ev.oxygen_initial_um)
    ros_dose_mm = ros_dose_um / 1000.0
    max_temperature = float(temperature.max())
    warm_fraction = float(np.mean((temperature >= 43.0) & (temperature <= 50.0)))
    overheat_fraction = float(np.mean(temperature > 55.0))
    ros_fraction = float(np.mean(ros_dose_mm >= ev.singlet_oxygen_threshold_mm))
    hypoxic_failure_fraction = float(np.mean((oxygen_um < 5.0) & (ros_dose_mm < ev.singlet_oxygen_threshold_mm)))
    thermal_window_score = warm_fraction - 1.5 * overheat_fraction
    ros_score = ros_fraction - 0.5 * hypoxic_failure_fraction
    dual_modality_score = warm_fraction * (0.25 + 0.75 * ros_fraction) - 1.5 * overheat_fraction - 0.5 * hypoxic_failure_fraction
    score = thermal_window_score
    return {
        "wavelength_nm": wavelength_nm,
        "length_nm": length_nm,
        "width_nm": width_nm,
        "plasmon_peak_nm": peak_nm,
        "relative_absorption": absorption,
        "volume_fraction": volume_fraction,
        "sar_mw_per_kg": sar_mw_per_kg,
        "intensity_w_cm2": intensity_w_cm2,
        "perfusion_s": perfusion_s,
        "duty_cycle": duty_cycle,
        "protocol_duration_s": protocol_duration_s,
        "max_temperature_c": max_temperature,
        "surface_temperature_c": float(temperature[-1]),
        "center_temperature_c": float(temperature[0]),
        "warm_fraction": warm_fraction,
        "overheat_fraction": overheat_fraction,
        "ros_threshold_fraction": ros_fraction,
        "hypoxic_failure_fraction": hypoxic_failure_fraction,
        "thermal_window_score": thermal_window_score,
        "ros_score": ros_score,
        "dual_modality_score": dual_modality_score,
        "therapeutic_score": score,
        "r_um": (r * 1e6).tolist(),
        "temperature_c_profile": temperature.tolist(),
        "oxygen_um_profile": oxygen_um.tolist(),
        "ros_mm_profile": ros_dose_mm.tolist(),
        "photosensitizer_um_profile": s0_um.tolist(),
    }


def write_csv(path: Path, rows: list[dict], omit_profiles: bool = True) -> None:
    excluded = {"r_um", "temperature_c_profile", "oxygen_um_profile", "ros_mm_profile", "photosensitizer_um_profile"} if omit_profiles else set()
    fieldnames = [key for key in rows[0].keys() if key not in excluded]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def color_map(value: float, vmin: float, vmax: float) -> tuple[int, int, int]:
    t = 0.0 if vmax == vmin else (value - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))
    stops = [
        (0.00, (41, 65, 120)),
        (0.25, (84, 150, 168)),
        (0.50, (242, 222, 121)),
        (0.75, (224, 112, 67)),
        (1.00, (121, 35, 52)),
    ]
    for (a, ca), (b, cb) in zip(stops[:-1], stops[1:]):
        if a <= t <= b:
            u = (t - a) / (b - a)
            return tuple(int(ca[i] + u * (cb[i] - ca[i])) for i in range(3))
    return stops[-1][1]


def tick_label(value: float, span: float) -> str:
    if span >= 100:
        return f"{value:.0f}"
    if span >= 10:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{value:.2f}".rstrip("0").rstrip(".")


def make_heatmap(
    matrix: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    vmin = float(np.nanmin(matrix) if vmin is None else vmin)
    vmax = float(np.nanmax(matrix) if vmax is None else vmax)
    cell = 10
    left, top, right, bottom = 112, 48, 26, 72
    h, w = matrix.shape
    image = Image.new("RGB", (left + w * cell + right, top + h * cell + bottom), "white")
    draw = ImageDraw.Draw(image)
    for iy in range(h):
        for ix in range(w):
            color = color_map(float(matrix[iy, ix]), vmin, vmax)
            x0 = left + ix * cell
            y0 = top + (h - 1 - iy) * cell
            draw.rectangle([x0, y0, x0 + cell, y0 + cell], fill=color)
    draw.rectangle([left, top, left + w * cell, top + h * cell], outline=(30, 30, 30), width=1)
    font = ImageFont.load_default()
    draw.text((left, 16), title, fill=(0, 0, 0), font=font)
    draw.text((left + w * cell // 2 - 45, top + h * cell + 34), xlabel, fill=(0, 0, 0), font=font)
    draw.text((8, top + h * cell // 2), ylabel, fill=(0, 0, 0), font=font)
    for frac in [0, 0.25, 0.5, 0.75, 1.0]:
        x = left + int(frac * w * cell)
        value = x_values[0] + frac * (x_values[-1] - x_values[0])
        draw.line([(x, top + h * cell), (x, top + h * cell + 5)], fill=(0, 0, 0))
        draw.text((x - 18, top + h * cell + 9), tick_label(value, x_values[-1] - x_values[0]), fill=(0, 0, 0), font=font)
        y = top + h * cell - int(frac * h * cell)
        yvalue = y_values[0] + frac * (y_values[-1] - y_values[0])
        draw.line([(left - 5, y), (left, y)], fill=(0, 0, 0))
        draw.text((left - 72, y - 5), tick_label(yvalue, y_values[-1] - y_values[0]), fill=(0, 0, 0), font=font)
    bar_x = left + w * cell + 8
    for i in range(h * cell):
        val = vmax - (i / max(1, h * cell - 1)) * (vmax - vmin)
        draw.line([(bar_x, top + i), (bar_x + 12, top + i)], fill=color_map(val, vmin, vmax))
    draw.rectangle([bar_x, top, bar_x + 12, top + h * cell], outline=(30, 30, 30))
    draw.text((bar_x - 2, top - 14), f"{vmax:.2g}", fill=(0, 0, 0), font=font)
    draw.text((bar_x - 2, top + h * cell + 4), f"{vmin:.2g}", fill=(0, 0, 0), font=font)
    image.save(out_path)


def make_profile_plot(case: dict, out_path: Path) -> None:
    r = np.array(case["r_um"])
    profiles = [
        ("T (C)", np.array(case["temperature_c_profile"]), (121, 35, 52), 37.0, 60.0),
        ("O2 (uM)", np.array(case["oxygen_um_profile"]), (41, 65, 120), 0.0, 42.0),
        ("ROS (mM)", np.array(case["ros_mm_profile"]), (84, 150, 83), 0.0, max(1.0, max(case["ros_mm_profile"]))),
    ]
    image = Image.new("RGB", (1040, 430), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    left, top, width, height = 82, 42, 720, 300
    draw.text((left, 14), "Radial end-state profiles for best screening case", fill=(0, 0, 0), font=font)
    draw.rectangle([left, top, left + width, top + height], outline=(40, 40, 40))
    for name, vals, color, ymin, ymax in profiles:
        points = []
        for x_um, yval in zip(r, vals):
            x = left + int(width * x_um / r[-1])
            y = top + height - int(height * (float(yval) - ymin) / (ymax - ymin))
            points.append((x, max(top, min(top + height, y))))
        if len(points) > 1:
            draw.line(points, fill=color, width=3)
    for frac in [0, 0.25, 0.5, 0.75, 1]:
        x = left + int(frac * width)
        draw.line([(x, top + height), (x, top + height + 5)], fill=(0, 0, 0))
        draw.text((x - 15, top + height + 10), f"{frac*r[-1]:.0f}", fill=(0, 0, 0), font=font)
    draw.text((left + width // 2 - 60, top + height + 36), "radius (um)", fill=(0, 0, 0), font=font)
    for i, (name, _, color, _, _) in enumerate(profiles):
        y = top + 18 + i * 22
        draw.line([(left + width + 20, y + 6), (left + width + 52, y + 6)], fill=color, width=3)
        draw.text((left + width + 60, y), name, fill=(0, 0, 0), font=font)
    image.save(out_path)


def run() -> None:
    ensure_dirs()
    ev = EvidenceDefaults()
    cfg = ModelConfig()
    wavelengths = np.linspace(650.0, 1100.0, 32)
    sar_values = np.linspace(0.0, 2.0, 28)
    intensity_values = np.linspace(0.2, 1.4, 28)
    phi_values = np.linspace(ev.volume_fraction_min, ev.volume_fraction_max, 28)
    duty_values = np.linspace(0.1, 1.0, 28)
    baseline = {
        "length_nm": 70.0,
        "width_nm": 18.0,
        "volume_fraction": 0.003,
        "intensity_w_cm2": ev.benchmark_laser_w_cm2,
        "perfusion_s": 0.009,
    }
    sar_rows = []
    score_sar = np.zeros((len(sar_values), len(wavelengths)))
    max_t_sar = np.zeros_like(score_sar)
    ros_sar = np.zeros_like(score_sar)
    dual_sar = np.zeros_like(score_sar)
    for iy, sar in enumerate(sar_values):
        for ix, wavelength in enumerate(wavelengths):
            result = simulate_case(wavelength, sar_mw_per_kg=float(sar), cfg=cfg, ev=ev, **baseline)
            sar_rows.append(result)
            score_sar[iy, ix] = result["therapeutic_score"]
            max_t_sar[iy, ix] = result["max_temperature_c"]
            ros_sar[iy, ix] = result["ros_threshold_fraction"]
            dual_sar[iy, ix] = result["dual_modality_score"]
    phi_rows = []
    score_phi = np.zeros((len(phi_values), len(intensity_values)))
    max_t_phi = np.zeros_like(score_phi)
    dual_phi = np.zeros_like(score_phi)
    for iy, phi in enumerate(phi_values):
        for ix, intensity in enumerate(intensity_values):
            result = simulate_case(
                808.0,
                volume_fraction=float(phi),
                intensity_w_cm2=float(intensity),
                sar_mw_per_kg=0.1,
                cfg=cfg,
                ev=ev,
                **{k: v for k, v in baseline.items() if k not in {"volume_fraction", "intensity_w_cm2"}},
            )
            phi_rows.append(result)
            score_phi[iy, ix] = result["therapeutic_score"]
            max_t_phi[iy, ix] = result["max_temperature_c"]
            dual_phi[iy, ix] = result["dual_modality_score"]
    geometry_rows = []
    lengths = np.linspace(ev.nanorod_length_min_nm, ev.nanorod_length_max_nm, 25)
    widths = np.linspace(ev.nanorod_width_min_nm, ev.nanorod_width_max_nm, 25)
    peak_map = np.zeros((len(widths), len(lengths)))
    abs_808 = np.zeros_like(peak_map)
    for iy, width in enumerate(widths):
        for ix, length in enumerate(lengths):
            peak = gans_peak_nm(float(length), float(width), ev.medium_refractive_index)
            absorption = relative_gans_absorption(808.0, float(length), float(width), ev.medium_refractive_index)
            peak_map[iy, ix] = peak
            abs_808[iy, ix] = absorption
            geometry_rows.append({"length_nm": length, "width_nm": width, "plasmon_peak_nm": peak, "relative_absorption_808": absorption})
    pulse_rows = []
    pulse_dual = np.zeros((len(duty_values), len(intensity_values)))
    pulse_temp = np.zeros_like(pulse_dual)
    pulse_ros = np.zeros_like(pulse_dual)
    for iy, duty in enumerate(duty_values):
        for ix, intensity in enumerate(intensity_values):
            result = simulate_case(
                808.0,
                volume_fraction=0.003,
                intensity_w_cm2=float(intensity),
                sar_mw_per_kg=0.0,
                duty_cycle=float(duty),
                cfg=cfg,
                ev=ev,
                **{k: v for k, v in baseline.items() if k not in {"volume_fraction", "intensity_w_cm2"}},
            )
            pulse_rows.append(result)
            pulse_dual[iy, ix] = result["dual_modality_score"]
            pulse_temp[iy, ix] = result["max_temperature_c"]
            pulse_ros[iy, ix] = result["ros_threshold_fraction"]
    all_screening_rows = sar_rows + phi_rows + pulse_rows
    best = max(all_screening_rows, key=lambda row: row["therapeutic_score"])
    best_dual = max(all_screening_rows, key=lambda row: row["dual_modality_score"])
    write_csv(OUT / "phase_wavelength_sar.csv", sar_rows)
    write_csv(OUT / "phase_intensity_volume_fraction.csv", phi_rows)
    write_csv(OUT / "phase_pulse_duty_intensity.csv", pulse_rows)
    write_csv(OUT / "geometry_absorption_screen.csv", geometry_rows)
    (OUT / "best_case_profile.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    (OUT / "best_dual_modality_case_profile.json").write_text(json.dumps(best_dual, indent=2), encoding="utf-8")
    (OUT / "model_defaults.json").write_text(json.dumps({"evidence_defaults": asdict(ev), "model_config": asdict(cfg)}, indent=2), encoding="utf-8")
    make_heatmap(score_sar, wavelengths, sar_values, "Therapeutic score: wavelength vs SAR", "wavelength (nm)", "SAR (MW/kg)", FIG / "panel_a_score_wavelength_sar.png", vmin=-1.0, vmax=1.75)
    make_heatmap(max_t_sar, wavelengths, sar_values, "Max temperature (C): wavelength vs SAR", "wavelength (nm)", "SAR (MW/kg)", FIG / "panel_b_temperature_wavelength_sar.png", vmin=37.0, vmax=65.0)
    make_heatmap(ros_sar, wavelengths, sar_values, "ROS threshold fraction: wavelength vs SAR", "wavelength (nm)", "SAR (MW/kg)", FIG / "panel_c_ros_wavelength_sar.png", vmin=0.0, vmax=1.0)
    make_heatmap(score_phi, intensity_values, phi_values, "Therapeutic score: intensity vs nanoparticle fraction", "intensity (W/cm2)", "volume fraction", FIG / "panel_d_score_intensity_phi.png", vmin=-1.0, vmax=1.75)
    make_heatmap(max_t_phi, intensity_values, phi_values, "Max temperature (C): intensity vs nanoparticle fraction", "intensity (W/cm2)", "volume fraction", FIG / "panel_e_temperature_intensity_phi.png", vmin=37.0, vmax=75.0)
    make_heatmap(abs_808, lengths, widths, "Relative absorption at 808 nm: geometry screen", "length (nm)", "width (nm)", FIG / "panel_f_geometry_absorption_808.png", vmin=0.0, vmax=1.0)
    make_profile_plot(best, FIG / "panel_g_best_case_profiles.png")
    make_profile_plot(best_dual, FIG / "panel_j_best_dual_case_profiles.png")
    make_heatmap(dual_sar, wavelengths, sar_values, "Dual-modality score: wavelength vs SAR", "wavelength (nm)", "SAR (MW/kg)", FIG / "panel_h_dual_score_wavelength_sar.png", vmin=-1.0, vmax=1.0)
    make_heatmap(dual_phi, intensity_values, phi_values, "Dual-modality score: intensity vs nanoparticle fraction", "intensity (W/cm2)", "volume fraction", FIG / "panel_i_dual_score_intensity_phi.png", vmin=-1.0, vmax=1.0)
    make_heatmap(pulse_dual, intensity_values, duty_values, "Pulsed dual-modality score: intensity vs duty", "intensity (W/cm2)", "duty cycle", FIG / "panel_k_pulse_dual_score.png", vmin=-1.0, vmax=1.0)
    make_heatmap(pulse_temp, intensity_values, duty_values, "Pulsed max temperature (C): intensity vs duty", "intensity (W/cm2)", "duty cycle", FIG / "panel_l_pulse_temperature.png", vmin=37.0, vmax=75.0)
    make_heatmap(pulse_ros, intensity_values, duty_values, "Pulsed ROS threshold fraction: intensity vs duty", "intensity (W/cm2)", "duty cycle", FIG / "panel_m_pulse_ros.png", vmin=0.0, vmax=1.0)
    summary = {
        key: best[key]
        for key in [
            "wavelength_nm",
            "length_nm",
            "width_nm",
            "plasmon_peak_nm",
            "relative_absorption",
            "volume_fraction",
            "sar_mw_per_kg",
            "intensity_w_cm2",
            "perfusion_s",
            "duty_cycle",
            "protocol_duration_s",
            "max_temperature_c",
            "warm_fraction",
            "overheat_fraction",
            "ros_threshold_fraction",
            "hypoxic_failure_fraction",
            "thermal_window_score",
            "ros_score",
            "dual_modality_score",
            "therapeutic_score",
        ]
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
