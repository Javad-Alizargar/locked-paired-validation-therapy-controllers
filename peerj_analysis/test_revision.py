"""Regression checks for the sampling-unit and multiplicity defects found in review."""
from pathlib import Path
import numpy as np
import pandas as pd
from analysis import load_paired, bootstrap_indices, holm

ROOT = Path(__file__).resolve().parents[1]

def main():
    df = load_paired(ROOT)
    assert df.cluster_id.nunique() == 96
    blocks = df.groupby(["controller_family", "cluster_id"], observed=True)
    assert blocks.size().eq(48).all()
    assert blocks.model_family.nunique().eq(4).all()
    one = df[df.controller_family == "Constrained optimized"]
    groups = one.groupby("cluster_id", observed=True).validation_seed.first()
    idx = bootstrap_indices(one, 100, np.random.default_rng(42))
    for seed in groups.unique():
        assert (groups.to_numpy()[idx] == seed).sum(axis=1).tolist() == [32] * 100
    # Known Holm case exercises monotonicity and assignment back to original order.
    assert np.allclose(holm([.04, .001, .01]), [.04, .003, .02])
    result = pd.read_csv(ROOT / "peerj_analysis/results/primary_endpoints.csv")
    cols = ["thermal_failure", "ros_failure", "hypoxic_failure", "overheat_failure", "energy_j_cm2"]
    all_rows = result[result.stratum == "all"]
    expected = holm([r[f"delta_{c}_p"] for _, r in all_rows.iterrows() for c in cols])
    actual = [r[f"delta_{c}_p_holm"] for _, r in all_rows.iterrows() for c in cols]
    assert np.allclose(actual, expected)
    assert result[result.stratum == "low"].query("controller != 'Reduced-order MPC'").delta_success_p_holm.eq(1).all()
    import re
    manuscript = (ROOT / "peerj_submission/main_manuscript.md").read_text(encoding="utf-8")
    order = list(dict.fromkeys(re.findall(r"Fig\.\s*(\d+)", manuscript)))
    assert order == ["1", "2", "3", "4", "5"], order
    print("PASS: parent-block reuse, seed allocation, Holm known case, joint 20-test family, low-band correction")

if __name__ == "__main__":
    main()
