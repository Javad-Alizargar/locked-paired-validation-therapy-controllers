"""Entry point: regenerate the PeerJ Computer Science analysis tables."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from analysis import run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "results"))
    args = ap.parse_args()
    summary = run(ROOT, Path(args.out))
    print("PeerJ analysis complete")
    print(f"  pairs per controller : {summary['n_pairs_per_controller']}")
    print(f"  clusters             : {summary['n_clusters']}")
    print(f"  bootstrap replicates : {summary['bootstrap_replicates']}")
    print(f"  validation AUC       : {summary['strata']['validation_auc_overall']:.3f}")
    print(f"  external AUC         : {summary['strata']['external_auc_overall']:.3f}")
    print()
    for r in summary["primary_all_stratum"]:
        print(f"  {r['controller']:<26} dS={r['delta_success']:+.3f} "
              f"[{r['delta_success_ci_low']:+.3f}, {r['delta_success_ci_high']:+.3f}] "
              f"p={r['delta_success_p']:.4f} MDE={r['delta_success_mde80']:.3f}")


if __name__ == "__main__":
    main()
