"""Run every experiment.   python run_all.py              (full, ~1 h on a laptop, single-threaded)
                           python run_all.py --quick      (smaller sizes / fewer seeds, ~10 min)
                           python run_all.py --only 1 3   (selected experiments)"""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", nargs="*", type=int, default=None, help="experiment numbers, e.g. 1 3")
    a = ap.parse_args()
    import exp1_quality, exp2_scalability, exp3_ablation, exp4_sensitivity, exp5_bel_adaptivity
    mods = {1: exp1_quality, 2: exp2_scalability, 3: exp3_ablation, 4: exp4_sensitivity,
            5: exp5_bel_adaptivity}
    for k, mod in mods.items():
        if a.only and k not in a.only:
            continue
        t = time.time(); print(f"\n===== {mod.__name__} =====", flush=True)
        mod.run(quick=a.quick); print(f"[{mod.__name__} done in {time.time()-t:.1f}s]", flush=True)
    print("\nCSV -> results/   tables -> results/tables/   plots -> results/plots/")


if __name__ == "__main__":
    main()
