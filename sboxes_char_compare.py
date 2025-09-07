# ===== SageMath: compute selected characteristics for all 8-bit S-boxes in a JSON =====
from sage.all import *
from sage.crypto.sbox import SBox
import json, csv
from numbers import Integral, Real

# ---------- config ----------
JSON_PATH = "sboxes.json"        # change to your file if needed
OUT_CSV   = "sbox_properties.csv"    # set to None to skip saving
SUBSET    = None                     # e.g., ["AES","SM4","SKINNY","ZUC_S0","ZUC_S1"]

# --- metrics to compute (exactly as requested) ---
METRICS = [
    "is_permutation",
    "is_involution",
    "num_fixed_points",
    "min_degree",
    "max_degree",
    "nonlinearity",
    "linearity",
    "maximal_linear_bias_absolute",
    "maximal_linear_bias_relative",
    "differential_uniformity",
    "maximal_difference_probability_abs",
    "maximal_difference_probability_rel",
    "linear_branch_number",
    "differential_branch_number",
    "boomerang_uniformity",
    "is_apn",
    "is_almost_bent",
    "is_bent",
]

def _to_py(v):
    """Coerce Sage/Python numerics to plain Python types, safely."""
    if isinstance(v, bool):
        return bool(v)
    if isinstance(v, Integral):
        return int(v)
    if isinstance(v, Real):
        return float(v)
    # fallbacks
    try:
        return int(v)
    except Exception:
        try:
            return float(v)
        except Exception:
            return v

def compute_selected_props(S):
    """Return dict of requested metrics for a list S of 256 ints."""
    sb = SBox(S)
    props = {}

    props["is_permutation"] = sb.is_permutation()
    props["is_involution"]  = sb.is_involution() if hasattr(sb, "is_involution") else None
    props["num_fixed_points"] = len(list(sb.fixed_points())) if hasattr(sb, "fixed_points") else None

    props["min_degree"] = sb.min_degree()
    props["max_degree"] = sb.max_degree()

    props["nonlinearity"] = sb.nonlinearity()
    props["linearity"]    = sb.linearity()
    props["maximal_linear_bias_absolute"] = sb.maximal_linear_bias_absolute()
    props["maximal_linear_bias_relative"] = sb.maximal_linear_bias_relative()

    props["differential_uniformity"]            = sb.differential_uniformity()
    props["maximal_difference_probability_abs"] = sb.maximal_difference_probability_absolute()
    props["maximal_difference_probability_rel"] = sb.maximal_difference_probability()

    props["linear_branch_number"]       = sb.linear_branch_number()
    props["differential_branch_number"] = sb.differential_branch_number()

    props["boomerang_uniformity"] = getattr(sb, "boomerang_uniformity", lambda: None)()
    props["is_apn"]         = getattr(sb, "is_apn", lambda: None)()
    props["is_almost_bent"] = getattr(sb, "is_almost_bent", lambda: None)()
    props["is_bent"]        = getattr(sb, "is_bent", lambda: None)()

    # Coerce to vanilla Python
    for k in props:
        if props[k] is not None:
            props[k] = _to_py(props[k])
    return props

def evaluate_sboxes(json_path=JSON_PATH, subset=SUBSET, out_csv=OUT_CSV):
    with open(json_path, "r") as f:
        boxes = json.load(f)

    if subset is not None:
        boxes = {k:v for k,v in boxes.items() if k in subset}

    rows = []
    header = ["name"] + METRICS

    for name, S in boxes.items():
        if not (isinstance(S, list) and len(S) == 256 and all(0 <= int(x) < 256 for x in S)):
            print(f"[WARN] Skipping {name}: invalid 8-bit S-box format.")
            continue
        props = compute_selected_props([int(x) for x in S])
        row = [name] + [props.get(m, None) for m in METRICS]
        rows.append(row)

    # Pretty print (compact)
    if rows:
        colw = max(4, max(len(r[0]) for r in rows))
        print("".join(["name".ljust(colw), "  "] + [m if i==0 else " | "+m for i,m in enumerate(METRICS)]))
        print("-" * (colw + 2 + sum(len(m)+3 for m in METRICS)))
        for r in rows:
            line = r[0].ljust(colw) + "  " + " | ".join(str(x) for x in r[1:])
            print(line)

    if out_csv and rows:
        import csv
        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        print(f"\nSaved: {out_csv}")

# -------- run --------
if __name__ == "__main__":
    evaluate_sboxes()