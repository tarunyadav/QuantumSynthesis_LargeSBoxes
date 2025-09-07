
# analyze_sbox_stats.py
from collections import defaultdict
import csv
from typing import Dict, List, Tuple, Any

def analyze_qc_primitives(qc, pair_temps: List[Tuple[int,int]], full_temps: List[Tuple[int,...]],
                          n_inputs: int = 8, n_outputs: int = 8,
                          mcx_to_ccx_conv: Dict[int, float] = None,
                          ccx_tdepth: float = 1.0,
                          mcx_tdepth_default: float = 3.0) -> Dict[str, Any]:
    """
    Analyze a QuantumCircuit built by build_anf_cse_ls.

    Returns a dictionary with:
      - raw counts: X, CX, CCX, MCX_by_deg
      - temps info: num_pair_temps, num_full_temps, total_temps
      - primitive_and_depth: (user should pass and_depth separately if needed)
      - converted costs: effective_ccx_equiv, estimated_tdepth
    Parameters:
      - mcx_to_ccx_conv: dict mapping k_controls -> effective CCX-equivalent (float).
          e.g. {3: 3.0, 4: 5.0, 5: 7.0, 6: 9.0, 7: 11.0}
          If None, uses a conservative default (see below).
      - ccx_tdepth: estimated T-depth per CCX (used to get T-depth estimate).
      - mcx_tdepth_default: fallback T-depth per MCX if not mapped (multiplies conv).
    """
    if mcx_to_ccx_conv is None:
        # conservative / configurable defaults (user-editable)
        mcx_to_ccx_conv = {2: 1.0, 3: 3.0, 4: 5.0, 5: 7.0, 6: 9.0, 7: 11.0, 8: 13.0}

    counts = {"x":0, "cx":0, "ccx":0, "mcx_by_deg": defaultdict(int), "mcx_total":0, "cnot":0}
    # parse qc.data for gate names and wire indices
    for inst, qargs, _ in qc.data:
        name = inst.name
        # MCX gates may have a name like 'mcx' or 'mcx_gray' depending on Qiskit; handle generically
        if name == "x":
            counts["x"] += 1
        elif name == "cx":
            counts["cx"] += 1
            counts["cnot"] += 1
        elif name == "ccx":
            counts["ccx"] += 1
        elif name.startswith("mcx"):
            # inst.num_ctrl_qubits or len(qargs)-1 gives control count in many Qiskit MCXGate variants
            try:
                k = inst.num_ctrl_qubits
            except Exception:
                k = len(qargs) - 1
            counts["mcx_by_deg"][k] += 1
            counts["mcx_total"] += 1
        else:
            # fallback: handle gates appended with MCXGate(k)
            if hasattr(inst, 'num_ctrl_qubits'):
                k = getattr(inst, 'num_ctrl_qubits')
                if k is not None and k >= 2:
                    counts["mcx_by_deg"][k] += 1
                    counts["mcx_total"] += 1

    # temps info (from caller)
    num_pair = len(pair_temps) if pair_temps is not None else 0
    num_full = len(full_temps) if full_temps is not None else 0
    total_temps = num_pair + num_full

    # convert MCX -> CCX-equivalent
    effective_ccx = counts["ccx"]
    for deg, c in counts["mcx_by_deg"].items():
        conv = mcx_to_ccx_conv.get(deg, (2*deg - 3))  # fallback heuristic if not provided
        effective_ccx += conv * c

    # crude T-depth estimate (user-specified per-primitive)
    est_tdepth = effective_ccx * ccx_tdepth
    # if user wants to separately account MCX tdepth multipliers:
    # We add fallback per-mcx contribution if they want:
    for deg,c in counts["mcx_by_deg"].items():
        est_tdepth += c * mcx_tdepth_default

    out = {
        "raw_counts": counts,
        "temps": {"pair_temps": num_pair, "full_temps": num_full, "total_temps": total_temps},
        "effective_ccx_equiv": effective_ccx,
        #"estimated_tdepth": est_tdepth
    }
    return out

def print_summary_row(name: str, stats: Dict[str, Any]):
    rc = stats["raw_counts"]
    print(f"=== {name} ===")
    print(f"X: {rc['x']}, CX: {rc['cx']}, CCX: {rc['ccx']}, MCX_total: {rc['mcx_total']}")
    print("MCX by control-degree:", dict(rc["mcx_by_deg"]))
    print("Temps: pair={}, full={}, total={}".format(
        stats["temps"]["pair_temps"], stats["temps"]["full_temps"], stats["temps"]["total_temps"]))
    print("Effective CCX-equivalent: {:.2f}".format(stats["effective_ccx_equiv"]))
    #print("Estimated T-depth (approx): {:.2f} (ccx_tdepth multiplier used)".format(stats["estimated_tdepth"]))
    print()

def write_rows_csv(path: str, rows: List[Dict[str,Any]]):
    # rows are dicts; flatten minimally for CSV
    if not rows:
        return
    header = ["name", "x", "cx", "ccx", "mcx_total", "mcx_by_deg", "pair_temps", "full_temps", "total_temps",
              "effective_ccx_equiv", "estimated_tdepth"]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in rows:
            rc = r["raw_counts"]
            writer.writerow([
                r.get("name",""),
                rc["x"], rc["cx"], rc["ccx"], rc["mcx_total"],
                dict(rc["mcx_by_deg"]),
                r["temps"]["pair_temps"], r["temps"]["full_temps"], r["temps"]["total_temps"],
                "{:.3f}".format(r["effective_ccx_equiv"]),
                #"{:.3f}".format(r["estimated_tdepth"])
            ])
