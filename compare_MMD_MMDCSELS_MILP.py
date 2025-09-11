
#!/usr/bin/env python3
"""

Compares MMD, ANF+CSE+Layered, and MILP-based synthesis for 8×8 S-boxes.
- Primitive/raw circuits are verified with a fast classical simulator (X/CX/CCX/MCX).
- Transpiled ({u,cx}) circuits are optionally verified with a Statevector check (--sv_check).

Prints a summary with depth, size, qubits, ancilla, and op counts.
"""
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import argparse, json, os, sys, importlib.util, random, csv
from typing import List, Tuple, Dict



# ---------------- Qiskit ----------------
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import MCXGate
    from qiskit.quantum_info import Statevector
except Exception:
    print("ERROR: Qiskit not available. Install with: pip install qiskit")
    sys.exit(1)

# ---------------- Load local modules ----------------
def load_module(path: str, name: str):
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod

#dutta = load_module("Dutta.py", "dutta_module")
CT = load_module("make_clifford_t.py", "Clifford+T")
make_and_report = getattr(CT, "make_and_report", None)


anf_cse_ls = load_module("ANF_CSE_LS.py", "anf_cse_ls")
milp = load_module("MILP.py", "MILP")
#if not (dutta and mmd and milp):
if not (anf_cse_ls and milp):
    print("ERROR: Missing ANF_CSE_LS.py or MILP.py in current directory.")
    sys.exit(1)

if not os.path.exists("sboxes.json"):
    print("ERROR: sboxes.json not found in current directory.")
    sys.exit(1)

with open("sboxes.json") as f:
    SBOX = json.load(f)
    
# ---------------------------------Standard MMD (as implemented by Dutta et al.)--------------------------
def _bits_msb(x: int, n: int = 8):
    # MSB-first: index 0 is the most significant bit
    return [(x >> (n - 1 - i)) & 1 for i in range(n)]

def _build_tables_msb(sbox: List[int], n: int = 8):
    N = 1 << n
    TT = [_bits_msb(sbox[x], n) for x in range(N)]  # outputs
    Tab = [_bits_msb(x, n) for x in range(N)]       # inputs
    return TT, Tab

def _apply_toffoli_src(qc: QuantumCircuit, ctrls: List[int], tgt: int):
    """
    EXACT source behavior:
      - len==1 -> CX
      - len>=2 -> MCX
      - len==0 -> NO gate
    """
    if len(ctrls) == 1:
        qc.cx(ctrls[0], tgt)
    elif len(ctrls) > 1:
        qc.mcx(ctrls, tgt)
    # else: do nothing

def synthesize_exact_like_source(sbox: List[int], n: int = 8, add_measure: bool = False) -> QuantumCircuit:
    TT, Tab = _build_tables_msb(sbox, n)
    N = 1 << n
    qc = QuantumCircuit(n, n if add_measure else 0)

    # Row-0 normalization (X then flip column in TT)
    for j in range(n):
        if TT[0][j] == 1:
            qc.x(n - 1 - j)
            for x in range(N):
                TT[x][j] ^= 1

    # Main loops (identical structure & control selection)
    mult = 1
    for i in range(1, N):
        for j in range(n - 1, -1, -1):
            if TT[i][j] == Tab[i][j]:
                continue

            # primary controls = bits < j that are 1
            controls = [n - k - 1 for k in range(0, j) if TT[i][k] == 1 and k != j]
            # fallback: none OR same prefix -> all ones (exclude j)
            if (len(controls) == 0) or (TT[i - 1][0:j] == TT[i][0:j]):
                controls = [n - k - 1 for k in range(n) if TT[i][k] == 1 and k != j]

            _apply_toffoli_src(qc, controls, n - 1 - j)

            # update TT column j using the same "mult" product trick
            for l in range(N):
                for m in range(len(controls)):
                    mult = mult * TT[l][n - controls[m] - 1]
                if mult == 1:
                    TT[l][j] ^= 1
                else:
                    mult = 1  # reset only in else (as in the source)

    # Final inverse + barrier + (optional) measure
    qc = qc.inverse()
    qc.barrier()
    if add_measure:
        qc.measure(range(n), range(n))
    return qc
    
#synthesize_exact_like_source = getattr(dutta, "synthesize_exact_like_source", None)
build_anf_cse_ls = getattr(anf_cse_ls, "build_anf_cse_ls", None)
if not build_anf_cse_ls:
    print("ERROR: Required functions not found ANF_CSE_LS.py")
    sys.exit(1)


def mcx_toffoli_counts(qc,ancilla = 0,):
    from collections import defaultdict
    hist = defaultdict(int)
    counts = {"ancilla": 0, "NOT": 0, "CNOT": 0, "Toffoli": 0}
    for inst, qargs, cargs in qc.data:
        name = inst.name
        if name == "x":
            counts["NOT"] += 1
        elif name == "cx":
            counts["CNOT"] += 1
        elif name == "ccx":
            counts["Toffoli"] += 1
        elif name == "mcx":
            k = len(qargs) - 1
            hist[k] += 1
    k_max = 1
    for k in sorted(hist.keys()):
        if k >= 3:
            k_max = max(k_max,k)
            counts[f"Toffoli{k}"] = hist[k]
    counts["depth"] = qc.depth()
    if (ancilla !=0 ):
        counts["ancilla"] = ancilla
    else:
        counts["ancilla"] = 5 if k_max==7 else 4 if k_max==6 else 3 if k_max==5 else 2 if k_max==4 else 1 if k_max==3 else 0
    return counts

def mcx_hist(qc):
    from collections import defaultdict
    hist = defaultdict(int); ccx=cx=x=0
    for inst, qargs, cargs in qc.data:
        name = inst.name
        if name == "mcx":
            hist[len(qargs)-1] += 1
        elif name == "ccx":
            ccx += 1
        elif name == "cx":
            cx += 1
        elif name == "x":
            x += 1
    return hist, ccx, cx, x

def toffoli_ccx_units(hist, ccx):
    total = ccx
    for k, c in hist.items():
        if k >= 3:
            total += (2*k - 3)*c
    return total

def depth_converted(orig_depth, hist):
    def depth_w(k):
        if k in (3,4): return 3
        if k in (5,6): return 5
        if k == 7:     return 7
        if k >= 8:     return 2*k - 3
        return 1
    sub = sum(c for k,c in hist.items() if k>=3)
    add = sum(depth_w(k)*c for k,c in hist.items() if k>=3)
    return orig_depth - sub + add

# ---------------- Classical simulation & verification (X/CX/CCX/MCX) ----------------
def qc_to_gate_list(qc: QuantumCircuit) -> List[Tuple[str, List[int]]]:
    ops: List[Tuple[str, List[int]]] = []
    for inst, qargs, _ in qc.data:
        name = inst.name.lower()
        idxs = [qc.find_bit(q).index for q in qargs]
        if name in ("x", "cx", "ccx") or name.startswith("mcx"):
            if name.startswith("mcx"):
                name = "mcx"
            ops.append((name, idxs))
    return ops

def emit_gate_list_from_solution(sol: Dict, name: str = "MILP_Sbox") -> Tuple[int, List[Tuple[str, List[int]]]]:
    """
    Rebuild a gate list (no MCX) using the CCX-chain emitter logic (matches transpiled mode structure).
    Qubit layout:
      0..7: inputs (LSB at 0)
      8..15: outputs (LSB at 8)
      16: TEMP
      17..: work ancillas
    Returns: (total_qubits, gates) where gates is list of (op, [qargs]) with op in {"x","cx","ccx"}.
    """
    used = {int(b): [tuple(m) for m in lst] for b, lst in sol["used_per_bit"].items()}

    unique_monos = set()
    for b in range(8):
        unique_monos.update(used[b])
    const_m = {m for m in unique_monos if len(m) == 0}
    lin_m   = {m for m in unique_monos if len(m) == 1}
    hi_m    = [m for m in unique_monos if len(m) >= 2]

    max_deg = max((len(m) for m in hi_m), default=0)
    work_needed = max(0, max_deg - 2)

    n_inputs = 8
    n_outputs = 8
    idx_temp = n_inputs + n_outputs
    idx_work0 = idx_temp + 1
    total_qubits = n_inputs + n_outputs + 1 + work_needed

    gates: List[Tuple[str, List[int]]] = []

    def compute_monomial_into_temp(mono):
        k = len(mono)
        ctrls = list(mono)
        ops = []
        if k == 2:
            gates.append(("ccx", [ctrls[0], ctrls[1], idx_temp])); ops.append(("ccx", ctrls[0], ctrls[1], idx_temp))
            return ops
        # k >= 3
        w = idx_work0
        gates.append(("ccx", [ctrls[0], ctrls[1], w])); ops.append(("ccx", ctrls[0], ctrls[1], w))
        for i in range(2, k):
            tgt = idx_temp if (i == k-1) else (idx_work0 + (i-1))
            src = w if (i == 2) else (idx_work0 + (i-2))
            gates.append(("ccx", [src, ctrls[i], tgt])); ops.append(("ccx", src, ctrls[i], tgt))
        return ops

    def uncompute_ops(ops):
        for _, a, b, c in reversed(ops):
            gates.append(("ccx", [a, b, c]))

    # Constants
    if () in const_m:
        for b in range(8):
            if () in used[b]:
                gates.append(("x", [n_inputs + b]))
    # Linear
    for mono in sorted(lin_m):
        src = mono[0]
        for b in range(8):
            if mono in used[b]:
                gates.append(("cx", [src, n_inputs + b]))
    # Higher-degree
    for mono in sorted(hi_m, key=len):
        ops = compute_monomial_into_temp(mono)
        for b in range(8):
            if mono in used[b]:
                gates.append(("cx", [idx_temp, n_inputs + b]))
        uncompute_ops(ops)

    return total_qubits, gates


def simulate_gate_list(total_qubits: int, gates: List[Tuple[str, List[int]]], inp_byte: int) -> List[int]:
    bits = [0] * total_qubits
    # Load input byte in little-endian (x[0] as LSB)
    for i in range(8):
        bits[i] = (inp_byte >> i) & 1
    for op, q in gates:
        if op == "x":
            t = q[0]; bits[t] ^= 1
        elif op == "cx":
            c, t = q; bits[t] ^= bits[c]
        elif op == "ccx":
            a, b, t = q; bits[t] ^= (bits[a] & bits[b])
        elif op == "mcx":
            *ctrls, t = q
            v = 1
            for c in ctrls:
                v &= bits[c]
                if v == 0: break
            if v == 1: bits[t] ^= 1
        # ignore others
    return bits

def verify_gates_against_truth( gates: List[Tuple[str, List[int]]], 
                               total_qubits: int, sbox: List[int],
                            inputs: List[int], outputs: List[int],
                            output_order: str, mode: str,
                            verbose: bool = False, check_ancilla: bool = True) -> None:
    """Verify a raw/primitive QC by classical simulation. Ensures ancillas (qubits not in inputs∪outputs) re-clear to 0."""
    #gates = qc_to_gate_list(qc)
    #total_qubits = qc.num_qubits
    io = set(inputs) | set(outputs)
    ancillas = [q for q in range(total_qubits) if q not in io]
    mismatches = 0
    for x in range(256):
        bits = simulate_gate_list(total_qubits, gates, x)
        # pack outputs
        y = 0
        if output_order == "lsb":
            for i, q in enumerate(outputs):
                y |= (bits[q] & 1) << i
        else:  # msb
            for q in outputs:
                y = (y << 1) | (bits[q] & 1)
        if verbose:
            print(f"[{mode}] 0x{x:02x} -> qc=0x{y:02x}, tt=0x{sbox[x]:02x}")
        if y != sbox[x]:
            mismatches += 1
        if check_ancilla:
            for q in ancillas:
                if bits[q] != 0:
                    mismatches += 1
    assert mismatches == 0, f"{mode} verification failed: {mismatches} mismatches/ancilla-leaks"
    print(f"[{mode}] All 256 inputs match. Ancillas cleared. ✅")
    
def verify_qc_against_truth(qc: QuantumCircuit, sbox: List[int],
                            inputs: List[int], outputs: List[int],
                            output_order: str, mode: str,
                            verbose: bool = False, check_ancilla: bool = True) -> None:
    """Verify a raw/primitive QC by classical simulation. Ensures ancillas (qubits not in inputs∪outputs) re-clear to 0."""
    gates = qc_to_gate_list(qc)
    total_qubits = qc.num_qubits
    io = set(inputs) | set(outputs)
    ancillas = [q for q in range(total_qubits) if q not in io]
    mismatches = 0
    for x in range(256):
        bits = simulate_gate_list(total_qubits, gates, x)
        # pack outputs
        y = 0
        if output_order == "lsb":
            for i, q in enumerate(outputs):
                y |= (bits[q] & 1) << i
        else:  # msb
            for q in outputs:
                y = (y << 1) | (bits[q] & 1)
        if verbose:
            print(f"[{mode}] 0x{x:02x} -> qc=0x{y:02x}, tt=0x{sbox[x]:02x}")
        if y != sbox[x]:
            mismatches += 1
        if check_ancilla:
            for q in ancillas:
                if bits[q] != 0:
                    mismatches += 1
    assert mismatches == 0, f"{mode} verification failed: {mismatches} mismatches/ancilla-leaks"
    #print(f"[{mode}] All 256 inputs match. Ancillas cleared. ✅")

# ---------------- Statevector verifier for transpiled {u,cx} circuits ----------------
def verify_transpiled_by_statevector(qc: QuantumCircuit, sbox: List[int],
                                     outputs: List[int], output_order: str,
                                     mode: str, sv_samples: int = 0, verbose: bool = False) -> None:
    n = 8; total = qc.num_qubits
    indices = list(range(256))
    if sv_samples and 0 < sv_samples < 256:
        random.seed(0xC0DE)
        indices = sorted(random.sample(indices, sv_samples))
    mism = 0
    for x in indices:
        # prepare |x,0...0> (little-endian qubit indexing)
        bits = [(x >> i) & 1 for i in range(n)] + [0] * (total - n)
        label = ''.join(str(b) for b in reversed(bits))  # msb-first string
        sv = Statevector.from_label(label).evolve(qc)
        idx = abs(sv.data).argmax()
        def bit_at(i: int) -> int:
            return (idx >> i) & 1
        y = 0
        if output_order == "lsb":
            for i, q in enumerate(outputs):
                y |= bit_at(q) << i
        else:
            for q in outputs:
                y = (y << 1) | bit_at(q)
        if verbose:
            print(f"[{mode}-sv] 0x{x:02x} -> qc=0x{y:02x}, tt=0x{sbox[x]:02x}")
        if y != sbox[x]:
            mism += 1
    assert mism == 0, f"{mode} statevector verification failed: {mism} mismatches"
    print(f"[{mode}] Statevector check passed ({len(indices)} inputs). ✅")

# ---------------- Stats helpers ----------------
def stats(qc: QuantumCircuit) -> Dict[str, int]:
    # return {"depth": qc.depth(), "size": qc.size(), "ops": dict(qc.count_ops()), "qubits": qc.num_qubits}
    return {"depth": qc.depth(), "size": qc.size(), "ops": dict(qc.count_ops())}

def expand_mcx_ccx(qc, max_reps=6, mcx_mode='v-chain', ancilla_indices=None):
    """
    Repeatedly decompose the circuit to expose MCX/CCX definitions.
    Optionally replace MCX gates with an explicit MCXGate(mode=...) construction
    so Qiskit provides a consistent .definition (choose mode 'v-chain' or 'noancilla').
    """
    qc2 = qc.copy()
    # If MCX gates are present with different internals, you can re-append a consistent MCX
    # Alternatively just decompose repeatedly to unfold nested defs
    for _ in range(max_reps):
        prev = qc2.count_ops()
        # attempt to replace high-level MCX/CCX with definitions
        qc2 = qc2.decompose(reps=1)
        # try to replace any remaining composite definitions (Terra versions differ)
        try:
            qc2 = qc2.replace_all_gates_with_definitions()
        except Exception:
            pass
        if qc2.count_ops() == prev:
            break
    return qc2

def transpiled_ucx(qc: QuantumCircuit) -> QuantumCircuit:
    """Decompose then transpile to {u,cx} for realistic hardware metrics."""
    return transpile(qc.decompose(), basis_gates=["u","cx"], optimization_level=0)
    # return transpile(qc.decompose(), basis_gates=['cz','id','rx','rz','rzz', 'sx', 'x'], optimization_level=0)
    # return transpile(qc.decompose(), basis_gates=['cz','rz', 'sx'], optimization_level=0)
    # return transpile(qc.decompose(), basis_gates=['rx', 'sx', 'x'], optimization_level=0)
    # return transpile(qc.decompose(), basis_gates=['h', 's', 't', 'tdg', 'cx'])
    # return transpile(qc.decompose())

def ancilla_count(qc: QuantumCircuit, inputs: List[int], outputs: List[int], temp: int = 0) -> int:
    io = set(inputs) | set(outputs)
    return qc.num_qubits - len(io) - temp

# ---------------- MILP primitive emitter (MCX) ----------------
def synth_with_mcx_from_solution(solution: dict, name: str = "MILP_MCX") -> QuantumCircuit:
    """Emit MCX-based circuit from MILP solution using 16 wires (8 in + 8 out)."""
    qc = QuantumCircuit(16, name=name)
    x = list(range(0, 8))
    y = list(range(8, 16))
    used_per_bit = {int(b): [tuple(m) for m in monos] for b, monos in solution["used_per_bit"].items()}
    monos_per_bit = [set(used_per_bit[b]) for b in range(8)]
    # constants
    for b in range(8):
        if () in monos_per_bit[b]:
            qc.x(y[b])
    # non-constants
    for b in range(8):
        for mono in monos_per_bit[b]:
            if len(mono) == 0:
                continue
            elif len(mono) == 1:
                qc.cx(x[mono[0]], y[b])
            else:
                ctrls = [x[i] for i in mono]
                qc.append(MCXGate(len(ctrls)), ctrls + [y[b]])
    return qc

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    # ap.add_argument("--key", required=True, choices=["ZUC_S0","ZUC_S1","AES","SM4"])
    ap.add_argument("--key", required=True)
    # ANF+CSE+LS knobs
    ap.add_argument("--ancilla", type=str, default="5")   
    ap.add_argument("--T_cap", type=str, default="None")
    ap.add_argument("--add_barriers", type=int, default=0, choices=[0,1])
    ap.add_argument("--use_full_temps", type=int, default=1, choices=[0,1])
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.2)
    # MILP knobs
    ap.add_argument("--d_max", type=str, default="7")
    ap.add_argument("--A_max", type=int, default=550) # ancilla for milp
    ap.add_argument("--w_peak_anc", type=float, default=1)
    ap.add_argument("--T_cap_milp", type=int, default=None)
    ap.add_argument("--limit_monos", type=int, default=None)
    ap.add_argument("--w_mono", type=float, default=1.0)
    ap.add_argument("--w_use", type=float, default=0.2)
    ap.add_argument("--w_depth", type=float, default=12.0)
    ap.add_argument("--w_deg5plus", type=float, default=2.0)
    ap.add_argument("--time_limit", type=int, default=300)
    ap.add_argument("--no_warmstart", action="store_true")
    # Verifier
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--sv_check", action="store_true")
    ap.add_argument("--sv_samples", type=int, default=0)
    # Output files
    ap.add_argument("--csv", type=str, default=None, help="Write summary CSV to this path")
    ap.add_argument("--method", type=str, default=None, choices=["MMD","ANF+CSE+LS","MILP","ALL"])
    args = ap.parse_args()
    sbox_list = ['AES','SM4','SKINNY','ZUC_S0','ZUC_S1']
    sbox_arr = {}
    if(args.key == "ALL"):
        for sbox_name in sbox_list:
            sbox_arr = sbox_arr | {sbox_name:SBOX[sbox_name]}
    else:
        sbox_name = args.key if args.key in SBOX else args.key.replace("ZUC_","")
        # sbox = SBOX[key]
        sbox_arr = sbox_arr | {sbox_name:SBOX[sbox_name]}
    rows = []
    for key, sbox in sbox_arr.items():
        if (args.method in ["MMD","ALL",None]):
            # -------- MMD --------
            dq = synthesize_exact_like_source(sbox, add_measure=True)
            # Verify primitive (LSB packing on 8 shared wires)
            verify_qc_against_truth(dq, sbox, inputs=list(range(8)), outputs=list(range(8)), output_order="lsb",
                                    mode="MMD-primitive", verbose=args.verbose, check_ancilla=True)
            # c0 = mcx_toffoli_counts(dq)
            
            c0 = mcx_toffoli_counts(dq)
            
            h0, ccx0, cx0, x0 = mcx_hist(dq)
            depth0 = dq.depth()
               
            mmd_toff0 = toffoli_ccx_units(h0, ccx0)
            our_depth0 = depth_converted(depth0, h0)

            dq_row = {"which": key, "mode": "MMD-primitive", **stats(dq),
                      "ancilla": c0["ancilla"],'Toffoli_N': mmd_toff0 ,'Depth_N':our_depth0}
            rows.append(dq_row)
            
  
            
            
            dqt = transpiled_ucx(dq)
            # Optional statevector verification post-transpile
            if args.sv_check:
                verify_transpiled_by_statevector(dqt, sbox, outputs=list(range(8)), output_order="lsb",
                                                 mode="MMD-transpiled", sv_samples=args.sv_samples, verbose=args.verbose)

            rows.append({"which": key, "mode": "MMD-transpiled", **stats(dqt),
                          "ancilla":  c0["ancilla"]})
        
        if (args.method in ["ANF+CSE+LS","ALL",None]):
            ancilla_arr = []
            if ("_" in args.ancilla):
                start_index, end_index, jump = args.ancilla.split("_")
                ancilla_arr = [int(anc) for anc in range(int(start_index),int(end_index)+1,int(jump))]
            else:
                ancilla_arr.append(int(args.ancilla))
            for ancilla in ancilla_arr: 
                # -------- ANF + CSE + LS --------
                T_cap = None if str(args.T_cap).strip().lower() == "none" else int(args.T_cap)
                mq, and_proxy, pair_temps, full_temps = build_anf_cse_ls(
                    sbox, T_cap=T_cap, ancilla=ancilla, measure=True,
                    add_barriers=bool(args.add_barriers), use_full_temps=bool(args.use_full_temps),
                    alpha=args.alpha, beta=args.beta
                )
                # outputs are at the tail (n+k .. n+k+7)
                n = 8; k = mq.num_qubits - 16
                mmd_outputs = list(range(n + k, n + k + 8))
                verify_qc_against_truth(mq, sbox, inputs=list(range(8)), outputs=mmd_outputs, output_order="lsb",
                                        mode="ANF_CSE_LS-primitive", verbose=args.verbose, check_ancilla=True)
                
                c0 = mcx_toffoli_counts(mq,int(args.ancilla))
                # df_mmd  = pd.DataFrame({key: c0}).T.fillna(0).astype(int)
                # print(df_mmd)
                
                h0, ccx0, cx0, x0 = mcx_hist(mq)
                depth0 = mq.depth()
                   
                mmd_toff0 = toffoli_ccx_units(h0, ccx0)
                our_depth0 = depth_converted(depth0, h0)
                
                rows.append({"which": key, "mode": "ANF_CSE_LS-primitive", **stats(mq),
                             "ancilla": ancilla_count(mq, list(range(8)), mmd_outputs), 'Toffoli_N': mmd_toff0 ,'Depth_N':our_depth0,"AND_depth_proxy": and_proxy})
                if ("_" in args.ancilla):
                    print(f" {key} | Ancilla: {ancilla} Completed")
            
                
                
                
                mqt = transpiled_ucx(mq)
                # recompute outputs indices after transpile (still tail, structure preserved)
                n = 8; k = mqt.num_qubits - 16
                mmd_t_outputs = list(range(n + k, n + k + 8))
                if args.sv_check:
                    verify_transpiled_by_statevector(mqt, sbox, outputs=mmd_t_outputs, output_order="lsb",
                                                     mode="ANF_CSE_LS-transpiled", sv_samples=args.sv_samples, verbose=args.verbose)
                rows.append({"which": key, "mode": "ANF_CSE_LS-transpiled", **stats(mqt),
                             "ancilla": ancilla_count(mqt, list(range(8)), mmd_t_outputs)})
        
        if (args.method in ["MILP","ALL",None]): 
            d_max_arr = []
            if ("_" in args.d_max): 
                start_index, end_index = args.d_max.split("_")
                d_max_arr = [c for c in range(int(start_index),int(end_index)+1)]
            else:
                d_max_arr.append(int(args.d_max))
            for d_max in d_max_arr: 
                # -------- MILP --------
                sol = milp.build_and_solve_gurobi(
                    sbox, dmax=d_max, T_cap=args.T_cap_milp, limit_monos=args.limit_monos,
                    w_mono=args.w_mono, w_use=args.w_use, w_depth=args.w_depth, w_deg5plus=args.w_deg5plus,
                    time_limit=args.time_limit,warm_start_anf=not args.no_warmstart, verbose=True
                )
            
                # Primitive MCX
                qc_m_prim = synth_with_mcx_from_solution(sol, name=f"MILP_{key}_primitive")
                verify_qc_against_truth(qc_m_prim, sbox, inputs=list(range(8)), outputs=list(range(8,16)),
                                        output_order="lsb", mode="MILP-Phase1-primitive", verbose=args.verbose, check_ancilla=True)

                
                
                c0 = mcx_toffoli_counts(qc_m_prim)
                # df_mmd  = pd.DataFrame({key: c0}).T.fillna(0).astype(int)
                # print(df_mmd)
                
                h0, ccx0, cx0, x0 = mcx_hist(qc_m_prim)
                depth0 = qc_m_prim.depth()
                   
                mmd_toff0 = toffoli_ccx_units(h0, ccx0)
                our_depth0 = depth_converted(depth0, h0)
                
                
                rows.append({"which": key, "mode": "MILP-Phase1-primitive", **stats(qc_m_prim),
                             "ancilla": c0["ancilla"],'Toffoli_N': mmd_toff0 ,'Depth_N':our_depth0,})
                

                
                # Transpiled A: directly transpile the primitive MCX circuit to {u,cx}
                qctA = transpiled_ucx(qc_m_prim)
                # Outputs remain 8..15 (no extra ancillas added by basis conversion)
                if args.sv_check:
                    verify_transpiled_by_statevector(qctA, sbox, outputs=list(range(8,16)), output_order="lsb",
                                                     mode="MILP-Phase1-transpiled", sv_samples=args.sv_samples, verbose=args.verbose)
                # rows.append({"which": key, "mode": "MILP-Phase1-transpiled", **stats(qctA),
                #               "ancilla": ancilla_count(qctA, list(range(8)), list(range(8,16)))})
                rows.append({"which": key, "mode": "MILP-Phase1-transpiled", **stats(qctA),
                              "ancilla": c0["ancilla"] })
            
            
                # Transpiled (ccx-chain emitter then to {u,cx})
                tq_ccx, gates_ccx = emit_gate_list_from_solution(sol)
                verify_gates_against_truth(gates_ccx,tq_ccx, sbox, inputs=list(range(8)), outputs=list(range(8,16)),
                                        output_order="lsb", mode="MILP-Phase2-transpiled", verbose=args.verbose, check_ancilla=True)
                
               # qc_m_naive = milp.synth_circuit_from_solution(sol, name=f"MILP_{key}_naive")  
                layers, peakA, Dsched = milp.schedule_layers_ancilla_aware(
                sol,
                dmax=d_max,
                A_max=args.A_max,              # pass your CLI arg
                w_peak_anc=args.w_peak_anc,    # pass your CLI arg
                time_limit=args.time_limit,
                verbose=True
                )
                
                sol["layers"] = {tuple(k): int(v) for k,v in layers.items()}
                sol["peak_ancilla"] = int(peakA)
                sol["D_max_sched"] = int(Dsched)
                
                qc_m_naive = milp.synth_circuit_from_solution_layered(sol, A_max=sol.get("peak_ancilla"), name=f"MILP_{key}_naive")
            
                
                qct = transpiled_ucx(qc_m_naive)
                # outputs are at the tail
                n = 8; k = qct.num_qubits - 16
                milp_t_outputs = list(range(n + k, n + k + 8))
                if args.sv_check:
                    verify_transpiled_by_statevector(qct, sbox, outputs=milp_t_outputs, output_order="lsb",
                                                     mode="MILP-Phase2-transpiled", sv_samples=args.sv_samples, verbose=args.verbose)
                rows.append({"which": key, "mode": "MILP-Phase2-transpiled", **stats(qct),"Temp": 1, "d_max":  int(Dsched),
                             "ancilla": ancilla_count(qct, list(range(8)), milp_t_outputs, 1)})

    # -------- Summary --------
    if (args.method  in ["ALL",None]):
        method = "MMD | ANF_CSE_LS | MILP-Phase1 | MILP-Phase2"
    elif (args.method  in ["MILP"]):
        method = "MILP-Phase1 | MILP-Phase2"
    else:
        method = args.method
    print(f"\n=== Summary {method}===")
    # print(f"\n=== Summary Ancilla = {int(args.ancilla)} ===")
    for r in rows:
        if "primitive" in r['mode']:
                print(f"{r['which']} | {r['mode']:<16} depth={r['depth']:<6} size={r['size']:<6} Toffoli_N={r['Toffoli_N']:<6} Depth_N={r['Depth_N']:<6}"
              f"ancilla={r.get('ancilla','?')} ops={r['ops']}")
        else:
            if r['mode'] == "MILP-Phase2-transpiled":
                print(f"{r['which']} | {r['mode']:<16} depth={r['depth']:<6} size={r['size']:<6}"
              f"d_max={r['d_max']} ancilla={r.get('ancilla','?')}  temp={1}  ops={r['ops']}")
            else:
                print(f"{r['which']} | {r['mode']:<16} depth={r['depth']:<6} size={r['size']:<6}"
              f"ancilla={r.get('ancilla','?')} ops={r['ops']}")


    # Optional CSV
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["which","mode","depth","size","qubits","ancilla","ops_json"])
            for r in rows:
                w.writerow([r["which"], r["mode"], r["depth"], r["size"], r["qubits"], r.get("ancilla",""),
                            json.dumps(r["ops"], ensure_ascii=False)])

if __name__ == "__main__":
    main()
