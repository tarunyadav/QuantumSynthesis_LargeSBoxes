from typing import List, Tuple, Dict, Optional
from collections import defaultdict, Counter
import json

# Qiskit types only needed at runtime
from qiskit import QuantumCircuit
from qiskit.circuit.library import MCXGate

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _bit_reverse8(x: int) -> int:
    x = ((x & 0xF0) >> 4) | ((x & 0x0F) << 4)
    x = ((x & 0xCC) >> 2) | ((x & 0x33) << 2)
    x = ((x & 0xAA) >> 1) | ((x & 0x55) << 1)
    return x

# ------------------------------------------------------------
# Stage A — ANF utilities (input MSB; output LSB)
# ------------------------------------------------------------

def truth_to_anf(truth_table: List[int], input_order: str = "msb", output_order: str = "lsb") -> List[List[Tuple[int, ...]]]:
    """Möbius transform to get ANF per output bit.
    input_order: 'msb' means variable 0 is MSB; 'lsb' means variable 0 is LSB.
    output_order: which output bit index we treat as bit position when slicing from bytes.
    """
    n = 8; m = 8
    anf: List[List[Tuple[int, ...]]] = [[] for _ in range(m)]

    # Möbius transform expects LSB-ordered variables; remap table if inputs are MSB-ordered
    if input_order == "msb":
        tt = [truth_table[_bit_reverse8(i)] for i in range(1 << n)]
    else:
        tt = truth_table[:]

    for out_idx in range(m):
        out_pos = (out_idx if output_order == "lsb" else (7 - out_idx))
        coeffs = [(y >> out_pos) & 1 for y in tt]
        a = coeffs[:]
        for k in range(n):
            step = 1 << k
            for i in range(1 << n):
                if i & step:
                    a[i] ^= a[i ^ step]
        poly: List[Tuple[int, ...]] = []
        for mask, c in enumerate(a):
            if c & 1:
                mon = tuple(i for i in range(n) if (mask >> i) & 1)
                poly.append(mon)
        anf[out_idx] = poly
    return anf


def eval_anf_bit(monomials: List[Tuple[int,...]], x: int, nbits: int = 8, order: str = "msb") -> int:
    """Evaluate one ANF bit on x. Empty monomial () contributes constant-1.
    """
    xs = [(x >> (nbits-1-i)) & 1 for i in range(nbits)] if order == "msb" else [(x >> i) & 1 for i in range(nbits)]
    y = 0
    for mon in monomials:
        if len(mon) == 0:
            y ^= 1
        else:
            v = 1
            for idx in mon:
                v &= xs[idx]
                if v == 0:
                    break
            y ^= v
    return y & 1


def anf_to_truth(anf: List[List[Tuple[int,...]]], nbits: int = 8, input_order: str = "msb", output_order: str = "lsb") -> List[int]:
    """Reconstruct LUT from ANF under (input_order, output_order)."""
    table = []
    for x in range(1 << nbits):
        y = 0
        for out_idx, monoms in enumerate(anf):
            b = eval_anf_bit(monoms, x, nbits=nbits, order=input_order)
            if output_order == "lsb":
                y |= (b & 1) << out_idx
            else:
                y = (y << 1) | b
        table.append(y)
    return table


def check_anf_matches_truth(truth_table: List[int], nbits: int = 8, input_order: str = "msb", output_order: str = "lsb"):
    """Stage A check with project conventions. Returns (ok, report, anf)."""
    anf = truth_to_anf(truth_table, input_order=input_order, output_order=output_order)
    rebuilt = anf_to_truth(anf, nbits=nbits, input_order=input_order, output_order=output_order)
    ok = (rebuilt == truth_table)
    report = {"ok": ok, "mismatches": []}
    if not ok:
        for x in range(1 << nbits):
            if rebuilt[x] != truth_table[x]:
                r, t = rebuilt[x], truth_table[x]
                diff_bits = [(7 - i) for i in range(8) if ((r ^ t) >> (7 - i)) & 1]
                report["mismatches"].append({
                    "x": x,
                    "truth": t,
                    "rebuilt": r,
                    "diff_bits_msb_idx": diff_bits
                })
                if len(report["mismatches"]) >= 8:
                    break
    return ok, report, anf

# ------------------------------------------------------------
# Depth estimator (AND-depth proxy)
# ------------------------------------------------------------

def _ceil_log2(k: int) -> int:
    if k <= 1: return 0
    d = 0; v = 1
    while v < k:
        v <<= 1; d += 1
    return d


def and_depth_from_anf(anf: List[List[Tuple[int, ...]]],
                       pair_temps: Optional[List[Tuple[int,int]]] = None,
                       full_temps: Optional[List[Tuple[int,...]]] = None) -> int:
    """Estimate AND-depth under chosen temporaries (quick layering sanity)."""
    pair_temps = pair_temps or []
    full_temps = full_temps or []
    sig_depth = {('x', i): 0 for i in range(8)}
    for a,b in pair_temps:
        sig_depth[('t', ('p', a,b))] = 1
    for mon in full_temps:
        d = _ceil_log2(len(mon))
        sig_depth[('t', ('m',) + tuple(sorted(mon)))] = d

    def mon_depth(mon: Tuple[int, ...]) -> int:
        vars_set = set(mon)
        factors = []
        for (a,b) in pair_temps:
            if a in vars_set and b in vars_set:
                factors.append(('t', ('p', a,b)))
                vars_set.remove(a); vars_set.remove(b)
        key_full = ('t', ('m',) + tuple(sorted(mon)))
        if key_full in sig_depth and len(vars_set) == 0 and not factors:
            factors = [key_full]
        else:
            for v in sorted(vars_set):
                factors.append(('x', v))
        max_in = max(sig_depth.get(f, 0) for f in factors) if factors else 0
        return max_in + _ceil_log2(len(factors))

    depth = 0
    for poly in anf:
        for mon in poly:
            if len(mon) >= 2:
                depth = max(depth, mon_depth(mon))
    return depth

# ------------------------------------------------------------
# Greedy CSE selection (pairs preferred)
# ------------------------------------------------------------

def build_pair_coverage(anf: List[List[Tuple[int, ...]]]) -> tuple[Dict[Tuple[int,int], set], List[Tuple[int,...]]]:
    """Map input pairs -> set of deg-3 monomials they cover (for greedy CSE scoring)."""
    pair_to_deg3: Dict[Tuple[int,int], set] = defaultdict(set)
    deg3_set = set()
    for poly in anf:
        for mon in poly:
            if len(mon) == 3:
                m = tuple(sorted(mon))
                a,b,c = m
                deg3_set.add(m)
                pair_to_deg3[(a,b)].add(m)
                pair_to_deg3[(a,c)].add(m)
                pair_to_deg3[(b,c)].add(m)
    return pair_to_deg3, sorted(deg3_set)


def select_pair_temps(pair_to_deg3: Dict[Tuple[int,int], set],
                      T_cap: Optional[int],
                      ancilla: int,
                      alpha: float=1.0,
                      beta: float=0.2):
    """Greedy CSE: pick input pairs that cover many degree-3 monomials (within ancilla/T_cap)."""
    scored = []
    for pair, covered in pair_to_deg3.items():
        cov = len(covered)
        score = alpha * cov - beta * (1 + cov)
        scored.append((score, pair, covered))
    scored.sort(reverse=True)
    cap = len(scored) if T_cap is None else min(len(scored), max(0, T_cap))
    cap = min(cap, max(0, ancilla))
    selected = []
    covered_any = set()
    for s, pair, covered in scored:
        if cap <= 0: break
        marg = len(set(covered) - covered_any)
        if s > 0 and marg > 0:
            selected.append(pair)
            covered_any |= set(covered)
            cap -= 1
    return selected, covered_any

# ------------------------------------------------------------
# Gate emission helpers
# ------------------------------------------------------------

def _emit_block_forward(qc: QuantumCircuit, block):
    for gate, wires in block:
        if gate == "x":
            qc.x(wires[0])
        elif gate == "cx":
            qc.cx(wires[0], wires[1])
        elif gate == "ccx":
            qc.ccx(wires[0], wires[1], wires[2])
        elif gate == "mcx":
            k = len(wires) - 1
            qc.append(MCXGate(k), list(wires))


def _uncompute_block(qc: QuantumCircuit, block):
    """Emit inverse of a block (used to clear temps only)."""
    for gate, wires in reversed(block):
        if gate == "x":
            qc.x(wires[0])
        elif gate == "cx":
            qc.cx(wires[0], wires[1])
        elif gate == "ccx":
            qc.ccx(wires[0], wires[1], wires[2])
        elif gate == "mcx":
            k = len(wires) - 1
            qc.append(MCXGate(k), list(wires))

# ------------------------------------------------------------
# Core ANF + CSE + layered schedule
# ------------------------------------------------------------

def synthesize_anf_with_temps(
    anf: List[List[Tuple[int, ...]]],
    measure: bool,
    T_cap: Optional[int],
    ancilla: int,
    lam: float = 0.0,
    R: int = 999999,
    G: int = 999999,
    *,
    add_barriers: bool = False,
    use_full_temps: bool = False,
    alpha: float = 1.0,
    beta: float = 0.2
):
    """
    ANF emitter with two features:
      - CSE: reuse selected pair/full monomial temporaries across outputs.
      - Layered schedule: compute temps → per-output blocks → uncompute temps.
    * Emit-inverse: we only invert the temp layer to clear ancillas; outputs persist.
    """
    n = 8; m = 8
    pair_to_deg3, _ = build_pair_coverage(anf)
    # pair_temps = []
    pair_temps, _ = select_pair_temps(pair_to_deg3, T_cap=T_cap, ancilla=ancilla, alpha=alpha, beta=beta)
    # print(f"pair temp -- {len(pair_temps)}")
    remaining = ancilla - len(pair_temps)
    full_temps: List[Tuple[int,...]] = []
    if use_full_temps and remaining > 0:
        occ = Counter()
        for poly in anf:
            for mon in poly:
                if len(mon) >= 2:
                    occ[tuple(sorted(mon))] += 1
        occ = {mon:c for mon,c in occ.items() if not (len(mon)==2 and mon in pair_temps)}
        cand = [(((c-1)*max(0,len(mon)-1)), mon, c) for mon,c in occ.items() if c>1]
        cand.sort(reverse=True)
        full_temps = [mon for _, mon, _ in cand[:remaining]]

    temps = [tuple(sorted(p)) for p in pair_temps] + [tuple(sorted(t)) for t in full_temps]
    idx_map = {mon: i for i, mon in enumerate(temps)}
    k = len(temps)

    qc = QuantumCircuit(n + k + m, m if measure else 0, name="ANF_CSE_LS")
    temp_off = n; targ_off = n + k

    # Layer 0: compute temps once (CSE across outputs)
    pre_block: List[Tuple[str, Tuple[int, ...]]] = []
    for mon, anc_idx in idx_map.items():
        ctrls = list(mon)
        tgt = temp_off + anc_idx
        if len(ctrls) == 2:
            pre_block.append(("ccx", (ctrls[0], ctrls[1], tgt)))
        else:
            pre_block.append(("mcx", tuple(ctrls + [tgt])))
    if pre_block:
        _emit_block_forward(qc, pre_block)
        if add_barriers:
            qc.barrier(label="temps")

    # Layer 1..: per-output blocks (can be merged further for deeper layering)
    for bit_msb_idx, poly in enumerate(anf):
        tgt = targ_off + bit_msb_idx
        block: List[Tuple[str, Tuple[int, ...]]] = []
        # Constant term toggles the target
        if () in poly:
            block.append(("x", (tgt,)))
        for mon in poly:
            if len(mon) == 0:
                continue  # already handled constant
            deg = len(mon)
            if deg == 1:
                block.append(("cx", (mon[0], tgt)))
            elif deg == 2:
                key = tuple(sorted(mon))
                if key in idx_map:
                    anc = temp_off + idx_map[key]
                    block.append(("cx", (anc, tgt)))
                else:
                    a,b = key
                    block.append(("ccx", (a, b, tgt)))
            elif deg == 3:
                a,b,c = tuple(sorted(mon))
                used_pair = None
                for pair in ((a,b),(a,c),(b,c)):
                    if pair in idx_map:
                        used_pair = pair; break
                if used_pair is not None:
                    anc = temp_off + idx_map[used_pair]
                    rem = ({a,b,c} - set(used_pair)).pop()
                    block.append(("ccx", (anc, rem, tgt)))
                else:
                    block.append(("mcx", (a,b,c,tgt)))
            else:
                key = tuple(sorted(mon))
                if key in idx_map:
                    anc = temp_off + idx_map[key]
                    block.append(("cx", (anc, tgt)))
                else:
                    block.append(("mcx", tuple(list(mon) + [tgt])))
        if block:
            _emit_block_forward(qc, block)
            if add_barriers:
                qc.barrier(label=f"out{bit_msb_idx}")

    # Uncompute temps only (emit inverse of pre_block), keep outputs intact
    if pre_block:
        _uncompute_block(qc, pre_block)
        if add_barriers:
            qc.barrier(label="clear_temps")

    if measure:
        qc.measure(range(targ_off, targ_off + m), range(m))

    return qc, pair_temps, full_temps

# ------------------------------------------------------------
# Small classical simulator for Stage-C
# ------------------------------------------------------------

def simulate_classical_circuit(qc, inputs, outputs, nbits=8, order="msb"):
    """Classical simulator for X/CX/CCX/MCX circuits to get LUT quickly."""
    ops=[]
    for inst, qargs, _ in qc.data:
        name=inst.name
        idx=[qc.find_bit(q).index for q in qargs]
        if name=="x": ops.append(("x",[idx[0]]))
        elif name=="cx": ops.append(("cx",[idx[0],idx[1]]))
        elif name=="ccx": ops.append(("ccx",[idx[0],idx[1],idx[2]]))
        elif name.startswith("mcx"): ops.append(("mcx", idx))

    def run(x):
        state=[0]*qc.num_qubits
        bits=[(x>>(nbits-1-i))&1 for i in range(nbits)] if order=="msb" else [(x>>i)&1 for i in range(nbits)]
        for b,q in zip(bits, inputs): state[q]=b
        for g,a in ops:
            if g=="x": state[a[0]] ^= 1
            elif g=="cx": c,t=a; state[t]^=state[c]
            elif g=="ccx": u,v,t=a; state[t]^=(state[u]&state[v])
            elif g=="mcx":
                *ctrls,t=a; v=1
                for c in ctrls:
                    v&=state[c]
                    if v==0: break
                state[t]^=v
        y=0
        for q in outputs: y=(y<<1)|(state[q]&1)
        return y
    return [run(x) for x in range(1<<nbits)]

# ------------------------------------------------------------
# Public builder
# ------------------------------------------------------------

def build_anf_cse_ls(truth: List[int],
                          T_cap: Optional[int] = None,
                          ancilla: int = 5,
                          measure: bool = False,
                          *,
                          add_barriers: bool = False,
                          use_full_temps: bool = False,
                          alpha: float = 1.0,
                          beta: float = 0.2):
    """Build ANF+CSE+LS circuit for a given S-box truth table.
    Returns (qc, AND_depth, pair_temps, full_temps).
    """
    anf = truth_to_anf(truth, input_order="lsb", output_order="lsb")
    qc, pair_temps, full_temps = synthesize_anf_with_temps(
        anf, measure=measure, T_cap=T_cap, ancilla=ancilla,
        add_barriers=add_barriers, use_full_temps=use_full_temps,
        alpha=alpha, beta=beta
    )
    and_depth = and_depth_from_anf(anf, pair_temps, full_temps)
    return qc, and_depth, pair_temps, full_temps
