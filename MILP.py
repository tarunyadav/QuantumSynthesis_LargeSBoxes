#!/usr/bin/env python3
import argparse, json, os, sys, itertools
from collections import Counter

# ------------- Gurobi -------------
try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception as e:
    print("ERROR: gurobipy is required. Install and ensure a valid Gurobi license.")
    sys.exit(1)

# ------------- Optional: Qiskit for circuit emission -------------
try:
    from qiskit import QuantumCircuit, transpile
except Exception:
    QuantumCircuit = None

def bit(x,i): return (x>>i)&1
def all_inputs(n=8):
    for x in range(1<<n):
        yield x

def eval_monomial(mono, x):
    v = 1
    for i in mono:
        v &= bit(x,i)
        if v == 0: break
    return v

def sbox_bit_tt(sbox, b):
    return [bit(y,b) for y in sbox]

def gen_candidates(n=8, dmax=4, include_const=True):
    cands = []
    if include_const:
        cands.append(())
    for d in range(1, dmax+1):
        for combo in itertools.combinations(range(n), d):
            cands.append(combo)
    return cands

# ---------- ANF utilities ----------
def compute_anf_sets(sbox):
    """Return list[8] of sets of monomials (tuples) for each output bit ANF (LSB=0..MSB=7)."""
    def mobius(fvals):
        n = 8
        a = fvals[:]
        mask = 1
        for i in range(n):
            for x in range(1<<n):
                if x & mask:
                    a[x] ^= a[x ^ mask]
            mask <<= 1
        return a
    def monofrom(idx):
        return tuple(i for i in range(8) if (idx>>i)&1)
    res = []
    for b in range(8):
        fvals = [bit(y,b) for y in sbox]
        coeffs = mobius(fvals)
        monos = set()
        for idx,c in enumerate(coeffs):
            if c & 1:
                monos.add(monofrom(idx))
        res.append(monos)
    return res

def anf_max_degree(sbox):
    sets = compute_anf_sets(sbox)
    return max((len(m) for S in sets for m in S), default=0)

def build_and_solve_gurobi(sbox, dmax=4, T_cap=None, limit_monos=None,
                           w_mono=1.0, w_use=0.2, w_depth=12.0,
                           w_deg5plus=2.0,  # extra penalty to discourage deg>=5
                           time_limit=None, warm_start_anf=True, verbose=True):
    n = 8

    # ---------- Auto-bump dmax to avoid infeasibility ----------
    true_deg = anf_max_degree(sbox)
    if dmax < true_deg:
        if verbose:
            print(f"[info] Requested dmax={dmax} < true ANF max degree={true_deg}. Bumping dmax to {true_deg} to ensure feasibility.")
        dmax = true_deg
    if T_cap is not None and T_cap < true_deg:
        if verbose:
            print(f"[info] T_cap={T_cap} < true degree={true_deg}. Bumping T_cap to {true_deg}.")
        T_cap = true_deg

    inputs = list(all_inputs(n))
    f = {b: sbox_bit_tt(sbox,b) for b in range(n)}  # LSB..MSB

    cands = gen_candidates(n=n, dmax=dmax, include_const=True)
    if T_cap is not None:
        cands = [m for m in cands if len(m) <= T_cap]
    M = len(cands)
    deg = [len(m) for m in cands]
    cost_m = [max(d-1,0) for d in deg]

    # Precompute eval table (independent of output bit)
    VAL = [[eval_monomial(cands[m], x) for m in range(M)] for x in inputs]

    model = gp.Model("SboxFactoringGF2")
    if not verbose:
        model.Params.OutputFlag = 0
    if time_limit is not None:
        model.Params.TimeLimit = time_limit

    # Variables
    y = model.addVars(M, vtype=GRB.BINARY, name="y")             # monomial chosen globally
    x = model.addVars(8, M, vtype=GRB.BINARY, name="x")          # monomial used by output b
    k = model.addVars(8, len(inputs), vtype=GRB.INTEGER, lb=0, name="k")  # parity slack
    D_b = model.addVars(8, vtype=GRB.INTEGER, lb=0, name="D_b")
    D_max = model.addVar(vtype=GRB.INTEGER, lb=0, name="D_max")

    # Linking: x_{b,m} <= y_m
    model.addConstrs( (x[b,m] <= y[m] for b in range(8) for m in range(M)), name="link")

    # Optional limit on total selected monomials
    if limit_monos is not None:
        model.addConstr( gp.quicksum(y[m] for m in range(M)) <= limit_monos, name="limit_monos")

    # Parity constraints: sum_m x_{b,m} * VAL[t][m] - 2*k_{b,t} = f_b(t)
    for b in range(8):
        fb = f[b]
        for t, xin in enumerate(inputs):
            expr = gp.quicksum( x[b,m]*VAL[t][m] for m in range(M) ) - 2*k[b,t]
            model.addConstr( expr == fb[t], name=f"parity_b{b}_t{t}")

    # Depth proxy: D_b >= deg(m) * x_{b,m}; D_max >= D_b
    for b in range(8):
        for m in range(M):
            if deg[m] > 0:
                model.addConstr( D_b[b] >= deg[m] * x[b,m], name=f"Dlink_b{b}_m{m}")
        model.addConstr( D_max >= D_b[b], name=f"Dmax_ge_Db{b}")

    # Objective (now with a penalty for deg>=5 monomials)
    obj = w_mono * gp.quicksum( y[m]*cost_m[m] for m in range(M) ) \
        + w_use  * gp.quicksum( x[b,m] for b in range(8) for m in range(M) ) \
        + w_depth * D_max \
        + w_deg5plus * gp.quicksum( y[m] for m in range(M) if deg[m] >= 5 )
    model.setObjective(obj, GRB.MINIMIZE)

    # Warm-start with ANF if desired
    if warm_start_anf:
        anf_sets = compute_anf_sets(sbox)
        allowed = set(cands)
        for b in range(8):
            for mono in anf_sets[b]:
                if mono in allowed:
                    m_idx = cands.index(mono)
                    x[b,m_idx].Start = 1.0
                    y[m_idx].Start = 1.0

    model.optimize()

    status = model.Status
    Dmax_val = int(round(D_max.X))
    y_sol = [int(round(y[m].X)) for m in range(M)]
    x_sol = [[int(round(x[b,m].X)) for m in range(M)] for b in range(8)]

    chosen_monos = [cands[m] for m,v in enumerate(y_sol) if v==1]
    used_per_bit = {b: [cands[m] for m in range(M) if x_sol[b][m]==1] for b in range(8)}
    deg_hist = Counter(len(m) for m in chosen_monos)

    return {
        "status": status,
        "D_max": Dmax_val,
        "chosen_monomials": [list(m) for m in chosen_monos],
        "used_per_bit": {str(b): [list(m) for m in used_per_bit[b]] for b in range(8)},
        "deg_hist_selected": {int(k): int(v) for k,v in deg_hist.items()},
        "meta": {
            "dmax": dmax, "T_cap": T_cap, "limit_monos": limit_monos,
            "w_mono": w_mono, "w_use": w_use, "w_depth": w_depth, "w_deg5plus": w_deg5plus,
            "true_ANF_degree": true_deg
        }
    }

# --------- Optional: emit a naive Qiskit circuit to gauge depth ----------

def synth_circuit_from_solution(sol, name="MILP_Sbox"):
    if QuantumCircuit is None:
        return None

    used = {int(b): [tuple(m) for m in lst] for b,lst in sol["used_per_bit"].items()}

    # Gather unique monomials used across all outputs
    unique_monos = set()
    for b in range(8):
        unique_monos.update(used[b])
    # Separate by degree
    const_m = {m for m in unique_monos if len(m) == 0}
    lin_m   = {m for m in unique_monos if len(m) == 1}
    hi_m    = [m for m in unique_monos if len(m) >= 2]

    # Determine max degree to size work ancilla pool
    max_deg = max((len(m) for m in hi_m), default=0)
    work_needed = max(0, max_deg - 2)   # number of work ancillas for CCX chain
    # Layout: 0..7 inputs, 8..15 outputs, then TEMP (one), then work ancillas
    n_inputs = 8
    n_outputs = 8
    idx_temp = n_inputs + n_outputs
    idx_work0 = idx_temp + 1
    total_qubits = n_inputs + n_outputs + 1 + work_needed

    qc = QuantumCircuit(total_qubits, name=name)

    # Helper: compute monomial m (len>=2) into TEMP using linear CCX chain with work ancillas
    def compute_monomial_into_temp(mono):
        # mono is a tuple of input indices (0..7)
        k = len(mono)
        assert k >= 2
        ctrls = list(mono)
        if k == 2:
            qc.ccx(ctrls[0], ctrls[1], idx_temp)
            return [("ccx", ctrls[0], ctrls[1], idx_temp)]
        # k >= 3: use work ancillas as a chain
        ops = []
        # Stage 1: combine first two controls into work[0]
        w = idx_work0
        qc.ccx(ctrls[0], ctrls[1], w); ops.append(("ccx", ctrls[0], ctrls[1], w))
        # Chain remaining controls, ending into TEMP
        for i in range(2, k):
            tgt = idx_temp if (i == k-1) else (idx_work0 + (i-1))
            src = w if (i == 2) else (idx_work0 + (i-2))
            qc.ccx(src, ctrls[i], tgt); ops.append(("ccx", src, ctrls[i], tgt))
        return ops

    def uncompute_ops(ops):
        # reverse the recorded CCX ops
        for op in reversed(ops):
            _, a, b, c = op
            qc.ccx(a, b, c)

    # 1) Handle constant monomial: X on outputs where used
    if () in const_m:
        for b in range(8):
            if () in used[b]:
                qc.x(n_inputs + b)

    # 2) Handle linear monomials: CX(input, output) directly
    for mono in sorted(lin_m):
        src = mono[0]
        # fanout to all outputs that include this mono
        for b in range(8):
            if mono in used[b]:
                qc.cx(src, n_inputs + b)

    # 3) Handle higher-degree monomials with compute -> fanout -> uncompute
    # Process in nondecreasing degree to keep work pool safe
    for mono in sorted(hi_m, key=len):
        # compute into TEMP
        ops = compute_monomial_into_temp(mono)
        # fanout to all outputs that include this monomial
        for b in range(8):
            if mono in used[b]:
                qc.cx(idx_temp, n_inputs + b)
        # uncompute
        uncompute_ops(ops)
        # clear TEMP if it was the final target in k==2 case (already cleared by uncompute since ccx is its own inverse)

    return qc



# =============== Ancilla-aware Stage-2 scheduler (peak model, sequential reuse) ===============
def schedule_layers_ancilla_aware(sol, dmax=4, A_max=None, w_peak_anc=0.0, time_limit=None, verbose=True):
    """Given a MILP selection solution (used_per_bit), schedule monomials across L=dmax layers
    with ancilla-aware peak constraints suited to CCX-chain emitter.
    Returns: layers dict {mono_tuple: layer(1..L)}, peak_ancilla (int), D_max_sched (int).
    """
    used = {int(b): [tuple(m) for m in lst] for b,lst in sol["used_per_bit"].items()}
    # Unique monomials (exclude constants for capacity accounting)
    monos = sorted({m for b in range(8) for m in used[b]})
    M = len(monos)
    if M == 0:
        return {}, 0, 0
    r = [max(0, len(m)-2) for m in monos]  # per-monomial work ancilla for CCX-chain
    max_r = max(r) if r else 0
    L = max(1, int(dmax))

    # Quick feasibility guard for hard cap
    if A_max is not None and A_max < max_r:
        raise ValueError(f"Infeasible cap: A_max={A_max} < max_r={max_r} (AES max_r ≤ 5 typically)." )

    # Build scheduling MILP (peak per layer, not sum)
    model = gp.Model("anc_sched_peak")
    if not verbose:
        model.Params.OutputFlag = 0
    if time_limit:
        model.Params.TimeLimit = time_limit

    # Place each monomial exactly once
    yL = model.addVars(M, L, vtype=GRB.BINARY, name="yL")
    model.addConstrs((gp.quicksum(yL[m,l] for l in range(L)) == 1 for m in range(M)), name="place_once" )

    # Peak ancilla across layers (dominate per-mono demand), and optional hard cap
    use_peak = True if (A_max is not None or w_peak_anc and float(w_peak_anc) != 0.0) else False
    A = model.addVar(vtype=GRB.INTEGER, lb=0, name="A_peak") if use_peak else None
    if A_max is not None:
         model.addConstr(A <= int(A_max), name="Ancilla_HardCap")

    # -- NEW: per-layer sum-based peak constraints and optional hard cap per-layer
    for l in range(L):
        if A is not None:
            model.addConstr(A >= gp.quicksum(r[m] * yL[m,l] for m in range(M)),
                            name=f"peak_dom_sum_l{l}")
        if A_max is not None:
            model.addConstr(gp.quicksum(r[m] * yL[m,l] for m in range(M)) <= int(A_max),
                            name=f"cap_sum_l{l}")
    # use_peak = (A_max is not None) or (w_peak_anc and float(w_peak_anc) != 0.0)
    # A = model.addVar(vtype=GRB.INTEGER, lb=0, name="A_peak") if use_peak else None
    # if A_max is not None:
    #      model.addConstr(A <= int(A_max), name="Ancilla_HardCap")
    # for l in range(L):
    #     for m in range(M):
    #         if A is not None:
    #             model.addConstr(A >= r[m] * yL[m,l], name=f"peak_dom_l{l}_m{m}")
    #         if A_max is not None:
    #             model.addConstr(r[m] * yL[m,l] <= int(A_max), name=f"cap_l{l}_m{m}")

    # Depth per output via z[b,m,l]
    # Map monomial index for quick lookup
    m_index = {monos[i]: i for i in range(M)}
    uses_idx = {b: {m_index[m] for m in used[b]} for b in range(8)}
    z = model.addVars(8, M, L, vtype=GRB.BINARY, name="z")
    # for b in range(8):
    #     for m in range(M):
    #         for l in range(L):
    #             model.addConstr(z[b,m,l] <= yL[m,l], name=f"z_le_yL_b{b}_m{m}_l{l}")
    #             if m not in uses_idx[b]:
    #                 model.addConstr(z[b,m,l] == 0, name=f"z_zero_b{b}_m{m}_l{l}")
    # Link z and yL precisely:
    # - if monomial m is used by bit b, then z[b,m,l] == yL[m,l] for every layer l.
    # - otherwise z[b,m,l] == 0.
    for b in range(8):
        for m in range(M):
            if m not in uses_idx[b]:
                # not used by this bit -> force zero
                for l in range(L):
                    model.addConstr(z[b,m,l] == 0, name=f"z_zero_b{b}_m{m}_l{l}")
            else:
                # used by this bit -> exact link to placement
                for l in range(L):
                    model.addConstr(z[b,m,l] == yL[m,l], name=f"z_eq_yL_b{b}_m{m}_l{l}")

    D_b = model.addVars(8, vtype=GRB.INTEGER, lb=0, name="D_b")
    D_max = model.addVar(vtype=GRB.INTEGER, lb=0, name="D_max")
    for b in range(8):
        for m in range(M):
            for l in range(L):
                model.addConstr(D_b[b] >= (l+1) * z[b,m,l], name=f"Db_ge_layer_b{b}_m{m}_l{l}")
        model.addConstr(D_max >= D_b[b], name=f"Dmax_ge_Db_{b}")

    # Objective: minimize depth + optional weighted peak ancilla
    
    A_ub = sum(r)
    # choose w such that w*A_ub is << 1 major change in D_max or scales appropriately
    # e.g. to prioritize depth but slightly penalize A, set w = 1.0 / (A_ub * 10)
    w = float(w_peak_anc) if w_peak_anc is not None else 1.0 / (A_ub * 10)
    model.setObjective(D_max + w * A, GRB.MINIMIZE)
    # obj = D_max
    # if A is not None and w_peak_anc and float(w_peak_anc) != 0.0:
    #     obj = obj + float(w_peak_anc) * A
    # model.setObjective(obj, GRB.MINIMIZE)
    # model.addConstr(obj>=5)
    # model.Params.PoolSearchMode = 2     # systematic search
    # model.Params.PoolGap        = 1.5   # only truly optimal solutions
    # model.Params.PoolSolutions  = 100  
    model.optimize()
    # Debug print
    print("=== schedule diagnostics ===")
    total_r_sum = sum(r)
    print("total possible anc demand (sum r over all monos):", total_r_sum)
    for l in range(L):
        anc_sum = sum(r[m] * yL[m,l].X for m in range(M))
        mons = [monos[m] for m in range(M) if yL[m,l].X > 0.5]
        print(f"Layer {l+1}: anc_sum={anc_sum}, #monomials={len(mons)}")
    print("Reported A (model):", None if A is None else int(round(A.X)))
    print("Reported D_max:", int(round(D_max.X)))

    # # after optimize
    # for l in range(L):
    #     anc_sum = sum(r[m] * yL[m,l].X for m in range(M))
    #     mons_in_layer = [monos[m] for m in range(M) if yL[m,l].X > 0.5]
    #     print(f"Layer {l+1}: anc_sum={anc_sum}, monomials={mons_in_layer}")

    # model.setParam(GRB.Param.SolutionNumber, 2)
    if model.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        raise RuntimeError(f"Ancilla-aware scheduling MILP status={model.Status}")

    # Extract layers (1..L), peak, depth
    layers = {}
    for m in range(M):
        lay = 1
        for l in range(L):
            if yL[m,l].X > 0.5:
                lay = l+1
                break
        layers[monos[m]] = lay
    A_val = max_r if A is None else int(round(A.X))
    #print(A_val)
    D_val = int(round(D_max.X))
    return layers, A_val, D_val


# =============== Layer-aware emitter that uses scheduled layers (sequential ancilla reuse) ===============

def synth_circuit_from_solution_layered(sol, A_max=None, name="MILP_layered", add_barriers=True, count_temp_in_Amax=False):
    """
    Layer-aware emitter that computes each monomial ONCE per layer, fans it to all
    outputs that use it, then uncomputes. Sequential ancilla reuse inside a layer.

    Qubit layout:
      0..7   : inputs x[0..7]
      8..15  : outputs y[0..7]
      16..   : work ancilla pool of size POOL (includes TEMP if count_temp_in_Amax=True)
    """
    def tup(m): return tuple(m) if not isinstance(m, tuple) else m

    used = {int(b): [tup(m) for m in lst] for b, lst in sol["used_per_bit"].items()}
    layers_map = {tup(k): int(v) for k, v in sol.get("layers", {}).items()}
    L = max(layers_map.values()) if layers_map else 1

    monos = sorted({m for lst in used.values() for m in lst})
    # per-monomial ladder ancillas (no temp)
    r_no_temp = {m: max(0, len(m) - 2) for m in monos if len(m) >= 1}
    max_ladder = max(r_no_temp.values(), default=0)

    # TEMP accounting
    temp_cost = 1 if count_temp_in_Amax else 0
    r_with_temp = {m: (max(0, len(m) - 2) + temp_cost) for m in monos if len(m) >= 1}
    max_r = max(r_with_temp.values(), default=0)

    # choose pool
    if A_max is None:
        A_pool = max_r
    else:
        A_pool = int(A_max)
    if A_pool < max_r:
        raise ValueError(f"A_max/pool={A_pool} < required max_r={max_r} (incl. TEMP={temp_cost}).")

    # allocate qubits
    # if TEMP is counted in pool, it will be anc[0]; else we tack a dedicated TEMP after the pool
    base = 16
    pool = A_pool
    total_qubits = base + pool + (0 if count_temp_in_Amax else 1)
    qc = QuantumCircuit(total_qubits, name=name)
    x = list(range(0, 8))
    y = list(range(8, 16))
    anc = list(range(base, base + pool))
    TEMP = anc[0] if count_temp_in_Amax else base + pool  # one temp bit

    # helpers
    def compute_ladder(qc, ctrls, anc_chain, out):
        """Compute AND(ctrls) into 'out' using anc_chain as ladder (no fanout yet)."""
        d = len(ctrls)
        if d == 0:
            qc.x(out)
            return []
        if d == 1:
            qc.cx(ctrls[0], out)
            return []
        if d == 2:
            qc.ccx(ctrls[0], ctrls[1], out)
            return []
        # d >= 3: ladder into anc_chain then into out
        r = d - 2
        assert len(anc_chain) >= r, f"need {r} ancillas, got {len(anc_chain)}"
        qc.ccx(ctrls[0], ctrls[1], anc_chain[0])
        for i in range(2, d-1):
            qc.ccx(anc_chain[i-2], ctrls[i], anc_chain[i-1])
        qc.ccx(anc_chain[r-1], ctrls[d-1], out)
        # return order to uncompute anc_chain later
        return list(range(r-1, -1, -1))

    def uncompute_ladder(qc, ctrls, anc_chain):
        d = len(ctrls)
        if d <= 2:
            return
        # reverse of compute (excluding final out toggle)
        for i in range(d-2, 1, -1):
            qc.ccx(anc_chain[i-2], ctrls[i], anc_chain[i-1])
        qc.ccx(ctrls[0], ctrls[1], anc_chain[0])

    # constants
    for b in range(8):
        if () in used[b]:
            qc.x(y[b])

    # group monomials by layer
    layer_to_m = {ell: [] for ell in range(1, L+1)}
    for m in monos:
        if len(m) == 0:  # already handled
            continue
        ell = layers_map.get(m, 1)
        layer_to_m[ell].append(m)

    # build layers
    for ell in range(1, L+1):
        if add_barriers and ell > 1:
            qc.barrier()

        # For each monomial in this layer:
        for m in layer_to_m[ell]:
            ctrls = [x[i] for i in m]
            # compute AND(m) into TEMP using ladder ancillas
            ladder_chain = anc[1:1 + max(0, len(ctrls)-2)] if count_temp_in_Amax else anc[:max(0, len(ctrls)-2)]
            # ensure TEMP is 0 at entry; we assume anc are cleared between monomials (sequential reuse)
            # compute
            compute_ladder(qc, ctrls, ladder_chain, TEMP)
            # fan-out to every output that uses m
            for b in range(8):
                if m in used[b]:
                    qc.cx(TEMP, y[b])
            # uncompute ladder and TEMP
            uncompute_ladder(qc, ctrls, ladder_chain)
            # TEMP is back to 0 here

    if add_barriers:
        qc.barrier()
    return qc
def synth_circuit_from_solution_layered_old(sol, A_max=None, name="MILP_layered", add_barriers=True):
    """Emit a circuit using solution["layers"] (1..L) with a per-layer ancilla pool of size A_max
    (or solution["peak_ancilla"] if A_max is None). Sequential reuse within a layer; no sum capacity.
    Returns a QuantumCircuit.
    """
    if QuantumCircuit is None:
        return None

    def ensure_tuple(m):
        return tuple(m) if not isinstance(m, tuple) else m

    def toggle_with_chain(qc, controls, target, anc):
        d = len(controls)
        if d == 0:
            qc.x(target)
        elif d == 1:
            qc.cx(controls[0], target)
        elif d == 2:
            qc.ccx(controls[0], controls[1], target)
        else:
            r = d - 2
            assert len(anc) >= r, f"Need {r} ancillas, got {len(anc)}"
            qc.ccx(controls[0], controls[1], anc[0])
            for i in range(2, d-1):
                qc.ccx(anc[i-2], controls[i], anc[i-1])
            qc.ccx(anc[r-1], controls[d-1], target)
            for i in range(d-2, 1, -1):
                qc.ccx(anc[i-2], controls[i], anc[i-1])
            qc.ccx(controls[0], controls[1], anc[0])

    used = {int(b): [ensure_tuple(m) for m in lst] for b,lst in sol["used_per_bit"].items()}
    layers_map = {ensure_tuple(k): int(v) for k,v in sol.get("layers", {}).items()}
    L = max(layers_map.values()) if layers_map else 1

    # unique monomials
    monos = sorted({m for lst in used.values() for m in lst})
    r_m = {m: max(0, len(m)-2) for m in monos if len(m) >= 1}
    max_r = max(r_m.values()) if r_m else 0

    pool = int(sol.get("peak_ancilla", max_r)) if A_max is None else int(A_max)
    if pool < max_r:
        raise ValueError(f"A_max/pool={pool} < max_r={max_r}; increase ancilla pool or change layer schedule.")

    total_qubits = 16 + pool
    qc = QuantumCircuit(total_qubits, name=name)
    x = list(range(0,8)); y = list(range(8,16)); anc = list(range(16,16+pool))

    # constants
    for b in range(8):
        if () in used[b]:
            qc.x(y[b])

    # group monomials by layer
    layer_to_m = {ell: [] for ell in range(1, L+1)}
    for m in monos:
        if len(m) == 0: 
            continue
        ell = layers_map.get(m, 1)
        layer_to_m[ell].append(m)

    # apply layer by layer
    for ell in range(1, L+1):
        if add_barriers and ell > 1:
            qc.barrier()
        for b in range(8):
            tgt = y[b]
            for m in layer_to_m[ell]:
                if m in used[b]:
                    ctrls = [x[i] for i in m]
                    if len(ctrls) <= 2:
                        toggle_with_chain(qc, ctrls, tgt, [])
                    else:
                        toggle_with_chain(qc, ctrls, tgt, anc[:len(ctrls)-2])
    if add_barriers:
        qc.barrier()
    return qc
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sbox", required=True, help="Path to JSON with {'KEY':[256 ints]}")
    ap.add_argument("--key", required=True, help="Key inside JSON (e.g., S0, S1, AES, SM4, SKINNY_S8)")
    ap.add_argument("--dmax", type=int, default=4)
    ap.add_argument("--T_cap", type=int, default=None)
    ap.add_argument("--limit_monos", type=int, default=None)
    ap.add_argument("--w_mono", type=float, default=1.0)
    ap.add_argument("--w_use", type=float, default=0.2)
    ap.add_argument("--w_depth", type=float, default=12.0)
    ap.add_argument("--w_deg5plus", type=float, default=2.0)
    ap.add_argument("--time_limit", type=int, default=300)
    ap.add_argument("--no_warmstart", action="store_true")
    ap.add_argument("--emit_qiskit", type=int, default=1, choices=[0,1])
    args = ap.parse_args()

    # Load from JSON
    try:
        data = json.load(open(args.sbox, "r"))
    except Exception as e:
        print(f"ERROR: failed to read JSON from {args.sbox}: {e}")
        sys.exit(1)

    if args.key not in data:
        print(f"ERROR: key '{args.key}' not found in {args.sbox}. Keys: {list(data.keys())}")
        sys.exit(1)

    sbox = data[args.key]
    if not (isinstance(sbox, list) and len(sbox) == 256 and all(isinstance(v, int) and 0 <= v <= 255 for v in sbox)):
        print("ERROR: S-box must be a list of 256 integers in [0,255].")
        sys.exit(1)

    sol = build_and_solve_gurobi(
        sbox, dmax=args.dmax, T_cap=args.T_cap, limit_monos=args.limit_monos,
        w_mono=args.w_mono, w_use=args.w_use, w_depth=args.w_depth, w_deg5plus=args.w_deg5plus,
        time_limit=args.time_limit, warm_start_anf=not args.no_warmstart, verbose=True
    )

    print("Status:", sol["status"])
    print("True ANF max degree:", sol["meta"]["true_ANF_degree"])
    print("D_max (degree proxy used in solve):", sol["D_max"])
    print("Selected monomials by degree:", sol["deg_hist_selected"])

    out_path = f"milp_{args.key}_solution.json"
    with open(out_path, "w") as f:
        json.dump(sol, f, indent=2)
    print("Wrote", out_path)

    
    if args.emit_qiskit and QuantumCircuit is not None:
        qc = synth_circuit_from_solution(sol, name=f"MILP_{args.key}")
        if qc is not None:
            qcd = qc.decompose()
            qct = transpile(qcd, basis_gates=["u","cx"], optimization_level=0)
            print(f"Naive emitted circuit (after decompose/transpile) depth for {args.key}:", qct.depth())
            try:
                qasm_str = qct.qasm(formatted=True)
            except Exception:
                try:
                    from qiskit.qasm3 import dumps as qasm3_dumps
                    qasm_str = qasm3_dumps(qct)
                except Exception:
                    qasm_str = None
            if qasm_str is not None:
                with open(f"milp_{args.key}.qasm", "w") as f:
                    f.write(qasm_str)
                print(f"Wrote milp_{args.key}.qasm")
            else:
                print("Warning: could not serialize circuit to QASM (version mismatch).")

if __name__ == "__main__":
    main()
