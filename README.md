# QuantumSynthesis_LargeSBoxes

## Execution

### MMD

`python compare_MMD_MMDCSELS_MILP.py --key ALL --method MMD`

### ANF+CSE+LS

`python compare_MMD_MMDCSELS_MILP.py --key ALL --ancilla 0_250_5 --method ANF+CSE+LS`
`python compare_MMD_MMDCSELS_MILP.py --key AES --ancilla 250 --method ANF+CSE+LS`

### MILP

`python compare_MMD_MMDCSELS_MILP.py --key AES  --d_max 25 --A_max 550 --w_peak_anc 1 --method MILP`
`python compare_MMD_MMDCSELS_MILP.py --key AES  --d_max 1_5 --method MILP`
`python compare_MMD_MMDCSELS_MILP.py --key AES  --d_max 25 --method MILP` (A_max default is 550)

### S-boxes Characteristics Comparison using SAGE

`python sboxes_char_compare.py`
