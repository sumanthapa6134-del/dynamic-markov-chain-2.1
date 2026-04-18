"""
Markov Chain Q-System Predictor
================================
Forward Markov chain prediction model for Q-system rock mass classification.
Reads TPMs from <folder>/dynamictpm/transition_probability_matrices.xlsx and
predicts Q-values at j=1..20 future chainage steps.

Usage:
    python markov_predictor.py --folder "<data_folder>"
    python markov_predictor.py --folder "<data_folder>" --demo
    python markov_predictor.py --folder "<data_folder>" --inputs '{"RQD":[45],"Jn":[6],"Jr":[1.5],"Ja":[2],"Jw":[1.0],"SRF":[2.5]}'
"""

import argparse
import json
import os
import re
import sys
import numpy as np
import pandas as pd

# ── Parameter definitions ──────────────────────────────────────────────────────
#
# RATINGS: representative value for each state, ordered State 1 -> N.
#   RQD  : range midpoints  (0-25 -> 12.5,  26-50 -> 38,  51-75 -> 63,  76-100 -> 88)
#   Jr   : ordered best->worst (State 1 = Discontinuous = 4)
#   SRF  : non-monotonic    (fault zones | medium stress | rock burst)
#
# State tables:
#   RQD (4)  : Very poor / Poor / Fair / Good
#   Jn  (9)  : None(0.5) One(2) OnePlus(3) Two(4) TwoPlus(6) Three(9) ThreePlus(12) FourPlus(15) Earth(20)
#   Jr  (7)  : Discontinuous(4) UndulRough(3) UndulSmooth(2.5) UndulSlick(2) PlanarRough(1.5) PlanarSmooth(1) PlanarSlick(0.5)
#   Ja  (10) : Healed(0.75) Unaltered(1) SlightlyAltered(2) CoatedNonSoft(3) CoatedSoft(4)
#              ThinNonSoftClay(6) ThinSoftClay(8) ThinSwellingClay(12) ThickClayMed(13) ThickSwellingClay(20)
#   Jw  (6)  : Dry(1.0) Wet(0.66) HighPressUnfilled(0.5) HighPressFilled(0.33) ExcInflowsDecay(0.1) ExcInflowsNoDecay(0.05)
#   SRF (9)  : MultClayZones(10) MultNonClay(7.5) SingleWeak<50m(5) SingleWeak>50m(2.5) MedStress(1)
#              HighStressTight(2) ModSlabbing(50) SlabbingBurst(200) HeavyBurst(400)

RATINGS = {
    'RQD': [12.5, 38.0, 63.0, 88.0],
    'Jn':  [0.5, 2, 3, 4, 6, 9, 12, 15, 20],
    'Jr':  [4, 3, 2.5, 2, 1.5, 1, 0.5],
    'Ja':  [0.75, 1, 2, 3, 4, 6, 8, 12, 13, 20],
    'Jw':  [1.0, 0.66, 0.5, 0.33, 0.1, 0.05],
    'SRF': [10, 7.5, 5, 2.5, 1, 2, 50, 200, 400],
}

RQD_RANGES = [
    (0,   25,  0),  # State 1 - Very poor
    (26,  50,  1),  # State 2 - Poor
    (51,  75,  2),  # State 3 - Fair
    (76, 100,  3),  # State 4 - Good
]

NUM_STATES = {param: len(vals) for param, vals in RATINGS.items()}
PARAMS     = ['RQD', 'Jn', 'Jr', 'Ja', 'Jw', 'SRF']

STATE_LABELS = {
    # Table 1: RQD
    'RQD': [
        'Very poor (0-25)',
        'Poor (26-50)',
        'Fair (51-75)',
        'Good (76-100)',
    ],
    # Table 2: Jn
    'Jn': [
        'None (0.5)',
        'One (2)',
        'One plus (3)',
        'Two (4)',
        'Two plus (6)',
        'Three (9)',
        'Three plus (12)',
        'Four or more (15)',
        'Earth (20)',
    ],
    # Table 3: Jr
    'Jr': [
        'Discontinuous (4)',
        'Undulating - Rough (3)',
        'Undulating - Smooth (2.5)',
        'Undulating - Slickensided (2)',
        'Planar - Rough (1.5)',
        'Planar - Smooth (1)',
        'Planar - Slickensided (0.5)',
    ],
    # Table 4: Ja
    'Ja': [
        'No fills - Healed (0.75)',
        'No fills - Unaltered joint wall (1)',
        'No fills - Slightly altered wall (2)',
        'No fills - Coated non-softening (3)',
        'No fills - Coated softening or disintegrated sandy particles (4)',
        'Thin fills - Thin non-softening clay fillings (6)',
        'Thin fills - Thin softening clay fillings (8)',
        'Thin fills - Thin swelling clays (12)',
        'Thick fills - Clay band; medium to low over consolidated (13)',
        'Thick fills - Clay band; swelling clay (20)',
    ],
    # Table 5: Jw
    'Jw': [
        'Dry (1.0)',
        'Wet (0.66)',
        'High pressure in unfilled joints (0.5)',
        'High pressure with fillings outwash (0.33)',
        'Exc. inflows with decay (0.1)',
        'Exc. inflows without decay (0.05)',
    ],
    # Table 6: SRF
    'SRF': [
        'Multiple clay zones (10)',
        'Multiple non-clay zones (7.5)',
        'Single weak zone (Depth<50m) or heavily jointed (5)',
        'Single weak zone (Depth>50m) or low stress >200 (2.5)',
        'Medium stress, sigma_c/sigma_1 = 200-10 (1)',
        'High stress with tight structure, sigma_c/sigma_1 = 10-5 (2)',
        'Moderate slabbing, sigma_c/sigma_1 = 5-3 (50)',
        'Slabbing and rock burst, sigma_c/sigma_1 = 3-2 (200)',
        'Heavy rock burst, sigma_c/sigma_1 < 2 (400)',
    ],
}


# ── Step 2: Snap to nearest defined rating ────────────────────────────────────

def snap(value, param):
    """
    Map a field-measured value to its defined state representative.
    RQD: range-based (0-25->12.5, 26-50->38.0, 51-75->63.0, 76-100->88.0).
    All others: nearest value in RATINGS by absolute difference.
    """
    if param == 'RQD':
        for lo, hi, idx in RQD_RANGES:
            if lo <= value <= hi:
                return RATINGS['RQD'][idx]
        return RATINGS['RQD'][0] if value < 0 else RATINGS['RQD'][-1]
    return min(RATINGS[param], key=lambda r: abs(r - value))


def preprocess(inputs):
    """Snap all input values to their defined state representatives. Print mapping notes."""
    snapped = {}
    for param, vals in inputs.items():
        snapped_vals = []
        for v in vals:
            s = snap(v, param)
            if param == 'RQD':
                state_no = RATINGS['RQD'].index(s) + 1
                label    = STATE_LABELS['RQD'][state_no - 1]
                print(f"  RQD: {v} -> State {state_no} ({label}), representative value = {s}")
            elif abs(s - v) > 1e-9:
                print(f"  {param}: {v} -> snapped to nearest defined rating {s}")
            snapped_vals.append(s)
        snapped[param] = snapped_vals
    return snapped


# ── Step 3: Convert ratings to 1-indexed states ───────────────────────────────

def rating_to_state(value, param):
    return RATINGS[param].index(value) + 1


def to_states(snapped):
    return {param: [rating_to_state(v, param) for v in vals]
            for param, vals in snapped.items()}


# ── Step 4: Build row probability vector ─────────────────────────────────────

def make_row_vector(state_list, num_states):
    """
    1xN probability vector. Weight = fraction of inputs that landed in each state.
    E.g. two inputs both hitting state 3 -> [0, 0, 1.0, 0, ...]
         one input in state 1, one in state 3 -> [0.5, 0, 0.5, 0, ...]
    """
    if not state_list:
        return np.zeros(num_states)
    vector = np.zeros(num_states)
    for s in state_list:
        vector[s - 1] += 1
    return vector / len(state_list)


# ── Step 5: Load TPMs and multiply ───────────────────────────────────────────

def load_tpms(tpm_file):
    """
    Load all 6-parameter TPMs from transition_probability_matrices.xlsx.
    Each sheet has 20 N×N matrices stacked vertically, each preceded by
    a label row 'TPM j=k' and followed by a blank row.

    Returns: tpms[param][j] = np.ndarray (N×N), j is 1-indexed (1..20).
    """
    if not os.path.exists(tpm_file):
        print(f"\nERROR: TPM file not found:\n  {tpm_file}")
        print("\nRun the 'tpm-rock-mass' skill first to generate")
        print("'transition_probability_matrices.xlsx' in the dynamictpm/ folder,")
        print("then re-run this script.")
        sys.exit(1)

    tpms = {}
    xl   = pd.ExcelFile(tpm_file)

    for param in PARAMS:
        if param not in xl.sheet_names:
            print(f"  WARNING: Sheet '{param}' not found in TPM file -- skipping.")
            continue

        df = xl.parse(param, header=None)
        n  = NUM_STATES[param]
        tpms[param] = {}

        i = 0
        while i < len(df):
            cell = str(df.iloc[i, 0]).strip()
            # Label format in Excel: "TPM  j = 1"  (spaces around = are variable)
            m = re.search(r'TPM.*j\s*=\s*(\d+)', cell, re.IGNORECASE)
            if m:
                j = int(m.group(1))
                # Row i+1 is the column-header row ("State \ To  1  2 ...") -- skip it
                matrix_start = i + 2
                matrix_end   = matrix_start + n
                block     = df.iloc[matrix_start:matrix_end, :].copy()
                # Column 0 = state row-labels (1,2,...); values begin at column 1
                data_cols = list(block.columns[1:n + 1])
                mat = block[data_cols].values.astype(float)
                if mat.shape == (n, n):
                    tpms[param][j] = mat
                else:
                    print(f"  WARNING: {param} TPM j={j} -- shape {mat.shape} "
                          f"(expected ({n},{n})); skipping block.")
                i = matrix_end
            else:
                i += 1

        found = sorted(tpms[param].keys())
        print(f"  {param}: loaded {len(found)} TPMs "
              f"(j={found[0]}..{found[-1]})" if found else f"  {param}: NO TPMs loaded!")

    return tpms


def compute_predictions(row_vectors, tpms):
    """
    6 parameters x 20 steps = 120 multiplications.
    predicted_vector[param][j] = row_vector[param] @ TPM[param][j]
    """
    predictions = {}
    for param in PARAMS:
        if param not in tpms:
            continue
        predictions[param] = {}
        rv = row_vectors[param]
        for j in range(1, 31):
            if j not in tpms[param]:
                continue
            result = rv @ tpms[param][j]
            if np.sum(result) < 1e-9:
                print(f"  WARNING: {param} j={j} -- all-zero prediction vector "
                      "(no historical transitions from this starting state).")
            predictions[param][j] = result
    return predictions


# ── Step 6: Interpret -- probability-weighted average rating ──────────────────

def predicted_rating_value(prob_vector, param):
    """
    Expected value: sum(prob[i] * rating[i]).
    Reflects genuine uncertainty -- if states 3 and 4 are equally probable,
    the prediction falls between them rather than arbitrarily picking one.
    """
    ratings = RATINGS[param]
    return float(sum(prob_vector[i] * ratings[i] for i in range(len(ratings))))


def most_probable_rating(prob_vector, param):
    """
    Rating of the most probable state (argmax of the probability vector).
    Used for Jn, Jr, Ja, Jw, SRF.
    Returns (rating_value, state_number_1indexed, state_label).
    """
    idx   = int(np.argmax(prob_vector))
    return RATINGS[param][idx], idx + 1, STATE_LABELS[param][idx]


def interpret(predictions, snapped_inputs):
    """
    Convert predicted probability vectors to scalar ratings for each param at each j.

    - RQD  : probability-weighted average rating (reflects gradual Q-value shift)
    - Others: rating of the MAXIMUM PROBABLE STATE (argmax) per the defined rating tables
    - Fallback: if the probability vector is all-zero (no historical transitions from
      the starting state), use the mean of the user's snapped input values so the
      Q calculation remains meaningful.
    """
    pred_ratings = {}
    for param, steps in predictions.items():
        pred_ratings[param] = {}

        # Fallback = mean of the user's own input representative values for this param
        fallback_vals = snapped_inputs.get(param, [])
        fallback = float(sum(fallback_vals) / len(fallback_vals)) if fallback_vals else float('nan')

        for j, pvec in steps.items():
            if np.sum(pvec) < 1e-9:
                # All-zero vector: no transitions observed -- use input data as fallback
                val = fallback
                print(f"  NOTE: {param} j={j} -- zero prediction vector; "
                      f"using input fallback value {fallback}")
            elif param == 'RQD':
                val = predicted_rating_value(pvec, param)
            else:
                val, _, _ = most_probable_rating(pvec, param)

            pred_ratings[param][j] = val
    return pred_ratings


# ── Step 7: Compute Q values ─────────────────────────────────────────────────

def compute_q(pred_ratings, j):
    """Q(j) = (RQD/Jn) x (Jr/Ja) x (Jw/SRF)"""
    try:
        rqd = pred_ratings['RQD'][j]
        jn  = pred_ratings['Jn'][j]
        jr  = pred_ratings['Jr'][j]
        ja  = pred_ratings['Ja'][j]
        jw  = pred_ratings['Jw'][j]
        srf = pred_ratings['SRF'][j]
        if jn == 0 or ja == 0 or srf == 0:
            return float('nan')
        return (rqd / jn) * (jr / ja) * (jw / srf)
    except KeyError:
        return float('nan')


def classify_q(q):
    """Barton Q-system rock mass quality class."""
    if   q < 0.01:  return "Exceptionally poor"
    elif q < 0.1:   return "Extremely poor"
    elif q < 1:     return "Very poor"
    elif q < 4:     return "Poor"
    elif q < 10:    return "Fair"
    elif q < 40:    return "Good"
    elif q < 100:   return "Very good"
    elif q < 400:   return "Extremely good"
    else:           return "Exceptionally good"


# ── Step 1: Interactive input ─────────────────────────────────────────────────

def collect_inputs():
    """
    Prompt the user to enter one or more measured values per Q-system parameter.
    - Type a number and press Enter to add a value.
    - Type E or T to finish the current parameter.
    - Type D or press Enter on a blank line (after at least one value) to continue.
    """
    print()
    print("=" * 62)
    print("  Markov Chain Q-System Predictor -- Parameter Input")
    print("=" * 62)
    print("Enter one or more measured field values per parameter.")
    print("Press Enter after each value. Type 'E' when done with a parameter.\n")
    print("Valid ranges / defined ratings:")
    print("  RQD : 0-100  (state by range: 0-25 / 26-50 / 51-75 / 76-100)")
    print("  Jn  : 0.5  2  3  4  6  9  12  15  20")
    print("  Jr  : 4  3  2.5  2  1.5  1  0.5    (best -> worst)")
    print("  Ja  : 0.75  1  2  3  4  6  8  12  13  20")
    print("  Jw  : 1.0  0.66  0.5  0.33  0.1  0.05")
    print("  SRF : 10  7.5  5  2.5  1  2  50  200  400")
    print()

    inputs = {}
    for param in PARAMS:
        vals = []
        print(f"  Enter {param} value(s). Type 'E' when done with this parameter.")
        while True:
            try:
                raw = input("    > ").strip()
            except EOFError:
                # Non-interactive context -- stop collecting
                break
            if not raw:
                if vals:
                    continue  # blank Enter after values -- ignore
                print("    (Enter a number first, then type E to finish.)")
                continue
            if raw.upper() in ('E', 'T'):
                if not vals:
                    print("    (Please enter at least one value first.)")
                    continue
                break
            if raw.upper() == 'D' and vals:
                continue
            try:
                v = float(raw)
                vals.append(v)
                print(f"    Added {v}. Enter another value or type E to finish.")
            except ValueError:
                print(f"    '{raw}' is not a number. Try again or type E to finish.")
        inputs[param] = vals
        print()

    return inputs


# ── Output helpers ────────────────────────────────────────────────────────────

def print_results_table(pred_ratings):
    header = (f"{'Step j':>6} | {'RQD':>7} | {'Jn':>5} | {'Jr':>5} | "
              f"{'Ja':>5} | {'Jw':>6} | {'SRF':>6} | {'Q':>9} | Quality")
    sep = "-" * len(header)

    print()
    print("=" * len(header))
    print("  Markov Chain Q Prediction Results")
    print("=" * len(header))
    print(header)
    print(sep)

    for j in range(1, 31):
        rqd = pred_ratings.get('RQD', {}).get(j, float('nan'))
        jn  = pred_ratings.get('Jn',  {}).get(j, float('nan'))
        jr  = pred_ratings.get('Jr',  {}).get(j, float('nan'))
        ja  = pred_ratings.get('Ja',  {}).get(j, float('nan'))
        jw  = pred_ratings.get('Jw',  {}).get(j, float('nan'))
        srf = pred_ratings.get('SRF', {}).get(j, float('nan'))
        q   = compute_q(pred_ratings, j)

        q_str  = f"{q:.4f}"   if not np.isnan(q) else "    N/A "
        qlabel = classify_q(q) if not np.isnan(q) else ""

        print(f"  j={j:>2}  | {rqd:>7.3f} | {jn:>5.3f} | {jr:>5.3f} | "
              f"{ja:>5.3f} | {jw:>6.4f} | {srf:>6.3f} | {q_str:>9} | {qlabel}")

    print(sep)


def save_excel(pred_ratings, predictions, out_folder):
    """
    Save markov_q_prediction.xlsx to out_folder with:
      - Sheet 'Q Predictions': full table (Step_j, param preds, Q_value, Quality)
      - Sheet '<param>_probs' for each parameter: probability vector at each j step
    """
    from datetime import datetime

    def _resolve_path(folder):
        """Return a writable output path, falling back to a timestamped name if locked."""
        primary = os.path.join(folder, "markov_q_prediction.xlsx")
        if not os.path.exists(primary):
            return primary
        # Test if the existing file can be opened for writing (i.e. not locked by Excel)
        try:
            with open(primary, 'a'):
                pass
            return primary
        except PermissionError:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            alt = os.path.join(folder, f"markov_q_prediction_{stamp}.xlsx")
            print(f"\n  NOTE: markov_q_prediction.xlsx is open in another program.")
            print(f"  Saving to: {alt}")
            return alt

    out_path = _resolve_path(out_folder)

    # Sheet 1: summary table
    rows = []
    for j in range(1, 31):
        row = {'Step_j': j}
        for param in PARAMS:
            row[f'{param}_pred'] = pred_ratings.get(param, {}).get(j, float('nan'))
        q = compute_q(pred_ratings, j)
        row['Q_value'] = q
        row['Quality'] = classify_q(q) if not np.isnan(q) else ""
        rows.append(row)
    summary_df = pd.DataFrame(rows)

    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Q Predictions', index=False)

        # Per-parameter probability evolution sheets
        for param in PARAMS:
            if param not in predictions:
                continue
            n = NUM_STATES[param]
            prob_rows = []
            for j in range(1, 31):
                pvec = predictions[param].get(j, [float('nan')] * n)
                prob_row = {'Step_j': j}
                for s in range(1, n + 1):
                    col = f'State_{s} ({RATINGS[param][s-1]})'
                    prob_row[col] = float(pvec[s - 1]) if len(pvec) >= s else float('nan')
                prob_rows.append(prob_row)
            pd.DataFrame(prob_rows).to_excel(
                writer, sheet_name=f'{param}_probs', index=False)

    print(f"\nResults saved -> {out_path}")
    return out_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Markov Chain Q-System Predictor')
    parser.add_argument(
        '--folder', default=None,
        help='Path to the project folder that contains dynamictpm/ and receives output Excel.'
    )
    parser.add_argument(
        '--demo', action='store_true',
        help='Run with built-in example values (no interactive input).'
    )
    parser.add_argument(
        '--inputs', default=None,
        help='JSON string of inputs, e.g. \'{"RQD":[45,60],"Jn":[6],"Jr":[1.5],"Ja":[2],"Jw":[1.0],"SRF":[2.5]}\''
    )
    args = parser.parse_args()

    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir   = os.path.abspath(args.folder) if args.folder else script_dir
    tpm_file   = os.path.join(data_dir, "dynamictpm", "transition_probability_matrices.xlsx")

    print()
    print("=" * 62)
    print("  Markov Chain Q-System Predictor")
    print("=" * 62)
    print(f"  Data folder : {data_dir}")
    print(f"  TPM file    : {tpm_file}")

    # ── Step 1: Collect inputs ────────────────────────────────────────────────
    if args.inputs:
        try:
            inputs = json.loads(args.inputs)
            # Ensure all params are present
            missing = [p for p in PARAMS if p not in inputs]
            if missing:
                print(f"\nERROR: --inputs JSON missing parameters: {missing}")
                sys.exit(1)
            print("\n[Step 1] Inputs from --inputs argument:")
            for param in PARAMS:
                print(f"  {param}: {inputs[param]}")
        except json.JSONDecodeError as e:
            print(f"\nERROR: Could not parse --inputs JSON: {e}")
            sys.exit(1)
    elif args.demo:
        print("\n[Step 1] Demo mode -- using built-in example values:")
        print("  RQD=20,55  Jn=6  Jr=1.5  Ja=2  Jw=1.0  SRF=2.5")
        inputs = {
            'RQD': [20, 55],
            'Jn':  [6],
            'Jr':  [1.5],
            'Ja':  [2],
            'Jw':  [1.0],
            'SRF': [2.5],
        }
    else:
        inputs = collect_inputs()

    # ── Step 2: Snap to defined ratings ──────────────────────────────────────
    print("\n[Step 2] Mapping values to defined state representatives...")
    snapped = preprocess(inputs)

    # ── Step 3: Convert to state numbers ─────────────────────────────────────
    print("\n[Step 3] Converting to state numbers (1-indexed)...")
    states = to_states(snapped)
    for param in PARAMS:
        st  = states[param]
        sv  = snapped[param]
        labels = [STATE_LABELS[param][s - 1] for s in st]
        print(f"  {param}: {sv} -> states {st}")
        for s, lbl in zip(st, labels):
            print(f"         State {s} = {lbl}")

    # ── Step 4: Build row probability vectors ────────────────────────────────
    print("\n[Step 4] Building row probability vectors...")
    row_vectors = {}
    for param in PARAMS:
        rv = make_row_vector(states[param], NUM_STATES[param])
        row_vectors[param] = rv
        nonzero = [(i + 1, f"{v:.4f}") for i, v in enumerate(rv) if v > 0]
        print(f"  {param}: non-zero -> {nonzero}")

    # ── Step 5: Load TPMs and run 120 multiplications ────────────────────────
    print(f"\n[Step 5] Loading TPMs...")
    tpms = load_tpms(tpm_file)
    print("  Running 6 params x 30 steps = 180 matrix multiplications...")
    predictions = compute_predictions(row_vectors, tpms)

    # ── Step 6: Interpret predictions ────────────────────────────────────────
    print("\n[Step 6] Interpreting results...")
    print("  RQD   : probability-weighted average rating")
    print("  Others: rating of maximum probable state (argmax)")
    print("  Fallback (zero vector): input data value")
    pred_ratings = interpret(predictions, snapped)

    # ── Step 7: Q values and output ───────────────────────────────────────────
    print("\n[Step 7] Computing Q(j) for j=1..30...")
    print_results_table(pred_ratings)

    save_excel(pred_ratings, predictions, data_dir)
    print("\nDone.")


if __name__ == '__main__':
    main()
