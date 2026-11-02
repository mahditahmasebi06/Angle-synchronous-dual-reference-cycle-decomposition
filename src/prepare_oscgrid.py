"""Extract the OscGrid file unlabeled_50_6400.7z (50 Hz, 6400 Hz) and store one array per record in
data/oscgrid/u6400_recs.npz (columns IA, IB, IC, UA BB, UB BB, UC BB, UN BB, UA CL, UB CL, UC CL, UN CL).

Download unlabeled_50_6400.7z from https://doi.org/10.6084/m9.figshare.28465427 (CC BY-NC-ND 4.0) into
data/oscgrid/ (or set OSCGRID_7Z). The extracted data are only used locally and are not redistributed."""
import csv, os, sys
import numpy as np
import paths, mini7z

COLS = ['IA', 'IB', 'IC', 'UA BB', 'UB BB', 'UC BB', 'UN BB', 'UA CL', 'UB CL', 'UC CL', 'UN CL']

if __name__ == '__main__':
    out_dir = os.path.join(os.path.dirname(paths.OSC_NPZ), 'u6400')
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'unlabeled_50_6400.csv')
    if not os.path.exists(csv_path):
        mini7z.extract(paths.OSC_7Z, out_dir)
    recs, names = {}, []
    with open(csv_path, newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            nm = row['file_name']
            if nm not in recs:
                recs[nm] = []; names.append(nm)
            recs[nm].append([float(row[c]) if row[c] != '' else np.nan for c in COLS])
    arr = {f'r{i}': np.array(recs[n]) for i, n in enumerate(names)}
    np.savez_compressed(paths.OSC_NPZ, names=np.array(names), **arr)
    print(paths.OSC_NPZ, len(names), 'records')
