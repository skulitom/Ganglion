"""Census step 4: follow-ups (photoreceptor side via rootSide/instance, LPTC naming, neck MNs in flight graph,
exhaustive 'ocell' search, soma bounding boxes)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather

RAW = Path(r'C:\DEV\Haltere\data\raw\male-cns-v1.0')
BUILT = Path(r'C:\DEV\Haltere\data\built')
pd.set_option('display.width', 300)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_colwidth', 90)
ann = feather.read_feather(RAW / 'body-annotations-male-cns-v1.0-minconf-0.5.feather')


def section(t):
    print('\n' + '=' * 100 + f'\n{t}\n' + '=' * 100)


section("photoreceptors by rootSide (all, and status=='Traced' only)")
pr = ann[ann['superclass'] == 'ol_sensory'].copy()
pr['instSide'] = pr['instance'].str.extract(r'_([LRM])$')[0]
print('rootSide == instance suffix for all rows:', bool((pr['rootSide'].fillna('?') == pr['instSide'].fillna('?')).all()),
      '; mismatches:', int((pr['rootSide'].fillna('?') != pr['instSide'].fillna('?')).sum()))
print("EXPR: pd.crosstab(pr['type'], pr['rootSide'], margins=True)   [all ol_sensory rows]")
print(pd.crosstab(pr['type'].fillna('<NA>'), pr['rootSide'].fillna('<NA>'), margins=True).to_string())
prt = pr[pr['status'] == 'Traced']
print("\nEXPR: ... restricted to status == 'Traced' (the only rows present in the traced-only weights table)")
print(pd.crosstab(prt['type'].fillna('<NA>'), prt['rootSide'].fillna('<NA>'), margins=True).to_string())
print('\nstatus x rootSide for R1-R6:')
r16 = pr[pr['type'] == 'R1-R6']
print(pd.crosstab(r16['status'].fillna('<NA>'), r16['rootSide'].fillna('<NA>'), margins=True).to_string())

section('LPTC / HS / VS naming across columns')
lp = ann[ann['type'].str.match(r'^(HS|VS|H1$|H2$|VCH|DCH|CH$|Am1)', na=False)]
print(lp[['bodyId', 'type', 'flywireType', 'hemibrainType', 'synonyms', 'instance', 'superclass', 'somaSide', 'somaLocation']].sort_values(['type', 'somaSide']).to_string())

section('neck MNs present in the flight graph')
nodes = pd.read_parquet(BUILT / 'flight.nodes.parquet')
print(nodes.loc[nodes['subclass'] == 'nm', ['bodyId', 'type', 'superclass', 'somaSide', 'exitNerve', 'nt', 'sign']].sort_values('type').to_string())
print('\ncb_motor nodes in flight graph by subclass:', nodes.loc[nodes['superclass'] == 'cb_motor', 'subclass'].value_counts().to_dict())
print('vnc_motor nodes in flight graph by subclass:', nodes.loc[nodes['superclass'] == 'vnc_motor', 'subclass'].value_counts().to_dict())

section("exhaustive search for 'ocel' / 'ocg' over every string column")
strcols = [c for c in ann.columns if ann[c].dtype == object or str(ann[c].dtype) in ('str', 'string', 'category')]
strcols = [c for c in strcols if c not in ('somaLocation', 'tosomaLocation')]
hits = {}
for c in strcols:
    s = ann[c].astype('string')
    m = s.str.contains(r'ocel|ocell', case=False, regex=True).fillna(False)
    if m.any():
        hits[c] = ann.loc[m, c].astype(str).value_counts().to_dict()
print('columns with "ocel" hits:', {k: v for k, v in hits.items()})
print('(class values containing "vis":', sorted(set(v for v in ann['class'].dropna().unique() if 'vis' in v.lower())), ')')

section('soma bounding boxes (voxel units) per group, for drawing')
loc = ann['somaLocation']
has = loc.notna()
xyz = np.array(loc[has].tolist(), dtype=np.int64)
sub = ann.loc[has, ['superclass', 'somaSide', 'type']].reset_index(drop=True)
groups = {
    'ol_intrinsic R': (sub['superclass'] == 'ol_intrinsic') & (sub['somaSide'] == 'R'),
    'ol_intrinsic L': (sub['superclass'] == 'ol_intrinsic') & (sub['somaSide'] == 'L'),
    'visual_projection R': (sub['superclass'] == 'visual_projection') & (sub['somaSide'] == 'R'),
    'visual_projection L': (sub['superclass'] == 'visual_projection') & (sub['somaSide'] == 'L'),
    'cb_intrinsic (all)': sub['superclass'] == 'cb_intrinsic',
    'descending_neuron (all)': sub['superclass'] == 'descending_neuron',
    'vnc_intrinsic (all)': sub['superclass'] == 'vnc_intrinsic',
    'vnc_motor (all)': sub['superclass'] == 'vnc_motor',
    'lamina L1 R': (sub['type'] == 'L1') & (sub['somaSide'] == 'R'),
    'medulla Mi1 R': (sub['type'] == 'Mi1') & (sub['somaSide'] == 'R'),
    'T4a R': (sub['type'] == 'T4a') & (sub['somaSide'] == 'R'),
    'LC4 R': (sub['type'] == 'LC4') & (sub['somaSide'] == 'R'),
    'ALL with somaLocation': pd.Series(True, index=sub.index),
}
print(f"{'group':26s} {'n':>7s}  {'min xyz':>24s}  {'max xyz':>24s}  {'centroid xyz':>24s}")
for k, m in groups.items():
    a = xyz[m.to_numpy()]
    if len(a) == 0:
        print(f'{k:26s} {0:7d}')
        continue
    print(f'{k:26s} {len(a):7d}  {str(a.min(0).tolist()):>24s}  {str(a.max(0).tolist()):>24s}  {str(a.mean(0).round(0).astype(int).tolist()):>24s}')
print('\nnote: x axis separates sides (somaSide R = small x); z runs brain (small z) -> VNC (large z).')

section('type-level L/R table for the requested visual types (compact, for the report)')
want = ['R1-R6', 'R7p', 'R7y', 'R7d', 'R7_unclear', 'R8p', 'R8y', 'R8d', 'R8_unclear', 'R7R8_unclear', 'HBeyelet',
        'L1', 'L2', 'L3', 'L4', 'L5', 'C2', 'C3', 'T1', 'Lawf1', 'Lawf2', 'Lai',
        'Mi1', 'Mi4', 'Mi9', 'Tm1', 'Tm2', 'Tm3', 'Tm4', 'Tm9', 'Tm20', 'T2', 'T2a', 'T3',
        'T4a', 'T4b', 'T4c', 'T4d', 'T4_unclear', 'T5a', 'T5b', 'T5c', 'T5d', 'T5a_unclear',
        'HSN', 'HSE', 'HSS', 'HST', 'VS', 'VSm', 'VST1', 'VST2', 'H1', 'H2', 'VCH', 'DCH', 'Am1',
        'LC4', 'LC6', 'LC9', 'LC10a', 'LC10b', 'LC10c-1', 'LC10c-2', 'LC10d', 'LC10e', 'LC10_unclear', 'LC11', 'LC12', 'LC15', 'LC16', 'LC17',
        'LC18', 'LC21', 'LC22', 'LPLC1', 'LPLC2', 'LPLC4', 'LLPC1', 'LLPC2', 'LLPC3', 'LLPC4', 'LPC1', 'LPC2',
        'DNp01', 'DNp02', 'DNp03', 'DNp04', 'DNp06', 'DNp11', 'DNp09', 'DNa01', 'DNa02', 'MDN',
        'OCG01a', 'OCG01b', 'OCG01c', 'OCG01d', 'OCG01e', 'OCG01f', 'OCG02b', 'OCG02c', 'OCG03', 'OCG06', 'OCC01b', 'OCC02a', 'OCC02b']
side = ann['somaSide'].fillna(ann['rootSide']).fillna('<NA>')
t = ann[ann['type'].isin(want)]
ct = pd.crosstab(t['type'], side[t.index]).reindex(want).fillna(0).astype(int)
ct['total'] = ct.sum(axis=1)
ct['superclass'] = t.groupby('type')['superclass'].first().reindex(want)
ct['traced'] = t[t['status'] == 'Traced'].groupby('type').size().reindex(want).fillna(0).astype(int)
print("EXPR: side = ann['somaSide'].fillna(ann['rootSide']); pd.crosstab(ann['type'], side)")
print(ct.to_string())
missing = [x for x in want if x not in set(ann['type'].dropna())]
print('requested names with NO exact type match:', missing)
for pat in [r'^T2', r'^T3', r'^Am', r'^Y\d', r'^Tlp', r'^LOLP', r'^MeLo', r'^Li\d', r'^LT\d', r'^Y']:
    s = ann.loc[ann['type'].str.match(pat, na=False), 'type'].value_counts()
    print(f'  {pat:8s} -> {s.sum():6d} neurons, {len(s):3d} types: {dict(list(s.items())[:12])}')
