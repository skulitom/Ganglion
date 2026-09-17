"""Census step 3a: detail listings (neck MNs, wing/haltere MNs, leg MNs, ocellar-nerve rows, photoreceptor fields,
neurotransmitter coverage for the visual populations, flight-graph node composition)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather

RAW = Path(r'C:\DEV\Haltere\data\raw\male-cns-v1.0')
BUILT = Path(r'C:\DEV\Haltere\data\built')
pd.set_option('display.width', 300)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 60)
pd.set_option('display.max_colwidth', 70)

ann = feather.read_feather(RAW / 'body-annotations-male-cns-v1.0-minconf-0.5.feather')


def section(t):
    print('\n' + '=' * 100 + f'\n{t}\n' + '=' * 100)


COLS = ['bodyId', 'type', 'superclass', 'subclass', 'somaSide', 'somaNeuromere', 'exitNerve', 'entryNerve', 'flywireType',
        'hemibrainType', 'mancType', 'synonyms', 'status']

section("neck motor neurons: EXPR: ann[ann['subclass'] == 'nm']")
nm = ann[ann['subclass'] == 'nm'].sort_values(['superclass', 'type', 'somaSide'])
print(nm[COLS].to_string())
print('\ncount by superclass x somaSide:')
print(pd.crosstab(nm['superclass'], nm['somaSide'].fillna('<NA>'), margins=True).to_string())
print('type list cb_motor nm:', sorted(nm.loc[nm.superclass == 'cb_motor', 'type'].dropna().unique()))
print('type list vnc_motor nm:', sorted(nm.loc[nm.superclass == 'vnc_motor', 'type'].dropna().unique()))
print("\nEXPR: ann[ann['exitNerve'] == 'CvN']  (cervical nerve)")
cv = ann[ann['exitNerve'] == 'CvN']
print(pd.crosstab([cv['superclass'], cv['subclass'].fillna('<NA>')], cv['somaSide'].fillna('<NA>'), margins=True).to_string())

section("wing + haltere motor neurons: EXPR: ann[(ann['superclass']=='vnc_motor') & ann['subclass'].isin(['wm','hm'])]")
wm = ann[(ann['superclass'] == 'vnc_motor') & ann['subclass'].isin(['wm', 'hm'])]
print(pd.crosstab([wm['subclass'], wm['type'].fillna('<NA>')], wm['somaSide'].fillna('<NA>'), margins=True).to_string())
print('by exitNerve:')
print(pd.crosstab(wm['subclass'], wm['exitNerve'].fillna('<NA>'), margins=True).to_string())

section("leg motor neurons: EXPR: ann[(ann['superclass']=='vnc_motor') & ann['subclass'].isin(['fl','ml','hl'])]")
lg = ann[(ann['superclass'] == 'vnc_motor') & ann['subclass'].isin(['fl', 'ml', 'hl'])]
print(pd.crosstab(lg['subclass'], [lg['somaNeuromere'].fillna('<NA>'), lg['somaSide'].fillna('<NA>')], margins=True).to_string())
print('distinct types per leg subclass:', lg.groupby('subclass')['type'].nunique().to_dict())
print('type naming examples:', lg['type'].dropna().unique()[:30].tolist())
print("\nall vnc_motor subclass x somaSide:")
vm = ann[ann['superclass'] == 'vnc_motor']
print(pd.crosstab(vm['subclass'].fillna('<NA>'), vm['somaSide'].fillna('<NA>'), margins=True).to_string())
print("cb_motor subclass x somaSide:")
cm = ann[ann['superclass'] == 'cb_motor']
print(pd.crosstab(cm['subclass'].fillna('<NA>'), cm['somaSide'].fillna('<NA>'), margins=True).to_string())
print('cb_motor types by subclass:')
print(cm.groupby('subclass')['type'].apply(lambda s: sorted(s.dropna().unique())).to_string())

section("ocellar nerve entries: EXPR: ann[ann['entryNerve'] == 'ON']")
on = ann[ann['entryNerve'] == 'ON']
print(on[COLS + ['class', 'instance', 'somaLocation']].to_string())
print("\nEXPR: ann[ann['exitNerve'] == 'ON']")
print(ann[ann['exitNerve'] == 'ON'][COLS + ['class', 'instance']].to_string())
print("\nOCG / OCC types (ocellar interneurons): EXPR: ann[ann['type'].str.match(r'^OC[GC]\\d', na=False)]")
oc = ann[ann['type'].str.match(r'^OC[GC]\d', na=False)]
print(oc[COLS + ['instance']].sort_values(['type', 'somaSide']).to_string())

section('photoreceptors (ol_sensory): fields that could carry the eye side')
pr = ann[ann['superclass'] == 'ol_sensory']
print('instance suffix value_counts: EXPR: pr["instance"].str.extract(r"_([LRM])$")[0].value_counts(dropna=False)')
print(pr['instance'].str.extract(r'_([LRM])$')[0].value_counts(dropna=False).to_dict())
print('instance examples:', pr['instance'].dropna().unique()[:10].tolist())
print('rootSide:', pr['rootSide'].value_counts(dropna=False).to_dict())
print('status:', pr['status'].value_counts(dropna=False).to_dict())
print('class:', pr['class'].value_counts(dropna=False).to_dict())
print('type x status:')
print(pd.crosstab(pr['type'].fillna('<NA>'), pr['status'].fillna('<NA>'), margins=True).to_string())
print('assignedOlHex for photoreceptors non-null:', int(pr['assignedOlHex1'].notna().sum()))
print('somaLocation non-null for photoreceptors:', int(pr['somaLocation'].notna().sum()))

section('class == visual / ol_bilateral')
print(pd.crosstab(ann.loc[ann['class'] == 'visual', 'type'].fillna('<NA>'), ann.loc[ann['class'] == 'visual', 'superclass'].fillna('<NA>')).to_string())
ob = ann[ann['class'] == 'ol_bilateral']
print("\nEXPR: ann[ann['class'] == 'ol_bilateral'] -> types:")
print(pd.crosstab(ob['type'].fillna('<NA>'), ob['somaSide'].fillna('<NA>'), margins=True).to_string())

section('neurotransmitter predictions for the visual populations')
nt = feather.read_feather(RAW / 'body-neurotransmitters-male-cns-v1.0.feather')
print(f'nt rows {len(nt):,d}; distinct body {nt.body.nunique():,d}')
print('consensus_nt value_counts (all bodies):', nt['consensus_nt'].value_counts(dropna=False).to_dict())
print('predicted_nt value_counts:', nt['predicted_nt'].value_counts(dropna=False).to_dict())
print('ground_truth value_counts:', nt['ground_truth'].value_counts(dropna=False).to_dict())
m = ann[['bodyId', 'type', 'superclass', 'somaSide']].merge(nt, left_on='bodyId', right_on='body', how='left')
print('annotation rows with an nt row:', int(m['body'].notna().sum()), 'of', len(m))
for t in ['R1-R6', 'R7p', 'R8p', 'L1', 'L2', 'L3', 'L4', 'L5', 'C2', 'C3', 'T1', 'Mi1', 'Mi4', 'Mi9', 'Tm1', 'Tm2', 'Tm3', 'Tm4', 'Tm9', 'Tm20',
          'T4a', 'T5a', 'HSE', 'HSN', 'HSS', 'VS', 'H1', 'H2', 'VCH', 'DCH', 'LC4', 'LC6', 'LPLC2', 'LPLC1', 'LPLC4', 'LC9', 'LC10a', 'LC11',
          'LC12', 'LC15', 'LC16', 'LC17', 'LC18', 'LC21', 'LC22', 'LLPC1', 'DNp01', 'DNp02', 'DNp11', 'DNp09', 'DNa02', 'DNa01', 'MDN',
          'OCG01a', 'OCG02b', 'GNG276', 'CvN4', 'ADNM1 MN', 'Lawf1', 'Lawf2', 'Dm2']:
    s = m[m['type'] == t]
    if len(s) == 0:
        print(f'  {t:9s} (no rows)')
        continue
    cons = s['consensus_nt'].value_counts(dropna=False).to_dict()
    ct = s['celltype_predicted_nt'].dropna().unique().tolist()
    conf = s['celltype_predicted_nt_confidence'].dropna().unique().round(2).tolist()[:3]
    gt = s['ground_truth'].dropna().unique().tolist()
    print(f'  {t:9s} n={len(s):5d} consensus={cons} celltype_pred={ct} conf={conf} ground_truth={gt}')
print('\nconsensus_nt by superclass for OL/visual superclasses:')
sel = m['superclass'].isin(['ol_intrinsic', 'ol_sensory', 'visual_projection', 'visual_centrifugal', 'descending_neuron'])
print(pd.crosstab(m.loc[sel, 'superclass'], m.loc[sel, 'consensus_nt'].fillna('<no nt row>'), margins=True).to_string())

section('flight graph node composition (data/built/flight.nodes.parquet)')
nodes = pd.read_parquet(BUILT / 'flight.nodes.parquet')
print(pd.crosstab(nodes['superclass'].fillna('<NA>'), nodes['somaSide'].fillna('<NA>'), margins=True).to_string())
print('\nflight nodes: visual types present (type in VPN/LPTC families):')
vis = nodes[nodes['superclass'].isin(['visual_projection', 'visual_centrifugal'])]
print(f'  visual_projection + visual_centrifugal nodes in flight graph: {len(vis)}')
print(vis['type'].value_counts().head(40).to_string())
print('\nflight populations (pop_* columns) sizes:', {c: int(nodes[c].sum()) for c in nodes.columns if c.startswith('pop_')})
print('flight pop_ocelli types:', nodes.loc[nodes['pop_ocelli'], 'type'].value_counts().to_dict())
print('flight pop_lptc types:', nodes.loc[nodes['pop_lptc'], 'type'].value_counts().to_dict())
print('flight pop_wing_mn types:', nodes.loc[nodes['pop_wing_mn'], 'type'].value_counts().to_dict())
print('neck MNs in flight graph:', int((nodes['subclass'] == 'nm').sum()))
print('DNp01/DNp09/DNa02/MDN in flight graph:', nodes.loc[nodes['type'].isin(['DNp01', 'DNp09', 'DNa02', 'DNa01', 'MDN', 'DNp02', 'DNp11']), 'type'].value_counts().to_dict())
