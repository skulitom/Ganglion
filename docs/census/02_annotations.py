"""Census step 2: totals, categorical fields, visual / motor / descending cell types, soma positions.
Every selector is printed as the exact pandas expression used."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather

RAW = Path(r'C:\DEV\Haltere\data\raw\male-cns-v1.0')
pd.set_option('display.width', 250)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_colwidth', 80)

ann = feather.read_feather(RAW / 'body-annotations-male-cns-v1.0-minconf-0.5.feather')
print(f'annotations rows: {len(ann):,d}')


def vc(s, dropna=False, n=None):
    v = s.value_counts(dropna=dropna)
    return v if n is None else v.head(n)


def section(t):
    print('\n' + '=' * 100 + f'\n{t}\n' + '=' * 100)


# ------------------------------------------------------------------ A. status
section('A. status / statusLabel')
print("EXPR: ann['status'].value_counts(dropna=False)")
print(vc(ann['status']).to_string())
print("\nEXPR: ann['statusLabel'].value_counts(dropna=False)")
print(vc(ann['statusLabel']).to_string())
print("\nEXPR: pd.crosstab(ann['status'].fillna('<NA>'), ann['statusLabel'].astype(str))")
print(pd.crosstab(ann['status'].fillna('<NA>'), ann['statusLabel'].astype(str)).to_string())

# ------------------------------------------------------------------ B. superclass / class / subclass
section('B. superclass x somaSide')
print("EXPR: pd.crosstab(ann['superclass'].fillna('<NA>'), ann['somaSide'].fillna('<NA>'), margins=True)")
print(pd.crosstab(ann['superclass'].fillna('<NA>'), ann['somaSide'].fillna('<NA>'), margins=True).to_string())
section('B2. superclass x status')
print(pd.crosstab(ann['superclass'].fillna('<NA>'), ann['status'].fillna('<NA>'), margins=True).to_string())
section('B3. class x somaSide')
print("EXPR: pd.crosstab(ann['class'].fillna('<NA>'), ann['somaSide'].fillna('<NA>'), margins=True)")
print(pd.crosstab(ann['class'].fillna('<NA>'), ann['somaSide'].fillna('<NA>'), margins=True).to_string())
section('B4. class x superclass (non-null class only)')
print(pd.crosstab(ann['class'].fillna('<NA>'), ann['superclass'].fillna('<NA>')).to_string())
section('B5. subclass (all values) x superclass')
print(pd.crosstab(ann['subclass'].fillna('<NA>'), ann['superclass'].fillna('<NA>')).to_string())
section('B6. other categorical columns')
for col in ['supertype', 'somaNeuromere', 'rootSide', 'entryNerve', 'exitNerve', 'receptorType', 'dimorphism',
            'birthtime', 'fruDsx', 'serialMotif']:
    print(f"\nEXPR: ann[{col!r}].value_counts(dropna=False)")
    print(vc(ann[col], n=60).to_string())

# ------------------------------------------------------------------ C. typed neurons
section('C. how many have types')
for col in ['type', 'flywireType', 'hemibrainType', 'mancType', 'synonyms', 'instance', 'somaSide', 'somaLocation']:
    print(f"{col:14s} non-null: {int(ann[col].notna().sum()):>8,d}   among status=='Traced': "
          f"{int((ann[col].notna() & (ann['status'] == 'Traced')).sum()):>8,d}")
print(f"\ndistinct type values: {ann['type'].nunique():,d}; distinct flywireType: {ann['flywireType'].nunique():,d}; "
      f"distinct hemibrainType: {ann['hemibrainType'].nunique():,d}; distinct mancType: {ann['mancType'].nunique():,d}")
print("\nEXPR: ann.groupby(ann['superclass'].fillna('<NA>'))['type'].agg(['size', 'count', 'nunique'])")
print(ann.groupby(ann['superclass'].fillna('<NA>'))['type'].agg(['size', 'count', 'nunique']).to_string())

# ------------------------------------------------------------------ D. optic lobes
section('D. optic lobe membership')
OL_CORE = ['ol_intrinsic', 'ol_sensory']
OL_ALL = OL_CORE + ['visual_projection', 'visual_centrifugal']
print(f"OL_CORE = {OL_CORE}; OL_ALL = {OL_ALL}")
print("EXPR: ol_core = ann['superclass'].isin(OL_CORE); ol_all = ann['superclass'].isin(OL_ALL)")
ol_core = ann['superclass'].isin(OL_CORE)
ol_all = ann['superclass'].isin(OL_ALL)
print(pd.crosstab(ann.loc[ol_all, 'superclass'], ann.loc[ol_all, 'somaSide'].fillna('<NA>'), margins=True).to_string())
print('\nby status:')
print(pd.crosstab(ann.loc[ol_all, 'superclass'], ann.loc[ol_all, 'status'].fillna('<NA>'), margins=True).to_string())
print("\nassignedOlHex1/2 (hex-column coordinates) non-null by superclass x somaSide:")
print("EXPR: hexed = ann['assignedOlHex1'].notna()")
hexed = ann['assignedOlHex1'].notna()
print(pd.crosstab(ann.loc[hexed, 'superclass'].fillna('<NA>'), ann.loc[hexed, 'somaSide'].fillna('<NA>'), margins=True).to_string())
print('\nassignedOlHex ranges: hex1 [%g, %g], hex2 [%g, %g]' % (ann['assignedOlHex1'].min(), ann['assignedOlHex1'].max(),
                                                               ann['assignedOlHex2'].min(), ann['assignedOlHex2'].max()))
print('distinct (hex1,hex2) pairs by side:')
print(ann.loc[hexed].groupby('somaSide')[['assignedOlHex1', 'assignedOlHex2']].apply(lambda d: len(d.drop_duplicates())).to_string())
print('\nEXPR: ann.loc[hexed, "type"].value_counts()  (top 60 types carrying hex coordinates)')
print(ann.loc[hexed, 'type'].value_counts().head(60).to_string())
print('\nnumber of types carrying hex coords:', ann.loc[hexed, 'type'].nunique())
print('ALL types with hex coords and counts per side:')
print(pd.crosstab(ann.loc[hexed, 'type'], ann.loc[hexed, 'somaSide'].fillna('<NA>')).to_string())

# ------------------------------------------------------------------ E. visual cell types
section('E. visual / escape / motor / descending cell types (case-insensitive regex over several columns)')
SEARCH_COLS = ['type', 'flywireType', 'hemibrainType', 'mancType', 'synonyms', 'instance']


def find(label, pattern, cols=SEARCH_COLS, show_cols=('type',), by='somaSide', limit=80):
    """Rows where any of `cols` matches `pattern` (case-insensitive). Prints the actual `type` strings found."""
    pat = re.compile(pattern, re.IGNORECASE)
    m = np.zeros(len(ann), dtype=bool)
    hits = {}
    for c in cols:
        s = ann[c]
        mc = s.map(lambda x: bool(pat.search(x)) if isinstance(x, str) else False).to_numpy()
        hits[c] = int(mc.sum())
        m |= mc
    print(f'\n--- {label}: pattern r{pattern!r} over {cols}')
    print(f'    EXPR: m = ann[cols].apply(lambda s: s.str.contains(r{pattern!r}, case=False, regex=True, na=False)).any(axis=1)')
    print(f'    matched rows: {int(m.sum()):,d}   per-column hits: {hits}')
    if m.sum() == 0:
        print('    (no match)')
        return m
    sub = ann.loc[m]
    ct = pd.crosstab(sub['type'].fillna('<NA type>'), sub[by].fillna('<NA>'), margins=True)
    if len(ct) > limit:
        print(f'    ({len(ct)-1} distinct types; showing first {limit})')
        ct = ct.head(limit)
    print(ct.to_string())
    # if matched via other columns but type differs, show mapping
    other = sub[[c for c in cols if c != 'type']]
    diff = sub[~sub['type'].map(lambda x: bool(pat.search(x)) if isinstance(x, str) else False)]
    if len(diff):
        print(f'    rows matched only via non-type columns: {len(diff)}')
        print(diff[['bodyId', 'type', 'superclass', 'somaSide'] + [c for c in cols if c != 'type']].head(25).to_string())
    return m


# photoreceptors
find('photoreceptors R1-6 / R7 / R8', r'^R(1-6|1|7|8)')
find('anything starting with R + digit (all photoreceptor-like types)', r'^R\d', cols=['type'])
find('ocellar / OCG / ocelli', r'ocell|^OCG|^OC\d')
print("\nocellar nerve entry: EXPR: ann[ann['entryNerve'] == 'ON']")
on = ann[ann['entryNerve'] == 'ON']
print(pd.crosstab(on['type'].fillna('<NA>'), on['somaSide'].fillna('<NA>'), margins=True).to_string())
print(pd.crosstab(on['superclass'].fillna('<NA>'), on['status'].fillna('<NA>'), margins=True).to_string())

# lamina
find('lamina L1-L5', r'^L[1-5]$', cols=['type'])
find('lamina C2/C3', r'^C[23]$', cols=['type'])
find('lamina T1', r'^T1$', cols=['type'])
find('lamina wide-field Lawf / Lai / Lat', r'^(Lawf|Lai|Lat)', cols=['type'])
# medulla
find('medulla Mi1/Mi4/Mi9', r'^Mi(1|4|9)$', cols=['type'])
find('all Mi*', r'^Mi\d', cols=['type'])
find('medulla Tm1/2/3/4/9/20', r'^Tm(1|2|3|4|9|20)$', cols=['type'])
find('all Tm* (not TmY)', r'^Tm\d', cols=['type'])
find('all TmY*', r'^TmY', cols=['type'])
find('all Dm*/Pm*/Sm*/Cm* (medulla interneurons)', r'^(Dm|Pm|Sm|Cm)\d', cols=['type'], limit=200)
find('all Mi/Tm/TmY/Dm/Pm/Sm/Cm/Y/Tlp/T2/T3 (medulla-lobula families)', r'^(Mi|Tm|TmY|Dm|Pm|Sm|Cm|Y|Tlp|T2|T3|T2a|T3)\d?', cols=['type'], limit=5)
# motion detectors
find('T4 a-d', r'^T4', cols=['type'])
find('T5 a-d', r'^T5', cols=['type'])
# LPTCs
find('HS (HSN/HSE/HSS/HST)', r'^HS', cols=['type'])
find('VS (VS1-10, VSm, VST)', r'^VS', cols=['type'])
find('H1 / H2', r'^H[12]$', cols=['type'])
find('CH (VCH / DCH)', r'^(V|D)?CH$', cols=['type'])
find('other LPTC-like: LPT, LPLC, LLPC, LPi?', r'^(LPT|LLPC|LPLC)', cols=['type'], limit=120)
find('lobula plate intrinsic LPi', r'^LPi', cols=['type'], limit=60)
# VPNs
find('LC4/6/9/10/11/12/15/16/17/18/21/22', r'^LC(4|6|9|10|11|12|15|16|17|18|21|22)([a-z]|$)', cols=['type'])
find('ALL LC*', r'^LC\d', cols=['type'], limit=120)
find('LPLC1/2/4', r'^LPLC(1|2|4)', cols=['type'])
find('LLPC', r'^LLPC', cols=['type'])
find('visual_projection superclass: every type', r'.', cols=['type'], limit=400) if False else None
vp = ann[ann['superclass'] == 'visual_projection']
print("\nEXPR: vp = ann[ann['superclass'] == 'visual_projection']; pd.crosstab(vp['type'], vp['somaSide'])")
print(f'visual_projection neurons: {len(vp):,d}; distinct types: {vp["type"].nunique()}')
print(pd.crosstab(vp['type'].fillna('<NA>'), vp['somaSide'].fillna('<NA>'), margins=True).to_string())
vcf = ann[ann['superclass'] == 'visual_centrifugal']
print("\nEXPR: vcf = ann[ann['superclass'] == 'visual_centrifugal']")
print(f'visual_centrifugal neurons: {len(vcf):,d}; distinct types: {vcf["type"].nunique()}')
print(pd.crosstab(vcf['type'].fillna('<NA>'), vcf['somaSide'].fillna('<NA>'), margins=True).to_string())

# giant fiber & DNs
find('DNp01 / Giant Fiber', r'^DNp01|giant fiber|\bGF\b')
find('GF partners DNp02 / DNp11 / PVLP / DNp03/04/06', r'^DNp(02|03|04|06|11)$|^PVLP', cols=['type'])
find('DNp09', r'^DNp09', cols=['type'])
find('DNa01 / DNa02 (steering)', r'^DNa0[12]', cols=['type'])
find('MDN (moonwalker)', r'MDN|moonwalk')
find('DNp/DNa/DNb/DNg (all named DN families, first 200)', r'^DN[a-z]+\d', cols=['type'], limit=200)
dn = ann[ann['superclass'] == 'descending_neuron']
print("\nEXPR: dn = ann[ann['superclass'] == 'descending_neuron']")
print(f'descending_neuron rows: {len(dn):,d}; typed: {int(dn["type"].notna().sum())}; distinct types: {dn["type"].nunique()}')
print(pd.crosstab(dn['somaSide'].fillna('<NA>'), dn['status'].fillna('<NA>'), margins=True).to_string())
print('DN type-prefix families (regex ^(DN[a-z]+|MDN|...) ):')
pref = dn['type'].fillna('<NA>').str.extract(r'^([A-Za-z]+)')[0].fillna('<NA>')
print(pd.crosstab(pref, dn['somaSide'].fillna('<NA>'), margins=True).to_string())
print('DN subclass values:')
print(pd.crosstab(dn['subclass'].fillna('<NA>'), dn['somaSide'].fillna('<NA>'), margins=True).to_string())
print('DN class values:')
print(vc(dn['class']).to_string())
an_sc = [s for s in ann['superclass'].dropna().unique() if 'ascend' in s]
print(f"\nascending superclasses present: {an_sc}")
an = ann[ann['superclass'].isin(an_sc)]
print("EXPR: an = ann[ann['superclass'].isin([s for s in ann['superclass'].dropna().unique() if 'ascend' in s])]")
print(pd.crosstab(an['superclass'], an['somaSide'].fillna('<NA>'), margins=True).to_string())
print(f'ascending typed: {int(an["type"].notna().sum())}; distinct types: {an["type"].nunique()}')
find('AN* type prefix', r'^AN\d|^AN[a-z]', cols=['type'], limit=10)

# motor neurons
section('F. motor neurons')
mn = ann[ann['superclass'].fillna('').str.contains('motor')]
print("EXPR: mn = ann[ann['superclass'].fillna('').str.contains('motor')]")
print(pd.crosstab(mn['superclass'], mn['somaSide'].fillna('<NA>'), margins=True).to_string())
print('\nmotor superclass x subclass:')
print(pd.crosstab(mn['subclass'].fillna('<NA>'), mn['superclass'], margins=True).to_string())
print('\nmotor subclass x somaNeuromere:')
print(pd.crosstab(mn['subclass'].fillna('<NA>'), mn['somaNeuromere'].fillna('<NA>'), margins=True).to_string())
print('\nmotor subclass x exitNerve:')
print(pd.crosstab(mn['subclass'].fillna('<NA>'), mn['exitNerve'].fillna('<NA>'), margins=True).to_string())
print('\nmotor class values:')
print(pd.crosstab(mn['class'].fillna('<NA>'), mn['superclass'], margins=True).to_string())
find('neck / NMN / cervical / CvN anywhere', r'neck|NMN|cervical|CvN', cols=SEARCH_COLS + ['subclass', 'class', 'exitNerve', 'entryNerve', 'matchingNotes'])
print("\nEXPR: ann[ann['exitNerve'].fillna('').str.contains('Cv', case=False)]")
cv = ann[ann['exitNerve'].fillna('').str.contains('Cv', case=False)]
print(pd.crosstab(cv['type'].fillna('<NA>'), [cv['superclass'].fillna('<NA>'), cv['somaSide'].fillna('<NA>')], margins=True).to_string())
find('wing / haltere MN names (MNwm, MNhm, DLM, DVM, b1, b2, i1, iii, hg, ps)', r'^MN(wm|hm)|^(DLM|DVM|b[123]|i[12]|iii[123]|hg[1-4]|ps[12]|tp[12]|tt|III|I1|I2)', cols=['type'], limit=100)
print("\nEXPR: legmn = mn[mn['subclass'].fillna('').str.contains('leg|^fl$|^ml$|^hl$|T[123]', case=False)]  -- see subclass table above for the actual codes")
print("\nleg-like MN types (type contains 'leg' or starts with MN + leg muscle abbreviations):")
find('leg MN types', r'leg|^MN(fl|ml|hl)|^(fe|ti|ta|tr|co|st|lt|ltm|Ti|Fe|Ta|Tr|Co|St)[a-z]*\d*$', cols=['type', 'subclass', 'synonyms'], limit=150)

# ------------------------------------------------------------------ G. soma positions
section('G. soma positions')
loc = ann['somaLocation']
has = loc.notna()
print(f"EXPR: xyz = np.array(ann.loc[ann['somaLocation'].notna(), 'somaLocation'].tolist())  ->  shape")
lens = loc[has].map(len)
print(f'somaLocation non-null: {int(has.sum()):,d}; element lengths: {lens.value_counts().to_dict()}')
xyz = np.array(loc[has & (lens == 3)].tolist(), dtype=np.int64)
print(f'xyz shape {xyz.shape}; per-axis min {xyz.min(0).tolist()} max {xyz.max(0).tolist()} mean {xyz.mean(0).round(0).tolist()}')
print('assuming 8 nm isotropic voxels (male-CNS EM voxel size): extents in um =', ((xyz.max(0) - xyz.min(0)) * 8 / 1000.0).round(1).tolist())
print('\nsomaLocation presence by superclass:')
print("EXPR: ann.groupby(ann['superclass'].fillna('<NA>'))['somaLocation'].agg(['size', 'count'])")
print(ann.groupby(ann['superclass'].fillna('<NA>'))['somaLocation'].agg(['size', 'count']).to_string())
print('\nsomaLocation presence by status:')
print(ann.groupby(ann['status'].fillna('<NA>'))['somaLocation'].agg(['size', 'count']).to_string())
print('\ntosomaLocation non-null:', int(ann['tosomaLocation'].notna().sum()), '; sample:', ann.loc[ann['tosomaLocation'].notna(), ['bodyId', 'type', 'tosomaLocation']].head(3).to_dict('records'))
# per-side centroid of optic-lobe somata to show L/R axis orientation
for sc in ['ol_intrinsic', 'descending_neuron', 'vnc_motor']:
    for side in ['L', 'R']:
        sel = has & (lens == 3) & (ann['superclass'] == sc) & (ann['somaSide'] == side)
        if sel.sum():
            a = np.array(loc[sel].tolist())
            print(f'  centroid {sc:18s} side {side}: {a.mean(0).round(0).tolist()}  (n={int(sel.sum())})')
for t in ['DNp01', 'HSE', 'VS', 'LC4', 'LPLC2', 'T4a', 'Mi1', 'L1']:
    sel = has & (lens == 3) & (ann['type'] == t)
    print(f'  {t:6s} n={int(sel.sum()):5d} soma xyz examples: {loc[sel].head(2).tolist()}  sides {ann.loc[sel, "somaSide"].value_counts().to_dict()}')
print('\nsomaSide value_counts among ol_intrinsic:', ann.loc[ann['superclass'] == 'ol_intrinsic', 'somaSide'].value_counts(dropna=False).to_dict())
print('rootSide value_counts:', ann['rootSide'].value_counts(dropna=False).to_dict())
