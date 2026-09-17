"""Census step 3: edge / synapse counts from the traced-only weights table, for compute estimates.
Also: photoreceptor side inference, GF / DN / neck-MN partner tables, flight-graph cross-check,
and the difference between the 'traced-only' and 'significant-only' weight files."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather

RAW = Path(r'C:\DEV\Haltere\data\raw\male-cns-v1.0')
BUILT = Path(r'C:\DEV\Haltere\data\built')
pd.set_option('display.width', 400)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 60)
t0 = time.time()


def section(t):
    print('\n' + '=' * 100 + f'\n{t}\n' + '=' * 100, flush=True)


ann = feather.read_feather(RAW / 'body-annotations-male-cns-v1.0-minconf-0.5.feather')
n = len(ann)
body_index = pd.Series(np.arange(n), index=ann['bodyId'].to_numpy())
sc = ann['superclass'].fillna('<NA>').to_numpy().astype(object)
side = ann['somaSide'].fillna('<NA>').to_numpy().astype(object)
sub = ann['subclass'].fillna('<NA>').to_numpy().astype(object)
typ = ann['type'].fillna('<NA>').to_numpy().astype(object)
status = ann['status'].fillna('<NA>').to_numpy().astype(object)

section('load weights (traced-only)')
print("EXPR: W = pyarrow.feather.read_table(path, columns=['body_pre','body_post','weight']).to_pandas()")
W = feather.read_table(str(RAW / 'connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather'),
                       columns=['body_pre', 'body_post', 'weight']).to_pandas()
print(f'rows {len(W):,d}; weight min {W.weight.min()} max {W.weight.max()} sum {int(W.weight.sum()):,d}; '
      f'distinct pre {W.body_pre.nunique():,d} post {W.body_post.nunique():,d}  ({time.time()-t0:.0f}s)')
wq = W['weight'].to_numpy()
print('weight distribution: ' + ', '.join(f'>={t}: {int((wq >= t).sum()):,d} edges / {int(wq[wq >= t].sum()):,d} syn'
                                          for t in (1, 2, 3, 5, 10, 20, 50)))
pre_f = body_index.reindex(W['body_pre'].to_numpy()).to_numpy()
post_f = body_index.reindex(W['body_post'].to_numpy()).to_numpy()
ok = ~np.isnan(pre_f) & ~np.isnan(post_f)
print(f'edges whose endpoints are both in the annotation table: {int(ok.sum()):,d} of {len(W):,d} '
      f'(pre missing {int(np.isnan(pre_f).sum()):,d}, post missing {int(np.isnan(post_f).sum()):,d})')
pre = pre_f[ok].astype(np.int64)
post = post_f[ok].astype(np.int64)
w = W['weight'].to_numpy()[ok].astype(np.int64)
self_loops = int((pre == post).sum())
print(f'self-loops (autapses) {self_loops:,d}')
del W, pre_f, post_f
print('endpoint status (pre):', pd.Series(status[pre]).value_counts().to_dict())
print('endpoint status (post):', pd.Series(status[post]).value_counts().to_dict())
bodies_in_W = np.zeros(n, dtype=bool)
bodies_in_W[pre] = True
bodies_in_W[post] = True
print(f'annotation rows that appear in the weights table: {int(bodies_in_W.sum()):,d}')
print('  by status:', pd.Series(status[bodies_in_W]).value_counts().to_dict())
print("  Traced rows NOT in weights:", int(((status == 'Traced') & ~bodies_in_W).sum()))

# ------------------------------------------------------------------ photoreceptor side inference
section('photoreceptor (ol_sensory) side inference from partners')
print("EXPR: for rows with somaSide NaN, side := majority somaSide over all synaptic partners (pre and post) with a known side")
known = (side == 'L') | (side == 'R')
votes_L = np.zeros(n, dtype=np.int64)
votes_R = np.zeros(n, dtype=np.int64)
# partner side votes, weighted by synapses
for a, b in ((pre, post), (post, pre)):  # a's partner is b
    kb = known[b]
    np.add.at(votes_L, a[kb & (side[b] == 'L')], w[kb & (side[b] == 'L')])
    np.add.at(votes_R, a[kb & (side[b] == 'R')], w[kb & (side[b] == 'R')])
side_inf = side.copy()
nan_side = ~known
inf_L = nan_side & (votes_L > votes_R)
inf_R = nan_side & (votes_R > votes_L)
side_inf[inf_L] = 'L'
side_inf[inf_R] = 'R'
ols = sc == 'ol_sensory'
print(f'ol_sensory rows {int(ols.sum())}: with annotated side {int((ols & known).sum())}, inferred L {int((ols & inf_L).sum())}, '
      f'inferred R {int((ols & inf_R).sum())}, no partners/tie {int((ols & nan_side & ~inf_L & ~inf_R).sum())}')
print(pd.crosstab(pd.Series(typ[ols], name='type'), pd.Series(side_inf[ols], name='side (annotated or inferred)'), margins=True).to_string())
print('ambiguity check (min votes share among inferred ol_sensory):')
tot = votes_L + votes_R
share = np.where(tot > 0, np.maximum(votes_L, votes_R) / np.maximum(tot, 1), np.nan)
print(pd.Series(share[ols & (inf_L | inf_R)]).describe().to_string())
print('\nunsided ol_intrinsic / visual_projection rows and their inferred side:',
      {k: int(v) for k, v in pd.Series(side_inf[(sc == 'ol_intrinsic') & nan_side]).value_counts().items()})

# ------------------------------------------------------------------ masks
section('population masks')
ol_core = np.isin(sc, ['ol_intrinsic', 'ol_sensory'])
vpn = sc == 'visual_projection'
vcf = sc == 'visual_centrifugal'
ol_all = ol_core | vpn | vcf
R = side_inf == 'R'
L = side_inf == 'L'
dn = sc == 'descending_neuron'
an = sc == 'ascending_neuron'
neck = sub == 'nm'
wing = np.isin(sub, ['wm', 'hm']) & (sc == 'vnc_motor')
cb = np.isin(sc, ['cb_intrinsic', 'cb_motor', 'cb_sensory', 'cb_endocrine', 'cb_efferent', 'descending_neuron',
                  'ascending_neuron', 'visual_projection', 'visual_centrifugal', 'sensory_descending', 'efferent_ascending',
                  'efferent_descending'])
nodes = pd.read_parquet(BUILT / 'flight.nodes.parquet')
flight = np.isin(ann['bodyId'].to_numpy(), nodes['bodyId'].to_numpy())
masks = {
    'ol_core (ol_intrinsic+ol_sensory), both sides': ol_core,
    'ol_core R (side annotated or inferred)': ol_core & R,
    'ol_core L': ol_core & L,
    'ol_intrinsic R': (sc == 'ol_intrinsic') & R,
    'ol_sensory R (photoreceptors, inferred side)': (sc == 'ol_sensory') & R,
    'visual_projection R': vpn & R,
    'visual_projection both': vpn,
    'visual_centrifugal R': vcf & R,
    'ol_all R (core+VPN+VCF)': ol_all & R,
    'descending_neuron (all)': dn,
    'ascending_neuron (all)': an,
    'neck MN (subclass nm)': neck,
    'wing+haltere MN (wm/hm)': wing,
    'flight graph nodes': flight,
    'central brain (cb_* + DN + AN + VPN + VCF)': cb,
    'not OL and not central brain (VNC etc.)': ~ol_all & ~cb,
}
for k, m in masks.items():
    print(f'  {k:48s} {int(m.sum()):>8,d} neurons  (in weights table: {int((m & bodies_in_W).sum()):,d})')

THR = (1, 3, 5, 10)


def count(mpre, mpost, label):
    sel = mpre[pre] & mpost[post]
    ws = w[sel]
    parts = []
    for t in THR:
        k = ws >= t
        parts.append(f'w>={t}: {int(k.sum()):>10,d} edges {int(ws[k].sum()):>12,d} syn')
    print(f'  {label:62s} ' + ' | '.join(parts))
    return sel


section('(a) edges WITHIN optic-lobe neuron sets')
count(ol_core & R, ol_core & R, 'within ol_core R')
count((sc == 'ol_intrinsic') & R, (sc == 'ol_intrinsic') & R, 'within ol_intrinsic R')
count(ol_all & R, ol_all & R, 'within ol_all R (core+VPN+VCF)')
count(ol_core & L, ol_core & L, 'within ol_core L')
count(ol_core, ol_core, 'within ol_core both sides')
count(ol_all, ol_all, 'within ol_all both sides')
count(ol_core & R, ol_core & L, 'ol_core R -> ol_core L (crossing)')
count(ol_core & L, ol_core & R, 'ol_core L -> ol_core R (crossing)')
count((sc == 'ol_sensory') & R, ol_core & R, 'photoreceptors R -> ol_core R')

section('(b) edges from optic lobe to the rest')
count(ol_core & R, ~ol_core, 'ol_core R -> anything not ol_core')
count(ol_core & R, vpn, 'ol_core R -> visual_projection (any side)')
count(ol_core & R, vpn & R, 'ol_core R -> visual_projection R')
count(ol_core & R, vcf, 'ol_core R -> visual_centrifugal')
count(ol_core & R, cb & ~vpn & ~vcf, 'ol_core R -> central brain proper (cb_*, DN, AN)')
count(ol_core & R, dn, 'ol_core R -> descending neurons (direct)')
count(vpn & R, ~ol_all, 'visual_projection R -> non-OL (central brain proper etc.)')
count(vpn & R, dn, 'visual_projection R -> descending neurons')
count(vpn & R, neck, 'visual_projection R -> neck MNs')
count(vpn, ~ol_all, 'visual_projection both -> non-OL')
count(~ol_core, ol_core & R, 'anything not ol_core -> ol_core R (feedback)')
count(vcf, ol_core & R, 'visual_centrifugal -> ol_core R')
count(ol_core, ~ol_core, 'ol_core both -> not ol_core')
count(ol_all, ~ol_all, 'ol_all both -> not ol_all')
count(~ol_all, ol_all, 'not ol_all -> ol_all both')

section('(b2) pre-superclass x post-superclass matrix (w>=1): edges and synapses')
print("EXPR: pd.DataFrame({'pre': sc[pre], 'post': sc[post], 'w': w}).groupby(['pre','post'])['w'].agg(['size','sum'])")
df = pd.DataFrame({'pre': sc[pre], 'post': sc[post], 'w': w})
g = df.groupby(['pre', 'post'])['w'].agg(['size', 'sum'])
print('--- EDGES ---')
print(g['size'].unstack(fill_value=0).to_string())
print('--- SYNAPSES ---')
print(g['sum'].unstack(fill_value=0).to_string())
del df, g

section('(c) flight graph data/built/flight.*')
z = np.load(BUILT / 'flight.npz')
fpre, fpost, fsyn = z['pre'], z['post'], z['n_syn']
print(f'nodes {len(nodes):,d}; edges {len(fpre):,d}; total synapses {int(fsyn.sum()):,d}; n_syn min {fsyn.min()} max {fsyn.max()} mean {fsyn.mean():.2f}')
print(f'mean out-degree {len(fpre)/len(nodes):.1f}; self-loops {int((fpre == fpost).sum())}')
print('flight nodes by superclass:')
print(nodes['superclass'].fillna('<NA>').value_counts().to_string())
print('flight nodes by somaSide:', nodes['somaSide'].fillna('<NA>').value_counts().to_dict())
print('flight nodes signs:', nodes['sign'].value_counts().to_dict(), ' nt:', nodes['nt'].value_counts().to_dict())
for k in z.files:
    if k.startswith('pop__'):
        print(f'  {k:18s} {len(z[k]):6d}')
count(flight, flight, 'cross-check: edges among flight nodes in raw table')
print('  (expected 2,767,698 at w>=3 if the built graph == raw table restricted to flight nodes)')
print('flight nodes that are OL / VPN / DN / neck:', {k: int((flight & m).sum()) for k, m in
      [('ol_core', ol_core), ('vpn', vpn), ('vcf', vcf), ('dn', dn), ('neck', neck), ('wing', wing), ('an', an)]})

section('(d) combined real-time set: right OL + VPNs + flight graph + DNs + neck MNs')
combos = {
    'A: ol_core R + VPN R': (ol_core & R) | (vpn & R),
    'B: ol_core R + VPN R + VCF R': (ol_all & R),
    'C: B + DN + neck MN': (ol_all & R) | dn | neck,
    'D: ol_core R + VPN R + flight + DN + neck MN': (ol_core & R) | (vpn & R) | flight | dn | neck,
    'E: ol_core R + VPN both + flight + DN + neck MN': (ol_core & R) | vpn | flight | dn | neck,
    'F: ol_all both sides + flight + DN + neck MN': ol_all | flight | dn | neck,
    'G: everything Traced in annotation table': status == 'Traced',
}
for k, m in combos.items():
    print(f'{k}: {int(m.sum()):,d} neurons')
    count(m, m, '   edges within set')

section('(e) partner tables: giant fiber, escape/steering DNs, neck MNs, loom VPNs')


def partners(sel_mask, label, direction='in', top=25, by='type', min_syn=1):
    if direction == 'in':
        e = sel_mask[post]
        other = pre
    else:
        e = sel_mask[pre]
        other = post
    e &= w >= min_syn
    key = typ[other[e]] if by == 'type' else sc[other[e]]
    d = pd.DataFrame({'k': key, 'w': w[e], 'side': side_inf[other[e]]})
    tot = int(d['w'].sum())
    g = d.groupby('k').agg(syn=('w', 'sum'), edges=('w', 'size'), sides=('side', lambda s: dict(pd.Series(s).value_counts())))
    g = g.sort_values('syn', ascending=False).head(top)
    g['pct'] = (100 * g['syn'] / max(tot, 1)).round(1)
    print(f'\n{label} [{direction}put by {by}] total syn {tot:,d}, edges {int(e.sum()):,d}, n neurons {int(sel_mask.sum())}')
    print(g.to_string())


for t in ['DNp01']:
    m = typ == t
    partners(m, f'{t} (GF)', 'in', by='superclass')
    partners(m, f'{t} (GF)', 'in', top=30)
    partners(m, f'{t} (GF)', 'out', top=25)
for t in ['DNp02', 'DNp11', 'DNp09', 'DNa02', 'DNa01', 'MDN', 'DNp03', 'DNp04', 'DNp06']:
    partners(typ == t, t, 'in', top=15)
partners(neck, 'neck MNs (subclass nm, all 44)', 'in', by='superclass')
partners(neck, 'neck MNs (subclass nm, all 44)', 'in', top=40)
partners(neck & (sc == 'cb_motor'), 'neck MNs in brain (cb_motor nm)', 'in', top=25)
partners(neck & (sc == 'vnc_motor'), 'neck MNs in VNC (vnc_motor nm)', 'in', top=25)
for t in ['LPLC2', 'LC4', 'LC6', 'LPLC4', 'LC16', 'LC22', 'LPLC1', 'LC9', 'LC10a', 'LC11', 'LC12', 'LC15', 'LC17', 'LC18', 'LC21']:
    partners(typ == t, t, 'out', top=12)
for t in ['HSE', 'HSN', 'HSS', 'VS', 'H2', 'H1']:
    partners(typ == t, t, 'out', top=12)
    partners(typ == t, t, 'in', top=12)
partners(wing, 'wing+haltere MNs', 'in', by='superclass')

section('(f) significant-only vs traced-only weight files')
print("EXPR: key = body_pre * 2**31 + body_post  (bodyIds < 2**31); np.setdiff1d over the two key arrays")
Wt = feather.read_table(str(RAW / 'connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather'),
                        columns=['body_pre', 'body_post', 'weight']).to_pandas()
Ws = feather.read_table(str(RAW / 'connectome-weights-male-cns-v1.0-minconf-0.5-significant-only.feather'),
                        columns=['body_pre', 'body_post', 'weight']).to_pandas()
assert Wt.body_pre.max() < 2**31 and Wt.body_post.max() < 2**31 and Ws.body_pre.max() < 2**31 and Ws.body_post.max() < 2**31
kt = Wt.body_pre.to_numpy() * (2**31) + Wt.body_post.to_numpy()
ks = Ws.body_pre.to_numpy() * (2**31) + Ws.body_post.to_numpy()
print(f'traced-only: {len(kt):,d} rows, sum w {int(Wt.weight.sum()):,d}, min w {Wt.weight.min()}, distinct bodies '
      f'{len(np.union1d(Wt.body_pre.unique(), Wt.body_post.unique())):,d}')
print(f'significant-only: {len(ks):,d} rows, sum w {int(Ws.weight.sum()):,d}, min w {Ws.weight.min()}, distinct bodies '
      f'{len(np.union1d(Ws.body_pre.unique(), Ws.body_post.unique())):,d}')
only_t = np.setdiff1d(kt, ks)
only_s = np.setdiff1d(ks, kt)
print(f'edges only in traced-only: {len(only_t):,d}; only in significant-only: {len(only_s):,d}; common {len(np.intersect1d(kt, ks)):,d}')
if len(only_s):
    m = np.isin(ks, only_s)
    bs = np.union1d(Ws.body_pre.to_numpy()[m], Ws.body_post.to_numpy()[m])
    st = ann.set_index('bodyId')['status'].reindex(bs)
    print('  status of bodies on significant-only-exclusive edges:', st.fillna('<not in annotations>').value_counts().to_dict())
    print('  weight distribution of those edges:', pd.Series(Ws.weight.to_numpy()[m]).describe().round(1).to_dict())
if len(only_t):
    m = np.isin(kt, only_t)
    bt = np.union1d(Wt.body_pre.to_numpy()[m], Wt.body_post.to_numpy()[m])
    st = ann.set_index('bodyId')['status'].reindex(bt)
    print('  status of bodies on traced-only-exclusive edges:', st.fillna('<not in annotations>').value_counts().to_dict())
    print('  weight distribution of those edges:', pd.Series(Wt.weight.to_numpy()[m]).describe().round(1).to_dict())
# do common edges have identical weights?
common = np.intersect1d(kt, ks)
it = pd.Series(Wt.weight.to_numpy(), index=kt).loc[common]
is_ = pd.Series(Ws.weight.to_numpy(), index=ks).loc[common]
print(f'  common edges with different weight: {int((it.to_numpy() != is_.to_numpy()).sum()):,d}')
print(f'\ndone in {time.time()-t0:.0f}s')
