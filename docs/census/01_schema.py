"""Census step 1: files, schemas, row counts, sample values of the raw male-CNS v1.0 flat connectome."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.feather as feather

RAW = Path(r'C:\DEV\Haltere\data\raw\male-cns-v1.0')
BUILT = Path(r'C:\DEV\Haltere\data\built')

pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_colwidth', 60)

print('=' * 100)
print('RAW FILES')
print('=' * 100)
for p in sorted(RAW.iterdir()):
    print(f'{p.stat().st_size:>14,d}  {p}')
print()
print('--- .download.log ---')
print((RAW / '.download.log').read_text(encoding='utf-8', errors='replace'))

for p in sorted(RAW.glob('*.feather')):
    print('=' * 100)
    print(p.name, f'({p.stat().st_size/1e6:.1f} MB)')
    print('=' * 100)
    with pa.memory_map(str(p)) as mm:
        try:
            rf = ipc.open_file(mm)
            print(f'  Arrow IPC file: {rf.num_record_batches} record batches')
            schema = rf.schema
            nrows = sum(rf.get_batch(i).num_rows for i in range(rf.num_record_batches))
        except Exception as e:  # feather v1 fallback
            print('  ipc.open_file failed:', e)
            tbl = feather.read_table(str(p))
            schema = tbl.schema
            nrows = tbl.num_rows
    print(f'  rows: {nrows:,d}')
    print(f'  columns ({len(schema.names)}):')
    for f in schema:
        print(f'    {f.name!r:45s} {str(f.type)}')
    if schema.metadata:
        md = {k.decode(): (v.decode()[:300] if isinstance(v, bytes) else v) for k, v in schema.metadata.items()}
        print('  schema metadata keys:', list(md.keys()))
        for k, v in md.items():
            if k != 'pandas':
                print(f'    {k}: {v}')
    # sample rows (small table -> whole thing, big -> first batch only)
    if p.stat().st_size < 100e6:
        df = feather.read_feather(str(p))
        print('  head(5):')
        print(df.head(5).to_string())
        print('  dtypes:')
        print(df.dtypes.to_string())
        print('  non-null counts:')
        print(df.notna().sum().to_string())
    else:
        with pa.memory_map(str(p)) as mm:
            rf = ipc.open_file(mm)
            b = rf.get_batch(0).to_pandas()
        print('  first batch head(5):')
        print(b.head(5).to_string())
        print('  dtypes:')
        print(b.dtypes.to_string())
    print()

print('=' * 100)
print('BUILT: data/built/flight.*')
print('=' * 100)
for p in sorted(BUILT.iterdir()):
    print(f'{p.stat().st_size:>14,d}  {p}')
print('--- flight.meta.json ---')
print(json.dumps(json.load(open(BUILT / 'flight.meta.json', encoding='utf-8')), indent=2))
z = np.load(BUILT / 'flight.npz')
print('--- flight.npz keys ---')
for k in z.files:
    a = z[k]
    print(f'  {k:28s} shape={a.shape} dtype={a.dtype}')
nodes = pd.read_parquet(BUILT / 'flight.nodes.parquet')
print('--- flight.nodes.parquet ---')
print(f'  rows: {len(nodes):,d}; columns ({len(nodes.columns)}): {list(nodes.columns)}')
print(nodes.dtypes.to_string())
print(nodes.head(3).to_string())
