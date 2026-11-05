"""Verify shipped data, figures, executed notebooks, and release checksums."""
from pathlib import Path
from zipfile import ZipFile
import hashlib
import json

import pandas as pd
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]

def verify():
    expected={'train':4969,'test-unseen-answers':540,'test-unseen-questions':733,'test-unseen-domains':4562}
    all_data=[]
    for split,n in expected.items():
        df=pd.read_csv(ROOT/f'data/processed/{split}.csv')
        assert len(df)==n,(split,len(df))
        assert df.official_split.eq(split).all()
        assert df.sample_id.is_unique
        assert df[['question_text','reference_answer','student_answer','label']].notna().all().all()
        all_data.append(df)
    data=pd.concat(all_data,ignore_index=True)
    assert data.sample_id.is_unique
    assert data.question_id.nunique()==196
    train=all_data[0]
    assert train.label.value_counts().to_dict()=={'correct':2008,'partially_correct_incomplete':1324,'irrelevant':1115,'contradictory':499,'non_domain':23}
    assert train.normalized_input_hash.nunique()==4826
    assert train.groupby('normalized_input_hash').label.nunique().gt(1).sum()==6
    provenance=json.loads((ROOT/'data/provenance.json').read_text())
    archive=ROOT/'data/raw/scientsbank_core.zip'
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==provenance['bundled_core_sha256']
    with ZipFile(archive) as z:
        assert len(z.namelist())==331
        assert z.testzip() is None
    figures=list((ROOT/'reports/figures').glob('*.png'))
    assert len(figures)==14
    for p in figures:
        with Image.open(p) as image:
            assert image.width>=1000 and image.height>=600
            image.verify()
    for name in ['00_dataset_audit','01_exploratory_analysis']:
        nb=json.loads((ROOT/f'notebooks/{name}.ipynb').read_text())
        assert nb['nbformat']==4
        cells=[c for c in nb['cells'] if c['cell_type']=='code']
        assert all(c['execution_count'] is not None for c in cells)
        assert not any(o['output_type']=='error' for c in cells for o in c['outputs'])
    nb=json.loads((ROOT/'notebooks/01_exploratory_analysis.ipynb').read_text())
    images=sum('image/png' in o.get('data',{}) for c in nb['cells'] if c['cell_type']=='code' for o in c['outputs'])
    assert images==14,images
    checksums=ROOT/'checksums.sha256'
    if checksums.exists():
        for line in checksums.read_text().splitlines():
            digest,path=line.split('  ',1)
            assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    print('PASS: four CSV splits, labels, input audit, 331 XML files, 14 figures, both executed notebooks, 14 embedded plots, and available checksums.')

if __name__=='__main__':verify()
