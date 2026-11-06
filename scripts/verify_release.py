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
    from build_splits import verify_split
    verify_split(ROOT)
    split_nb=json.loads((ROOT/'notebooks/02_split_design.ipynb').read_text())
    split_cells=[c for c in split_nb['cells'] if c['cell_type']=='code']
    assert all(c['execution_count'] is not None for c in split_cells)
    assert not any(o['output_type']=='error' for c in split_cells for o in c['outputs'])
    assert sum('image/png' in o.get('data',{}) for c in split_cells for o in c['outputs'])==3
    for file in ['01_partition_sizes.png','02_class_support.png','03_domain_question_support.png']:
        with Image.open(ROOT/'reports/splits/figures'/file) as image:
            assert image.width>=1000 and image.height>=600
            image.verify()
    from build_warehouse import validate_features
    validate_features(train,pd.read_csv(ROOT/'data/processed/training_features.csv'))
    advanced=json.loads((ROOT/'reports/advanced/analysis_manifest.json').read_text())
    assert advanced['source_sha256']==hashlib.sha256((ROOT/'data/processed/train.csv').read_bytes()).hexdigest()
    assert advanced['assignment_sha256']==hashlib.sha256((ROOT/'data/splits/v1/answer_assignments.csv').read_bytes()).hexdigest()
    assert advanced['official_test_files_read'] is False and advanced['model_training_performed'] is False
    quality=json.loads((ROOT/'reports/engineering/quality_report.json').read_text())
    assert quality['all_passed'] and len(quality['checks'])==12
    lineage=json.loads((ROOT/'reports/engineering/lineage.json').read_text())
    for name,digest in lineage['inputs_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    figures=list((ROOT/'reports/advanced/figures').glob('*.png'))
    assert len(figures)==7
    for path in figures:
        with Image.open(path) as image:
            assert image.width>=1000 and image.height>=400
            image.verify()
    for name,expected_images in [('01b_statistical_analysis',7),('02b_data_pipeline',0)]:
        notebook=json.loads((ROOT/f'notebooks/{name}.ipynb').read_text())
        cells=[c for c in notebook['cells'] if c['cell_type']=='code']
        assert all(c['execution_count'] is not None for c in cells)
        assert not any(o['output_type']=='error' for c in cells for o in c['outputs'])
        assert sum('image/png' in o.get('data',{}) for c in cells for o in c['outputs'])==expected_images
    checksums=ROOT/'checksums.sha256'
    if checksums.exists():
        for line in checksums.read_text().splitlines():
            digest,path=line.split('  ',1)
            assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    print('PASS: original data/splits preserved; 24 figures, five executed notebooks, feature contracts, SQL lineage, and release checksums verified.')

if __name__=='__main__':verify()
