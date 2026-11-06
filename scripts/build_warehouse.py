"""Build a constrained, normalized SQLite warehouse and reconcile SQL outputs."""
from pathlib import Path
import hashlib,json,os,sqlite3
import numpy as np
import pandas as pd
from build_splits import load_training,validate_assignments,verify_split,LABELS
ROOT=Path(__file__).resolve().parents[1]

def validate_features(data,features):
    if not features.sample_id.is_unique or set(features.sample_id)!=set(data.sample_id):
        raise ValueError('Features must cover each training answer exactly once.')
    if features.isna().any().any():raise ValueError('Missing feature values are prohibited.')
    if not features.select_dtypes('number').apply(lambda x:np.isfinite(x).all()).all():raise ValueError('Non-finite feature values.')
    for col in ['type_token_ratio','question_token_jaccard','reference_token_jaccard','punctuation_fraction']:
        if not features[col].between(0,1).all():raise ValueError(f'Out-of-range {col}.')
    if not features.answer_word_count.gt(0).all() or not features.answer_char_count.gt(0).all():raise ValueError('Empty answer feature.')
    aligned=features.set_index('sample_id').loc[data.sample_id]
    if not np.array_equal(aligned.answer_word_count.to_numpy(),data.answer_word_count.to_numpy()):raise ValueError('Word-count reconciliation failed.')
    if not np.allclose(aligned.reference_token_jaccard,data.reference_token_jaccard):raise ValueError('Reference-overlap reconciliation failed.')

def build(root=ROOT,db_path=None,write_reports=True):
    data=load_training(root);verify_split(root)
    features=pd.read_csv(root/'data/processed/training_features.csv')
    assignments=pd.read_csv(root/'data/splits/v1/answer_assignments.csv')
    validate_features(data,features);validate_assignments(data,assignments)
    contract=json.loads((root/'data/contracts/training_contract.json').read_text())
    if len(data)!=contract['expected_records'] or data.domain.nunique()!=contract['expected_domains'] or set(data.label)!=set(contract['labels']):raise ValueError('Data contract mismatch.')
    context=data.groupby('question_id')[['domain','question_text','reference_answer']].nunique()
    if not context.eq(1).all().all():raise ValueError('Conflicting question context.')
    for col in contract['text_fields_must_be_nonempty']:
        if data[col].isna().any() or data[col].str.strip().eq('').any():raise ValueError('Empty source text.')
    db_path=Path(db_path) if db_path else root/'data/warehouse/scientsbank.sqlite'
    db_path.parent.mkdir(parents=True,exist_ok=True);temporary=db_path.with_suffix('.building.sqlite')
    if temporary.exists():temporary.unlink()
    conn=sqlite3.connect(temporary)
    try:
        conn.executescript((root/'sql/01_schema.sql').read_text())
        pd.DataFrame({'domain':sorted(data.domain.unique())}).to_sql('domains',conn,if_exists='append',index=False)
        pd.DataFrame({'label':LABELS}).to_sql('labels',conn,if_exists='append',index=False)
        data[['question_id','domain','question_text','reference_answer']].drop_duplicates('question_id').sort_values('question_id').to_sql('questions',conn,if_exists='append',index=False)
        data[['sample_id','answer_id','question_id','student_answer','label','normalized_input_hash']].sort_values('sample_id').to_sql('answers',conn,if_exists='append',index=False)
        features.sort_values('sample_id').to_sql('answer_features',conn,if_exists='append',index=False)
        assignments[['sample_id','internal_split','input_multiplicity','has_conflicting_labels','primary_sample_weight','sensitivity_fit_weight']].sort_values('sample_id').to_sql('development_assignments',conn,if_exists='append',index=False)
        conn.executescript((root/'sql/02_analytics.sql').read_text())
        if conn.execute('PRAGMA foreign_key_check').fetchall():raise ValueError('Foreign-key violation.')
        if conn.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise ValueError('SQLite integrity check failed.')
        metrics={name:conn.execute(f'SELECT COUNT(*) FROM {name}').fetchone()[0] for name in ['domains','labels','questions','answers','answer_features','development_assignments']}
        if [metrics[k] for k in ['answers','answer_features','development_assignments']]!=[4969]*3:raise ValueError('Unexpected join coverage.')
        sql_labels=pd.read_sql_query('SELECT label,answer_count FROM v_label_summary ORDER BY label',conn).set_index('label').answer_count
        pd.testing.assert_series_equal(sql_labels,data.label.value_counts().sort_index(),check_names=False)
        sql_q=pd.read_sql_query('SELECT question_id,answer_count FROM v_question_metrics ORDER BY question_id',conn).set_index('question_id').answer_count
        pd.testing.assert_series_equal(sql_q,data.groupby('question_id').size().sort_index(),check_names=False)
        if write_reports:
            out=root/'reports/engineering';out.mkdir(parents=True,exist_ok=True)
            for view in ['v_question_metrics','v_domain_summary','v_label_summary','v_partition_summary']:
                columns={'v_question_metrics':'question_id','v_domain_summary':'domain','v_label_summary':'label','v_partition_summary':'internal_split'}
                pd.read_sql_query(f'SELECT * FROM {view} ORDER BY {columns[view]}',conn).to_csv(out/f'{view}.csv',index=False)
        conn.commit();conn.close();os.replace(temporary,db_path)
    except Exception:
        conn.close()
        if temporary.exists():temporary.unlink()
        raise
    report=dict(contract_version=contract['version'],all_passed=True,table_counts=metrics,checks=['source record/question/domain counts','source primary key and allowed labels','nonempty source text','single context per question','feature primary key and complete one-to-one coverage','finite/range-constrained features','word-count and reference-overlap reconciliation','development assignment preservation and disjointness','SQLite primary/foreign/check constraints','SQLite foreign-key and integrity checks','SQL/Pandas label-count reconciliation','SQL/Pandas question-count reconciliation'],official_test_files_read=False,model_training_performed=False)
    if write_reports:
        (root/'reports/engineering/quality_report.json').write_text(json.dumps(report,indent=2)+'\n')
        inputs=['data/processed/train.csv','data/processed/training_features.csv','data/splits/v1/answer_assignments.csv','data/contracts/training_contract.json','sql/01_schema.sql','sql/02_analytics.sql']
        lineage={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in inputs}
        (root/'reports/engineering/lineage.json').write_text(json.dumps({'warehouse':'data/warehouse/scientsbank.sqlite (locally rebuilt; not committed)','inputs_sha256':lineage,'source':'SciEntsBank official training Core only','transformations':['derive documented text features','preserve source text and labels','normalize question/answer/features/assignment tables','enforce relational constraints','materialize reconciled SQL report CSVs']},indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return report

if __name__=='__main__':build()
