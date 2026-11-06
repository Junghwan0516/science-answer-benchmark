"""Create versioned, question-grouped development splits from official training only."""
from pathlib import Path
from itertools import combinations
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
LABELS=['correct','partially_correct_incomplete','irrelevant','contradictory','non_domain']
PARTITIONS=['fit','development','calibration']
VERSION='question-grouped-v1'
SPLIT_DIR=ROOT/'data/splits/v1'
REPORT_DIR=ROOT/'reports/splits'

def load_training(root=ROOT):
    path=root/'data/processed/train.csv'
    data=pd.read_csv(path)
    if len(data)!=4969 or data.question_id.nunique()!=135:
        raise ValueError('Expected the 4,969-record, 135-question official training split.')
    if not data.official_split.eq('train').all():
        raise ValueError('Only official training records may enter development splitting.')
    if not data.sample_id.is_unique or set(data.label)!=set(LABELS):
        raise ValueError('Unexpected identifiers or labels.')
    return data

def make_assignments(data, seed=42, max_candidates=1000):
    """Accept the first candidate satisfying declared label-support and size constraints."""
    if not data.official_split.eq('train').all():
        raise ValueError('Official test records are prohibited.')
    questions=np.array(sorted(data.question_id.unique()))
    if len(questions)!=135:
        raise ValueError('Version v1 requires exactly 135 training questions.')
    rng=np.random.default_rng(seed)
    for candidate in range(1,max_candidates+1):
        permutation=rng.permutation(questions)
        qsets=[permutation[:95],permutation[95:115],permutation[115:]]
        pieces=[data[data.question_id.isin(qs)] for qs in qsets]
        enough_fit=pieces[0].label.value_counts().reindex(LABELS,fill_value=0).ge(10).all()
        eligible=all(
            piece.label.value_counts().reindex(LABELS,fill_value=0).ge(3).all()
            and .12<=len(piece)/len(data)<=.18
            for piece in pieces[1:]
        )
        if enough_fit and eligible:
            mapping={q:part for part,qs in zip(PARTITIONS,qsets) for q in qs}
            result=data[['sample_id','question_id','label','normalized_input_hash']].copy()
            result['internal_split']=result.question_id.map(mapping)
            multiplicity=data.groupby('normalized_input_hash').size()
            conflicts=data.groupby('normalized_input_hash').label.nunique().gt(1)
            result['input_multiplicity']=result.normalized_input_hash.map(multiplicity)
            result['has_conflicting_labels']=result.normalized_input_hash.map(conflicts)
            # Primary experiments retain unit weights. Alternative weights are fit-only.
            result['primary_sample_weight']=1.0
            result['sensitivity_fit_weight']=np.where(result.internal_split.eq('fit'),1/result.input_multiplicity,np.nan)
            return result.sort_values('sample_id').reset_index(drop=True), candidate
    raise ValueError('No candidate satisfied the predeclared constraints. Do not silently weaken them.')

def validate_assignments(data, assignments):
    if not assignments.sample_id.is_unique:
        raise ValueError('An answer was assigned more than once.')
    if len(assignments)!=len(data) or set(assignments.sample_id)!=set(data.sample_id):
        raise ValueError('Assignments must cover each official training answer exactly once.')
    if set(assignments.internal_split)!=set(PARTITIONS):
        raise ValueError('Unexpected internal split names.')
    original=data.set_index('sample_id')
    aligned=assignments.set_index('sample_id').loc[original.index]
    for column in ['question_id','label','normalized_input_hash']:
        if not aligned[column].equals(original[column]):
            raise ValueError(f'Source {column} must remain unchanged.')
    if assignments.groupby('question_id').internal_split.nunique().gt(1).any():
        raise ValueError('A question crosses internal partitions.')
    overlap=[]
    for left,right in combinations(PARTITIONS,2):
        a=assignments[assignments.internal_split.eq(left)]
        b=assignments[assignments.internal_split.eq(right)]
        record=dict(left_split=left,right_split=right,
                    shared_sample_ids=len(set(a.sample_id)&set(b.sample_id)),
                    shared_question_ids=len(set(a.question_id)&set(b.question_id)),
                    shared_normalized_inputs=len(set(a.normalized_input_hash)&set(b.normalized_input_hash)))
        if any(record[k] for k in ['shared_sample_ids','shared_question_ids','shared_normalized_inputs']):
            raise ValueError('Internal partition leakage detected.')
        overlap.append(record)
    return pd.DataFrame(overlap)

def build(root=ROOT):
    data=load_training(root)
    assignments,candidate=make_assignments(data)
    overlap=validate_assignments(data,assignments)
    splitdir=root/'data/splits/v1';reportdir=root/'reports/splits';figdir=reportdir/'figures'
    for p in [splitdir,reportdir,figdir]:p.mkdir(parents=True,exist_ok=True)
    assignments.to_csv(splitdir/'answer_assignments.csv',index=False)
    questions=assignments[['question_id','internal_split']].drop_duplicates().sort_values('question_id')
    questions=questions.merge(data.groupby('question_id').agg(domain=('domain','first'),answer_count=('sample_id','size')).reset_index(),on='question_id',validate='one_to_one')
    questions.to_csv(splitdir/'question_assignments.csv',index=False)
    counts=pd.crosstab(assignments.internal_split,assignments.label).reindex(index=PARTITIONS,columns=LABELS,fill_value=0)
    counts.to_csv(reportdir/'class_support.csv')
    summary=assignments.groupby('internal_split').agg(answer_count=('sample_id','size'),question_count=('question_id','nunique'),unique_inputs=('normalized_input_hash','nunique'),conflicting_rows=('has_conflicting_labels','sum')).reindex(PARTITIONS)
    summary['answer_percentage']=100*summary.answer_count/len(data)
    summary['question_percentage']=100*summary.question_count/135
    summary.to_csv(reportdir/'partition_summary.csv')
    overlap.to_csv(reportdir/'overlap_checks.csv',index=False)
    domains=pd.crosstab(questions.domain,questions.internal_split).reindex(columns=PARTITIONS,fill_value=0)
    domains.to_csv(reportdir/'domain_question_support.csv')
    hashes={}
    for p in [splitdir/'answer_assignments.csv',splitdir/'question_assignments.csv']:
        hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
    protocol=dict(version=VERSION,source_training_sha256=hashlib.sha256((root/'data/processed/train.csv').read_bytes()).hexdigest(),
        source_records=4969,source_questions=135,algorithm='PCG64 permutation of lexically sorted question IDs; accept first eligible candidate',
        seed=42,accepted_candidate=candidate,max_candidates=1000,
        question_counts=dict(fit=95,development=20,calibration=20),
        eligibility=dict(minimum_per_class_fit=10,minimum_per_class_development=3,minimum_per_class_calibration=3,development_and_calibration_answer_fraction=[.12,.18]),
        candidate_selection_inputs=['official training question IDs','official training label counts','official training record counts'],
        official_test_files_read=False,model_training_performed=False,
        primary_policy='Preserve all original rows and labels with unit sample weights; group whole questions.',
        sensitivity_policy='Optional future fit-only inverse complete-input multiplicity weights; never reweight development/calibration evaluation rows.',
        label_conflict_policy='Flag and retain; no majority-vote relabelling or deletion.',
        limitations=['Non-domain has only three records in development and calibration; class-specific estimates are unstable.',
                     'This is question-disjoint development, not domain-disjoint development or verified learner-disjoint evaluation.',
                     'Candidate acceptance uses training labels, so this is constrained allocation, not unconstrained random sampling.',
                     'Known official test inventory and input-overlap audits preceded this protocol; no model scores were used.'],
        assignment_sha256=hashes)
    (splitdir/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    # Figures are confined to the split report; the original 14 EDA figures are unchanged.
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10,4.4))
    colors=['#247881','#E8A847','#785CA6']
    for ax,column,title in zip(axes,['answer_count','question_count'],['Answer records','Whole questions']):
        ax.bar(PARTITIONS,summary[column],color=colors)
        for i,v in enumerate(summary[column]):ax.text(i,v+max(summary[column])*.025,f'{v:,}',ha='center')
        ax.set(title=title,ylabel='Count',ylim=(0,max(summary[column])*1.15))
    fig.suptitle('Question-grouped development partitions',fontweight='bold');fig.tight_layout()
    fig.savefig(figdir/'01_partition_sizes.png',dpi=150,bbox_inches='tight');plt.close(fig)
    for matrix,names,title,file in [(counts.T,['Correct','Partial / incomplete','Irrelevant','Contradictory','Non-domain'],'Every class is represented; rare-class supports remain small','02_class_support.png'),(domains,domains.index,'Training domains are unevenly represented across partitions','03_domain_question_support.png')]:
        fig,ax=plt.subplots(figsize=(8.5,4.8 if len(matrix)==5 else 6.2))
        im=ax.imshow(matrix.to_numpy(),cmap='Blues',aspect='auto',vmin=0)
        for i in range(len(matrix)):
            for j in range(3):
                v=int(matrix.iloc[i,j]);ax.text(j,i,str(v),ha='center',va='center',color='white' if v>matrix.to_numpy().max()*.6 else '#183748')
        ax.set_xticks(range(3),PARTITIONS);ax.set_yticks(range(len(matrix)),names);ax.set_title(title,fontsize=12,pad=16)
        fig.colorbar(im,ax=ax,label='Answer records' if len(matrix)==5 else 'Whole questions')
        fig.tight_layout();fig.savefig(figdir/file,dpi=150,bbox_inches='tight');plt.close(fig)
    print(summary.to_string())
    print('\nClass support:\n'+counts.to_string())
    print(f'\nAccepted candidate: {candidate}; seed: 42; no official test files read.')
    return protocol

def verify_split(root=ROOT):
    data=load_training(root)
    assignments=pd.read_csv(root/'data/splits/v1/answer_assignments.csv')
    overlap=validate_assignments(data,assignments)
    protocol=json.loads((root/'data/splits/v1/protocol.json').read_text())
    expected,candidate=make_assignments(data)
    pd.testing.assert_frame_equal(assignments,expected,check_dtype=False)
    assert candidate==protocol['accepted_candidate']==5
    assert protocol['source_training_sha256']==hashlib.sha256((root/'data/processed/train.csv').read_bytes()).hexdigest()
    for name,digest in protocol['assignment_sha256'].items():
        assert hashlib.sha256((root/'data/splits/v1'/name).read_bytes()).hexdigest()==digest
    assert assignments.groupby('question_id').internal_split.nunique().eq(1).all()
    assert (overlap.iloc[:,2:]==0).all().all()
    assert protocol['official_test_files_read'] is False and protocol['model_training_performed'] is False
    print('PASS: complete coverage, unchanged labels, reproducible assignments, zero question/input overlap, versioned hashes, and training-only split generation.')

if __name__=='__main__':
    build()
    verify_split()
