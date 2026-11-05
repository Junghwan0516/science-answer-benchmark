"""Rebuild the SciEntsBank data audit and descriptive figures; no model training."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import re
import unicodedata
import xml.etree.ElementTree as ET
from zipfile import ZipFile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
LABELS = ['correct', 'partially_correct_incomplete', 'irrelevant', 'contradictory', 'non_domain']
NAMES = ['Correct', 'Partial / incomplete', 'Irrelevant', 'Contradictory', 'Non-domain']
COLORS = ['#247881', '#E8A847', '#7F8C9A', '#BD5465', '#785CA6']
SPLITS = {'train':4969, 'test-unseen-answers':540, 'test-unseen-questions':733, 'test-unseen-domains':4562}
FIG = ROOT/'reports/figures'
TAB = ROOT/'reports/tables'

def normalize(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text).casefold()).strip()

def tokens(text):
    return set(re.findall(r"[a-z]+(?:'[a-z]+)?", text.casefold()))

def load_data():
    rows=[]
    with ZipFile(ROOT/'data/raw/scientsbank_core.zip') as archive:
        for name in sorted(archive.namelist()):
            parts=name.split('/')
            if len(parts)!=4 or parts[0]!='sciEntsBank' or parts[2]!='Core' or parts[1] not in SPLITS or not name.endswith('.xml'):
                continue
            root=ET.fromstring(archive.read(name))
            question=root.findtext('questionText', default='').strip()
            refs=[''.join(e.itertext()).strip() for e in root.findall('./referenceAnswers/referenceAnswer')]
            assert len(refs)==1, 'This release is expected to have one reference per question.'
            for answer in root.findall('./studentAnswers/studentAnswer'):
                text=''.join(answer.itertext()).strip()
                context=json.dumps([normalize(question),normalize(refs[0]),normalize(text)], ensure_ascii=False)
                rows.append(dict(official_split=parts[1],question_id=root.get('id'),domain=root.get('module'),answer_id=answer.get('id'),source_file=name,question_text=question,reference_answer=refs[0],student_answer=text,label=answer.get('accuracy'),normalized_input_hash=hashlib.sha256(context.encode()).hexdigest()))
    data=pd.DataFrame(rows)
    data['sample_id']=data['official_split']+':'+data['answer_id']
    data['answer_word_count']=data.student_answer.str.split().str.len()
    data['reference_word_count']=data.reference_answer.str.split().str.len()
    data['question_word_count']=data.question_text.str.split().str.len()
    data['reference_token_jaccard']=[len(tokens(a)&tokens(r))/len(tokens(a)|tokens(r)) if tokens(a)|tokens(r) else 0 for a,r in zip(data.student_answer,data.reference_answer)]
    # A transparent lexical indicator, not a semantic contradiction detector.
    data['has_negation']=data.student_answer.str.contains(r"\b(?:not|no|never|cannot|can't|don't|doesn't|isn't|aren't|won't)\b",case=False,regex=True)
    assert data.sample_id.is_unique
    assert set(data.label)==set(LABELS)
    for split,count in SPLITS.items():
        assert len(data[data.official_split.eq(split)])==count
    return data

def save_table(df,name,index=False):
    df.to_csv(TAB/f'{name}.csv',index=index)

def save(fig,name):
    fig.savefig(FIG/f'{name}.png',dpi=170,bbox_inches='tight',facecolor='white')
    plt.close(fig)

def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':14,'axes.titleweight':'bold','axes.spines.top':False,'axes.spines.right':False,'axes.grid':False,'figure.facecolor':'white'})

def build():
    for p in [FIG,TAB,ROOT/'data/processed']:p.mkdir(parents=True,exist_ok=True)
    style()
    data=load_data()
    train=data[data.official_split.eq('train')].copy()
    # Export all official splits separately; descriptive analysis below uses training only.
    for split in SPLITS:
        data[data.official_split.eq(split)].to_csv(ROOT/f'data/processed/{split}.csv',index=False)
    manifest=[]
    for split in SPLITS:
        d=data[data.official_split.eq(split)]
        manifest.append(dict(official_split=split,answers=len(d),questions=d.question_id.nunique(),domains=d.domain.nunique(),unique_normalized_inputs=d.normalized_input_hash.nunique()))
    save_table(pd.DataFrame(manifest),'split_manifest')
    labels=train.label.value_counts().reindex(LABELS,fill_value=0).rename_axis('label').reset_index(name='answer_count')
    labels['percentage']=100*labels.answer_count/len(train)
    save_table(labels,'label_distribution')
    counts=pd.crosstab(train.domain,train.label).reindex(columns=LABELS,fill_value=0)
    shares=counts.div(counts.sum(axis=1),axis=0)*100
    save_table(counts,'domain_label_counts',True);save_table(shares,'domain_label_percentages',True)
    domain=train.groupby('domain').agg(total_answers=('answer_id','size'),total_questions=('question_id','nunique'))
    domain['non_domain_count']=counts.non_domain
    domain['non_domain_percentage']=100*domain.non_domain_count/domain.total_answers
    save_table(domain,'minority_domain_distribution',True)
    minority=train[train.label.eq('non_domain')]
    minority_q=minority.groupby(['domain','question_id']).size().rename('non_domain_count').reset_index()
    save_table(minority_q,'minority_question_distribution')
    minority.to_csv(TAB/'minority_answers.csv',index=False)
    numeric=['answer_word_count','reference_token_jaccard']
    save_table(train.groupby('label')[numeric].describe().reindex(LABELS),'text_feature_summary',True)
    qcounts=pd.crosstab(train.question_id,train.label).reindex(columns=LABELS,fill_value=0)
    probs=qcounts.div(qcounts.sum(axis=1),axis=0)
    entropy=-(probs.where(probs.gt(0))*np.log2(probs.where(probs.gt(0)))).sum(axis=1)
    questions=train.groupby('question_id').agg(domain=('domain','first'),answer_count=('answer_id','size'))
    questions['label_entropy_bits']=entropy
    questions['correct_share']=probs.correct
    save_table(questions,'question_summary',True)
    groups=train.groupby('normalized_input_hash').agg(row_count=('answer_id','size'),label_count=('label','nunique'),question_id=('question_id','first'))
    duplicates=groups[groups.row_count.gt(1)]
    conflict_keys=groups.index[groups.label_count.gt(1)]
    conflicts=train[train.normalized_input_hash.isin(conflict_keys)].sort_values(['normalized_input_hash','label','answer_id'])
    save_table(duplicates.reset_index(),'duplicate_groups');save_table(conflicts,'conflicting_labels')
    train_keys=set(train.normalized_input_hash)
    overlap=[]
    for split in list(SPLITS)[1:]:
        d=data[data.official_split.eq(split)]
        overlap.append(dict(official_split=split,answers=len(d),matching_training_input_rows=int(d.normalized_input_hash.isin(train_keys).sum()),shared_question_ids=len(set(d.question_id)&set(train.question_id))))
    save_table(pd.DataFrame(overlap),'train_test_overlap')
    blanks=[]
    for col in ['question_id','answer_id','domain','question_text','reference_answer','student_answer','label']:
        blanks.append(dict(column=col,null_count=int(train[col].isna().sum()),blank_count=int(train[col].fillna('').astype(str).str.strip().eq('').sum())))
    save_table(pd.DataFrame(blanks),'missing_values')
    audit=dict(training_answers=len(train),training_questions=train.question_id.nunique(),training_domains=train.domain.nunique(),minority_answers=len(minority),minority_questions=minority.question_id.nunique(),minority_domains=minority.domain.nunique(),unique_normalized_training_inputs=len(groups),duplicate_groups=len(duplicates),duplicate_excess_rows=int((duplicates.row_count-1).sum()),conflicting_input_groups=len(conflict_keys),conflicting_rows=len(conflicts),train_ua_overlap_rows=overlap[0]['matching_training_input_rows'],blank_fields=sum(x['blank_count'] for x in blanks),median_answer_words=float(train.answer_word_count.median()),source_sha256=hashlib.sha256((ROOT/'source.zip').read_bytes()).hexdigest() if (ROOT/'source.zip').exists() else json.loads((ROOT/'data/provenance.json').read_text())['source_sha256'])
    (TAB/'audit_summary.json').write_text(json.dumps(audit,indent=2)+'\n')
    # 01. Class counts; a linear axis preserves the size of the imbalance.
    fig,ax=plt.subplots(figsize=(10,4.4));y=np.arange(5)
    ax.barh(y,labels.answer_count,color=COLORS);ax.set_yticks(y,NAMES);ax.invert_yaxis()
    for i,r in labels.iterrows():ax.text(r.answer_count+22,i,f'{r.answer_count:,} ({r.percentage:.2f}%)',va='center')
    ax.set(xlim=(0,2500),xlabel='Student answer records',title='Training answers are strongly imbalanced')
    save(fig,'01_class_distribution')
    # 02-03. Domain counts and within-domain shares have different denominators.
    for values,fmt,name,title in [(counts,'d','02_domain_label_counts','Answer counts by domain and label'),(shares,'.1f','03_domain_label_shares','Within-domain label composition (%)')]:
        fig,ax=plt.subplots(figsize=(9,6));im=ax.imshow(values,cmap='Blues' if fmt=='d' else 'YlGnBu',aspect='auto',vmin=0)
        for i in range(values.shape[0]):
            for j in range(5):
                v=values.iloc[i,j];ax.text(j,i,format(v,fmt),ha='center',va='center',color='white' if v>values.to_numpy().max()*.6 else '#193247',fontsize=9)
        ax.set_xticks(range(5),NAMES,rotation=22,ha='right');ax.set_yticks(range(len(values)),values.index);ax.set(ylabel='Source domain code',title=title);fig.colorbar(im,ax=ax,shrink=.8,label='Answer count' if fmt=='d' else '% of answers in domain');save(fig,name)
    # 04. ECDF retains the full length distribution.
    fig,ax=plt.subplots(figsize=(9,5))
    for label,name,color in zip(LABELS,NAMES,COLORS):
        vals=np.sort(train.loc[train.label.eq(label),'answer_word_count']);ax.step(vals,np.arange(1,len(vals)+1)/len(vals),where='post',label=f'{name} (n={len(vals):,})',color=color)
    ax.set(xlabel='Answer length (whitespace-separated words)',ylabel='Cumulative share',title='How answer length varies across labels',ylim=(0,1.02));ax.legend(fontsize=9);ax.grid(alpha=.15);save(fig,'04_answer_length_ecdf')
    # 05. Boxplots: outliers remain included in summaries and the ECDF.
    fig,ax=plt.subplots(figsize=(10,4.8));ax.boxplot([train.loc[train.label.eq(l),'answer_word_count'] for l in LABELS],tick_labels=NAMES,showfliers=True,flierprops={'markersize':2,'alpha':.3});ax.tick_params(axis='x',rotation=15);ax.set(ylabel='Answer length (words)',title='Length alone does not define an answer label');save(fig,'05_answer_length_boxplot')
    # 06. Jaccard uses lower-case word sets, keeps stop words, and has no fitted vocabulary.
    fig,ax=plt.subplots(figsize=(10,4.8));ax.boxplot([train.loc[train.label.eq(l),'reference_token_jaccard'] for l in LABELS],tick_labels=NAMES,showfliers=False);ax.tick_params(axis='x',rotation=15);ax.set(ylabel='Word-set Jaccard similarity',ylim=(0,1.02),title='Lexical overlap with the reference answer');save(fig,'06_reference_overlap')
    # 07. Unequal answer counts mean some questions dominate row-weighted statistics.
    ranked=questions.sort_values('answer_count',ascending=False)
    fig,ax=plt.subplots(figsize=(9,4.5));ax.plot(np.arange(1,len(ranked)+1),ranked.answer_count,color=COLORS[0]);ax.set(xlabel='Question rank by answer count',ylabel='Answer records',title='Questions contribute unequal numbers of answers');ax.grid(alpha=.15);save(fig,'07_question_sample_sizes')
    # 08. Each point represents a question, not a learner.
    fig,ax=plt.subplots(figsize=(8.5,5));sc=ax.scatter(questions.answer_count,questions.label_entropy_bits,c=questions.correct_share,cmap='viridis',s=40,alpha=.8);ax.set(xlabel='Answer records per question',ylabel='Label entropy (bits)',title='Label diversity varies across questions');fig.colorbar(sc,ax=ax,label='Share labelled correct');save(fig,'08_question_label_diversity')
    # 09. Counts versus rates for the rare class.
    d=domain.sort_values('non_domain_count',ascending=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4.6),sharey=True);yy=np.arange(len(d))
    axes[0].barh(yy,d.non_domain_count,color=COLORS[4]);axes[0].set_yticks(yy,d.index);axes[0].invert_yaxis();axes[0].set(xlabel='Non-domain answer count',ylabel='Source domain code')
    axes[1].barh(yy,d.non_domain_percentage,color=COLORS[4]);axes[1].set(xlabel='Non-domain share within domain (%)');fig.suptitle('Rare-class counts and rates answer different questions',fontweight='bold');fig.tight_layout();save(fig,'09_minority_domain_distribution')
    # 10. Conservative input duplicate normalization includes question and reference.
    fig,axes=plt.subplots(1,2,figsize=(10,4.5));size_counts=groups.row_count.value_counts().sort_index();axes[0].bar(size_counts.index.astype(str),size_counts.values,color=COLORS[0]);axes[0].set(xlabel='Records per normalized input group',ylabel='Number of groups',title='Input multiplicity (log y-axis)');axes[0].set_yscale('log')
    axes[1].bar(['Unique inputs','Repeated excess rows'],[len(groups),audit['duplicate_excess_rows']],color=[COLORS[0],COLORS[1]]);axes[1].set(ylabel='Count',title='Rows and inputs are different units');axes[1].tick_params(axis='x',rotation=12);fig.tight_layout();save(fig,'10_duplicate_structure')
    # 11. Show every conflicting group, keeping original labels unchanged.
    ctab=pd.crosstab(conflicts.normalized_input_hash,conflicts.label).reindex(columns=LABELS,fill_value=0)
    ctab.index=[f'Group {i+1}' for i in range(len(ctab))];save_table(ctab,'conflict_group_label_counts',True)
    fig,ax=plt.subplots(figsize=(9,4.6));left=np.zeros(len(ctab))
    for lab,name,color in zip(LABELS,NAMES,COLORS):ax.barh(ctab.index,ctab[lab],left=left,label=name,color=color);left+=ctab[lab].to_numpy()
    ax.set(xlabel='Answer records',title='Identical normalized inputs can have conflicting labels');ax.legend(loc='upper right',fontsize=8);save(fig,'11_label_conflicts')
    # 12. A boolean lexical marker cannot establish scientific contradiction.
    neg=pd.crosstab(train.label,train.has_negation).reindex(index=LABELS,columns=[False,True],fill_value=0)
    neg['marker_percentage']=100*neg[True]/neg.sum(axis=1);save_table(neg,'negation_marker_summary',True)
    fig,ax=plt.subplots(figsize=(9,4.4));ax.barh(NAMES,neg.marker_percentage,color=COLORS);ax.invert_yaxis();ax.set(xlabel='Answers containing a listed negation marker (%)',title='Negation markers occur in multiple answer classes');save(fig,'12_negation_markers')
    # 13. Question-normalized view gives each question equal weight.
    pooled=labels.set_index('label').percentage.reindex(LABELS)
    equalq=probs.mean(axis=0).reindex(LABELS)*100
    weighting=pd.DataFrame({'answer_weighted_percentage':pooled,'question_weighted_percentage':equalq});save_table(weighting,'weighting_comparison',True)
    fig,ax=plt.subplots(figsize=(10,4.6));xx=np.arange(5);ax.bar(xx-.18,pooled,.36,label='Each answer has equal weight',color=COLORS[0]);ax.bar(xx+.18,equalq,.36,label='Each question has equal weight',color=COLORS[1]);ax.set_xticks(xx,NAMES,rotation=15);ax.set(ylabel='Label share (%)',title='Aggregation weights can change the picture');ax.legend(fontsize=9);save(fig,'13_answer_vs_question_weighting')
    # 14. Domain-reference overlap is descriptive, with original source codes retained.
    fig,ax=plt.subplots(figsize=(10,5));domain_order=sorted(train.domain.unique());ax.boxplot([train.loc[train.domain.eq(d),'reference_token_jaccard'] for d in domain_order],tick_labels=domain_order,showfliers=False);ax.set(xlabel='Source domain code',ylabel='Word-set Jaccard similarity',ylim=(0,1.02),title='Reference-answer overlap differs across domains');save(fig,'14_domain_reference_overlap')
    print(json.dumps(audit,indent=2))
    return audit

if __name__=='__main__':build()
