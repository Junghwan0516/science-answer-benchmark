"""Training-only feature audit and question-cluster resampling diagnostics."""
from pathlib import Path
import hashlib,json,re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from build_splits import load_training,validate_assignments,LABELS
ROOT=Path(__file__).resolve().parents[1]
NAMES=['Correct','Partial / incomplete','Irrelevant','Contradictory','Non-domain']
COLORS=['#247881','#E8A847','#7F8C9A','#BD5465','#785CA6']
SEED=2026
REPLICATES=2000

def word_tokens(s):return re.findall(r"[a-z]+(?:'[a-z]+)?",s.casefold())
def jaccard(a,b):
    a,b=set(a),set(b)
    return len(a&b)/len(a|b) if a|b else 0.0

def make_features(data):
    rows=[]
    for r in data.itertuples():
        a=word_tokens(r.student_answer);q=word_tokens(r.question_text);ref=word_tokens(r.reference_answer)
        chars=len(r.student_answer)
        raw=json.dumps([r.question_text,r.reference_answer,r.student_answer],ensure_ascii=False)
        rows.append(dict(sample_id=r.sample_id,answer_char_count=chars,answer_word_count=len(r.student_answer.split()),lexical_token_count=len(a),unique_word_count=len(set(a)),type_token_ratio=len(set(a))/len(a) if a else 0.0,question_token_jaccard=jaccard(a,q),reference_token_jaccard=jaccard(a,ref),reference_length_ratio=len(r.student_answer.split())/len(r.reference_answer.split()),punctuation_fraction=sum(not c.isalnum() and not c.isspace() for c in r.student_answer)/chars if chars else 0.0,has_numeric_content=bool(re.search(r'\d',r.student_answer)),raw_input_hash=hashlib.sha256(raw.encode()).hexdigest()))
    return pd.DataFrame(rows)

def cluster_label_intervals(data,replicates=REPLICATES,seed=SEED):
    matrix=pd.crosstab(data.question_id,data.label).reindex(columns=LABELS,fill_value=0).to_numpy()
    rng=np.random.default_rng(seed)
    draws=rng.integers(0,len(matrix),size=(replicates,len(matrix)))
    sums=matrix[draws].sum(axis=1)
    proportions=sums/sums.sum(axis=1,keepdims=True)
    points=matrix.sum(axis=0)/matrix.sum()
    # Independent-row resampling is a diagnostic comparison, not the primary interval.
    naive=rng.multinomial(len(data),points,size=replicates)/len(data)
    return pd.DataFrame(dict(label=LABELS,answer_share=points,question_cluster_low=np.quantile(proportions,.025,axis=0),question_cluster_high=np.quantile(proportions,.975,axis=0),row_resampling_low=np.quantile(naive,.025,axis=0),row_resampling_high=np.quantile(naive,.975,axis=0),answer_support=matrix.sum(axis=0),question_support=(matrix>0).sum(axis=0)))

def build(root=ROOT):
    data=load_training(root)
    assignment=pd.read_csv(root/'data/splits/v1/answer_assignments.csv')
    validate_assignments(data,assignment)
    feature=make_features(data)
    feature.to_csv(root/'data/processed/training_features.csv',index=False)
    d=data.merge(feature.drop(columns=['answer_word_count','reference_token_jaccard']),on='sample_id',validate='one_to_one').merge(assignment[['sample_id','internal_split']],on='sample_id',validate='one_to_one')
    out=root/'reports/advanced';table=out/'tables';figdir=out/'figures'
    for p in [table,figdir]:p.mkdir(parents=True,exist_ok=True)
    def save(fig,name):
        fig.savefig(figdir/name,dpi=150,bbox_inches='tight',facecolor='white');plt.close(fig)
    plt.rcdefaults()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    intervals=cluster_label_intervals(data)
    intervals.to_csv(table/'label_share_resampling.csv',index=False)
    fig,ax=plt.subplots(figsize=(10,5));y=np.arange(5)
    for offset,lo,hi,color,name in [(-.11,'row_resampling_low','row_resampling_high','#A1ABB3','Independent-row resampling (comparison)'),(.11,'question_cluster_low','question_cluster_high','#247881','Whole-question resampling')]:
        for i,r in intervals.iterrows():ax.plot([100*r[lo],100*r[hi]],[i+offset]*2,color=color,lw=3)
        ax.scatter(100*intervals.answer_share,y+offset,color=color,s=25,label=name,zorder=3)
    ax.set_yticks(y,NAMES);ax.invert_yaxis();ax.set(xlabel='Observed answer share (%)',title='Uncertainty depends on the resampling unit');ax.legend(fontsize=8,loc='lower right');ax.grid(axis='x',alpha=.15);save(fig,'01_cluster_resampling.png')
    domain_rows=[]
    for i,(domain,g) in enumerate(data.groupby('domain')):
        c=g.assign(is_correct=g.label.eq('correct').astype(int)).groupby('question_id').agg(n=('sample_id','size'),correct=('is_correct','sum'))
        rng=np.random.default_rng(SEED+i);draws=rng.integers(0,len(c),size=(REPLICATES,len(c)));sums=c.to_numpy()[draws].sum(axis=1)
        rates=sums[:,1]/sums[:,0]
        domain_rows.append(dict(domain=domain,answers=len(g),questions=len(c),correct_share=g.label.eq('correct').mean(),cluster_low=np.quantile(rates,.025),cluster_high=np.quantile(rates,.975),interval_displayed=len(c)>=5))
    domains=pd.DataFrame(domain_rows).sort_values('correct_share')
    domains.to_csv(table/'domain_correct_share.csv',index=False)
    fig,ax=plt.subplots(figsize=(9,5.8))
    for i,r in enumerate(domains.itertuples()):
        if r.interval_displayed:ax.plot([100*r.cluster_low,100*r.cluster_high],[i,i],color='#247881',lw=2)
        ax.scatter(100*r.correct_share,i,color='#247881' if r.interval_displayed else '#999999',s=40)
    ax.set_yticks(range(len(domains)),[f'{r.domain} (questions={r.questions}, answers={r.answers})' for r in domains.itertuples()]);ax.set(xlabel='Answers labelled correct (%)',title='Domain composition with within-domain question resampling');ax.text(.01,-.14,'Intervals hidden when fewer than five questions are available.',transform=ax.transAxes,fontsize=9);save(fig,'02_domain_uncertainty.png')
    def composition(group):return pd.crosstab(data[group],data.label).reindex(columns=LABELS,fill_value=0).pipe(lambda x:x.div(x.sum(axis=1),axis=0)).mean()
    weights=pd.DataFrame({'answer_weighted':data.label.value_counts(normalize=True).reindex(LABELS),'question_weighted':composition('question_id'),'domain_weighted':composition('domain')})
    weights.to_csv(table/'aggregation_weights.csv')
    fig,ax=plt.subplots(figsize=(10,4.8));x=np.arange(5)
    for i,(column,color) in enumerate(zip(weights.columns,['#247881','#E8A847','#785CA6'])):ax.bar(x+(i-1)*.25,weights[column]*100,.25,label=column.replace('_',' '),color=color)
    ax.set_xticks(x,NAMES,rotation=12);ax.set(ylabel='Label share (%)',title='Answers, questions, and domains define different summaries');ax.legend(fontsize=9);save(fig,'03_aggregation_levels.png')
    short=[]
    rng=np.random.default_rng(SEED)
    for label,g in d.groupby('label'):
        q=g.assign(short=g.answer_word_count.le(5).astype(int)).groupby('question_id').agg(n=('sample_id','size'),short=('short','sum'))
        draw=rng.integers(0,len(q),size=(REPLICATES,len(q)));sums=q.to_numpy()[draw].sum(axis=1);rates=sums[:,1]/sums[:,0]
        short.append(dict(label=label,answers=len(g),questions=len(q),short_share=g.answer_word_count.le(5).mean(),cluster_low=np.quantile(rates,.025),cluster_high=np.quantile(rates,.975)))
    short=pd.DataFrame(short).set_index('label').reindex(LABELS)
    short.to_csv(table/'short_answer_slices.csv')
    fig,ax=plt.subplots(figsize=(9,4.6))
    for i,r in enumerate(short.itertuples()):ax.plot([100*r.cluster_low,100*r.cluster_high],[i,i],color=COLORS[i],lw=3);ax.scatter(100*r.short_share,i,color=COLORS[i],s=45)
    ax.set_yticks(range(5),[f'{n} (n={v})' for n,v in zip(NAMES,short.answers)]);ax.invert_yaxis();ax.set(xlabel='Answers with at most five whitespace-separated words (%)',title='A transparent short-answer slice, not a correctness rule');save(fig,'04_short_answer_slices.png')
    fig,axes=plt.subplots(1,5,figsize=(14,3.8),sharex=True,sharey=True)
    for ax,label,name,color in zip(axes,LABELS,NAMES,COLORS):
        g=d[d.label.eq(label)];ax.scatter(g.question_token_jaccard,g.reference_token_jaccard,s=8,alpha=.25 if len(g)>100 else .7,color=color);ax.plot([0,1],[0,1],ls=':',color='#999999',lw=1);ax.set(title=name,xlim=(0,1),ylim=(0,1),xlabel='Question overlap')
    axes[0].set_ylabel('Reference overlap');fig.suptitle('Question wording and reference wording are distinct signals',fontweight='bold');fig.tight_layout();save(fig,'05_question_reference_overlap.png')
    cols=['answer_char_count','answer_word_count','type_token_ratio','question_token_jaccard','reference_token_jaccard','reference_length_ratio']
    corr=d[cols].corr(method='spearman');corr.to_csv(table/'feature_spearman_correlation.csv')
    fig,ax=plt.subplots(figsize=(9.5,7));im=ax.imshow(corr,vmin=-1,vmax=1,cmap='RdBu_r')
    shortnames=['Characters','Words','Type-token ratio','Question overlap','Reference overlap','Answer / reference length']
    ax.set_xticks(range(6),shortnames,rotation=35,ha='right');ax.set_yticks(range(6),shortnames)
    for i in range(6):
        for j in range(6):ax.text(j,i,f'{corr.iloc[i,j]:.2f}',ha='center',va='center',color='white' if abs(corr.iloc[i,j])>.65 else '#222222')
    ax.set_title('Feature redundancy: descriptive Spearman correlations');fig.colorbar(im,ax=ax,label='Spearman correlation');fig.tight_layout();save(fig,'06_feature_redundancy.png')
    fit=d[d.internal_split.eq('fit')];dev=d[d.internal_split.eq('development')].copy()
    vocab=set(t for a in fit.student_answer for t in word_tokens(a))
    dev_tokens=[word_tokens(s) for s in dev.student_answer]
    dev['lexical_tokens']=[len(t) for t in dev_tokens];dev['unseen_tokens']=[sum(t not in vocab for t in a) for a in dev_tokens]
    novelty=dev.groupby('domain').agg(answers=('sample_id','size'),tokens=('lexical_tokens','sum'),unseen_tokens=('unseen_tokens','sum'))
    novelty['unseen_token_percentage']=100*novelty.unseen_tokens/novelty.tokens
    novelty=novelty.sort_values('unseen_token_percentage',ascending=False);novelty.to_csv(table/'development_vocabulary_novelty.csv')
    fig,ax=plt.subplots(figsize=(9,4.8));ax.barh(novelty.index,novelty.unseen_token_percentage,color='#247881');ax.invert_yaxis();ax.set(xlabel='Development token occurrences absent from fit vocabulary (%)',ylabel='Source domain code',title='Vocabulary novelty on development only');save(fig,'07_development_vocabulary_novelty.png')
    raw_counts=d.groupby(['raw_input_hash','label']).size().unstack(fill_value=0)
    unavoidable=int((raw_counts.sum(axis=1)-raw_counts.max(axis=1)).sum())
    ambiguity=dict(raw_input_groups=len(raw_counts),conflicting_raw_input_groups=int(raw_counts.gt(0).sum(axis=1).gt(1).sum()),minimum_training_label_disagreements=unavoidable,training_records=len(d),minimum_training_disagreement_percentage=100*unavoidable/len(d),interpretation='Empirical constraint for deterministic predictions on exactly identical raw question/reference/answer triples; not population Bayes error or test performance.')
    (table/'raw_input_ambiguity.json').write_text(json.dumps(ambiguity,indent=2)+'\n')
    manifest=dict(training_only=True,bootstrap_seed=SEED,bootstrap_replicates=REPLICATES,bootstrap_unit='question_id',feature_rows=len(feature),feature_columns=len(feature.columns),fit_vocabulary_types=len(vocab),development_tokens=int(novelty.tokens.sum()),development_unseen_tokens=int(novelty.unseen_tokens.sum()),calibration_vocabulary_analyzed=False,official_test_files_read=False,model_training_performed=False,source_sha256=hashlib.sha256((root/'data/processed/train.csv').read_bytes()).hexdigest(),assignment_sha256=hashlib.sha256((root/'data/splits/v1/answer_assignments.csv').read_bytes()).hexdigest(),raw_input_ambiguity=ambiguity)
    (out/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))
    return manifest

if __name__=='__main__':build()
