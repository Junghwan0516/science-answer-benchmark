"""Execute the statistics and SQL notebooks using Python and raw-MIME capture."""
from pathlib import Path
from contextlib import redirect_stdout,redirect_stderr
import io,json,os
from prepare_split_notebook import markdown as md,code
ROOT=Path(__file__).resolve().parents[1]
SETUP='''from pathlib import Path
import subprocess, sys, json, base64
import pandas as pd
ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / "data/processed/train.csv").exists()), None)
if ROOT is None:
    checkout = Path("/content/scientsbank_portfolio")
    if not checkout.exists():
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/Junghwan0516/science-answer-benchmark.git",
                        str(checkout)], check=True)
    ROOT = checkout
sys.path.insert(0, str(ROOT / "scripts"))
print("Project:", ROOT.name)'''
FIGURES=[('01_cluster_resampling.png','Question-cluster resampling','Resample whole questions, preserving the dependence among their answers. The independent-row comparison uses multinomial class-count draws. Both use 2,000 replicates and empirical percentile bounds; these are resampling-stability summaries conditional on the observed corpus, not a guarantee for future learners.'),('02_domain_uncertainty.png','Domain composition and uncertainty','Each domain is resampled by its own questions. Intervals are hidden for domains with fewer than five questions. The observed correct share describes annotations; it is not an intrinsic difficulty or ability estimate.'),('03_aggregation_levels.png','Three aggregation targets','Pooling answers, averaging question-level shares, and averaging domain-level shares define three different estimands. These are explicit weighting choices, not interchangeable estimates.'),('04_short_answer_slices.png','Short-answer slice','The cutoff of at most five whitespace-separated words is exploratory. Within each label, contributing questions are resampled. Small class supports remain a limitation. Length is not a correctness rule.'),('05_question_reference_overlap.png','Question versus reference overlap','Word-set Jaccard overlap is calculated separately against question wording and reference wording. Points are descriptive observations, not a fitted classifier or a causal relationship.'),('06_feature_redundancy.png','Feature engineering and redundancy','Spearman correlations summarize monotone associations among numeric features; class labels are not encoded as an ordinal numeric variable. Character count and word count are mechanically related. No independence-based p-values are reported.'),('07_development_vocabulary_novelty.png','Fit-to-development vocabulary novelty','Build the vocabulary from fit answers only. Count development token occurrences absent from that vocabulary. The calibration partition and official test sets are excluded from this vocabulary analysis.')]

def execute(name,cells):
    namespace={'__name__':'__main__'};old=Path.cwd();os.chdir(ROOT)
    try:
        count=0
        for cell in cells:
            if cell['cell_type']!='code':continue
            count+=1;outputs=[]
            def display(data,raw=False):
                if not raw or not isinstance(data,dict):raise ValueError('Only raw MIME display is used.')
                outputs.append(dict(output_type='display_data',data=data,metadata={}))
            namespace['display']=display;stdout=io.StringIO();stderr=io.StringIO()
            with redirect_stdout(stdout),redirect_stderr(stderr):exec(compile(''.join(cell['source']),f'{name}-cell-{count}','exec'),namespace)
            streams=[]
            for stream,value in [('stdout',stdout.getvalue()),('stderr',stderr.getvalue())]:
                if value:streams.append(dict(output_type='stream',name=stream,text=value.splitlines(keepends=True)))
            cell['outputs']=streams+outputs;cell['execution_count']=count
    finally:os.chdir(old)
    for i,c in enumerate(cells):c['id']=f'{name}-{i:02}'
    nb=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'execution_note':'Executed sequentially with Python stdout and raw-MIME capture. Outputs are computed.'})
    (ROOT/f'notebooks/{name}.ipynb').write_text(json.dumps(nb,indent=1,ensure_ascii=False)+'\n')
    print('Executed',name,count,'code cells')

def prepare():
    cells=[md('# Statistical Analysis and Feature Audit\n\nTraining-only descriptive analysis with question-level resampling, explicit weighting targets, and reproducible text features. No model is fitted. Data source: SciEntsBank / Dzikovska et al. (2013), https://aclanthology.org/S13-2045/. Data-derived material: CC BY-SA 3.0.'),code(SETUP),code('from analysis_extensions import build\nif not (ROOT / "reports/advanced/analysis_manifest.json").exists():\n    build(ROOT)\nmanifest = json.loads((ROOT / "reports/advanced/analysis_manifest.json").read_text())\nprint(json.dumps(manifest, indent=2))')]
    for filename,title,description in FIGURES:
        cells.extend([md('## '+title+'\n\n'+description),code('image_path = ROOT / "reports/advanced/figures/'+filename+'"\ndisplay({"image/png": base64.b64encode(image_path.read_bytes()).decode("ascii")}, raw=True)')])
    cells.extend([md('## Exact-input annotation ambiguity\n\nExactly identical raw question/reference/answer triples can have competing labels. The sum of group size minus the largest label count is an empirical minimum number of training-label disagreements for a deterministic classifier restricted to these raw inputs. It is not population Bayes error or a bound on test accuracy.'),code('print(json.dumps(manifest["raw_input_ambiguity"], indent=2))\nprint(pd.read_csv(ROOT / "reports/advanced/tables/label_share_resampling.csv").to_string(index=False))'),md('## Limits and follow-up\n\nThe training corpus, including its previously audited development/calibration records, is used for descriptive statistics. No calibration/model scores are inspected. The vocabulary novelty diagnostic alone restricts vocabulary construction to fit and evaluation to development. All original split assignments remain unchanged. Resampling does not resolve source selection bias, label uncertainty, or generalization beyond the observed questions and domains. Next: build models using the frozen development protocol, not by selecting thresholds from these exploratory slices.')])
    execute('01b_statistical_analysis',cells)
    engineering=[md('# Data Contracts and SQL Warehouse\n\nBuild a local relational warehouse, enforce integrity constraints, reconcile SQL and Pandas counts, and record lineage. The SQLite binary is generated locally and ignored by Git; SQL, contracts, portable CSV reports, and this executed notebook are committed.'),code(SETUP),code('from build_warehouse import build\nquality = build(ROOT)\nprint("Quality checks:", len(quality["checks"]))'),md('## Relational tables\n\nDomains and labels are dimensions. Questions contain one context/reference. Answers link to questions and labels. Feature rows and development assignments link one-to-one to answers. Primary keys, foreign keys, ranges, text non-emptiness, and split-weight checks are enforced by SQLite.'),code('import sqlite3\nconnection = sqlite3.connect(ROOT / "data/warehouse/scientsbank.sqlite")\nprint(pd.read_sql_query("SELECT name, type FROM sqlite_master WHERE type IN (\'table\', \'view\') ORDER BY type, name", connection).to_string(index=False))'),md('## SQL: domain-level composition\n\nThis query joins question, answer, and feature tables through a documented view. It returns observed label shares, not model performance.'),code('query = "SELECT * FROM v_domain_summary ORDER BY answer_count DESC"\nprint(query)\nprint(pd.read_sql_query(query, connection).to_string(index=False))'),md('## SQL: split coverage and rare labels'),code('print(pd.read_sql_query("SELECT * FROM v_partition_summary ORDER BY internal_split", connection).to_string(index=False))\nprint("Foreign-key violations:", connection.execute("PRAGMA foreign_key_check").fetchall())\nprint("Integrity:", connection.execute("PRAGMA integrity_check").fetchone()[0])\nconnection.close()'),md('## Lineage and reproducibility\n\nSHA-256 hashes pin the source CSV, derived features, assignments, schema, and query definitions. The builder creates a temporary database, verifies it, and atomically replaces the local warehouse only on success. Unknown labels, orphan records, missing feature rows, and invalid ranges are rejected. These are local batch-pipeline controls; no cloud orchestration, production deployment, or SLA is claimed.'),code('lineage = json.loads((ROOT / "reports/engineering/lineage.json").read_text())\nprint(json.dumps(lineage, indent=2))'),md('## Rebuild command\n\n`python scripts/run_pipeline.py --stage advanced` rebuilds feature/statistics/SQL outputs, executes these notebooks, runs the checks, and verifies release hashes. Use `--refresh-checksums` only after reviewing intentional artifact changes. `--stage full` also rebuilds the original audit and versioned split outputs. No prediction model is trained.\n\nSource data and data-derived outputs: CC BY-SA 3.0; Dzikovska et al. (2013).')]
    execute('02b_data_pipeline',engineering)

if __name__=='__main__':prepare()
