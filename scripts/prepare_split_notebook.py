"""Generate and sequentially execute the split notebook with captured output.

Uses Python execution and a raw-MIME display adapter; no network kernel required.
The notebook's display calls use the standard Jupyter-provided display function.
"""
from pathlib import Path
from contextlib import redirect_stdout,redirect_stderr
import io
import json
import os

ROOT=Path(__file__).resolve().parents[1]

def markdown(text):return dict(cell_type='markdown',metadata={},source=text.splitlines(keepends=True))
def code(text):return dict(cell_type='code',metadata={},source=text.splitlines(keepends=True),outputs=[],execution_count=None)

def prepare():
    setup='''from pathlib import Path
import subprocess
import sys
import json
import base64
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
from build_splits import build, load_training, verify_split
print("Split source: official training CSV only")'''
    cells=[markdown('''# 02. Question-Grouped Split Design

**Question:** How can we prepare development data without placing answers to the same question in different partitions?

This notebook establishes versioned fit/development/calibration assignments, duplicate and label-conflict policies, and explicit overlap checks. No model is trained. The official UA/UQ/UD files are not read by this workflow.

Source: SciEntsBank, Dzikovska et al. (2013), https://aclanthology.org/S13-2045/. Data-derived material: CC BY-SA 3.0. Read `docs/split_design.md` for the full protocol.'''),
    code(setup),
    markdown('''## 1. Freeze the allocation rule

Sort the 135 question IDs, permute with NumPy PCG64 seed 42, and allocate 95/20/20 whole questions. Accept the first candidate with at least 10 records of each label in fit, at least 3 in each held-out partition, and 12–18% of answer records in each held-out partition. Stop after at most 1,000 candidates.

This constrained allocation uses only training IDs, counts, and labels. It does not search model scores or official test labels. A support of three is a coverage minimum, not a reliable sample size.'''),
    code('protocol = build(ROOT)\nprint("\\nProtocol version:", protocol["version"])\nprint("Source SHA-256:", protocol["source_training_sha256"])'),
    markdown('''## 2. Check question and class supports

The first eligible candidate is candidate 5. Fit has 3,536 records, development 715, and calibration 718. Every question belongs to exactly one partition. The rare-class counts are 17/3/3; their uncertainty must remain visible in later model reports.'''),
    code('summary = pd.read_csv(ROOT / "reports/splits/partition_summary.csv")\nsupport = pd.read_csv(ROOT / "reports/splits/class_support.csv")\nprint(summary.to_string(index=False))\nprint("\\nClass supports:")\nprint(support.to_string(index=False))'),
    markdown('''## 3. Inspect the partition figures

Answer proportions differ from question proportions because questions contain unequal numbers of answers. Domain coverage is reported after allocation, not optimized. All three plots are computed from training-only assignments.''')]
    for file in ['01_partition_sizes.png','02_class_support.png','03_domain_question_support.png']:
        cells.append(code('image_path = ROOT / "reports/splits/figures/'+file+'"\ndisplay({"image/png": base64.b64encode(image_path.read_bytes()).decode("ascii")}, raw=True)'))
    cells.extend([
    markdown('''## 4. Verify disjointness and preserve annotations

The checks cover answer IDs, question IDs, and normalized complete inputs. They do not establish learner-disjointness or detect all paraphrases. Original rows and labels remain unchanged. Primary sample weights are one. Optional inverse-input-multiplicity weights are prepared for fit-only sensitivity experiments; no weighting experiment is run here.'''),
    code('verify_split(ROOT)\nprint(pd.read_csv(ROOT / "reports/splits/overlap_checks.csv").to_string(index=False))\nassignments = pd.read_csv(ROOT / "data/splits/v1/answer_assignments.csv")\nprint("\\nFlagged conflicting rows by partition:")\nprint(assignments.groupby("internal_split")["has_conflicting_labels"].sum().to_string())'),
    markdown('''## 5. Join the manifest to the source data

Use a one-to-one join on sample_id. Assignment tables do not duplicate answer text. Downstream vectorizers, scalers, and model parameters must be learned from fit only. Development selects classifier settings; calibration is reserved for the selected model's probabilities and review threshold.'''),
    code('training = load_training(ROOT)\njoined = training.merge(assignments[["sample_id", "internal_split"]], on="sample_id", validate="one_to_one")\nassert len(joined) == 4969\nprint(joined[["sample_id", "question_id", "label", "internal_split"]].head(8).to_string(index=False))'),
    markdown('''## 6. Final evaluation is still a separate design decision

This split is for development. A model fitted only on 95 question IDs has not seen every question appearing in the official UA partition. Before any official scoring, freeze the final-fit procedure: either document a full-training refit with cross-fitted calibration, or retain the held-out setup and report the actual seen-question/held-out-question slices within official UA.

Do not refit on calibration labels while presenting those same predictions as independent calibration. No official UA/UQ/UD scores, fitted model, classifier hyperparameters, or production accuracy are produced by this notebook.

**Next task:** implement classical baselines in `03_classical_baselines.ipynb` using these immutable assignments. Consider fixed-five-class macro-F1 with class supports, alongside accuracy, weighted-F1, and the confusion matrix. The tiny minority support limits model-selection certainty.''')])
    namespace={'__name__':'__main__'}
    previous=Path.cwd();os.chdir(ROOT)
    try:
        count=0
        for cell in cells:
            if cell['cell_type']!='code':continue
            count+=1;outputs=[]
            def display(data,raw=False):
                if not raw or not isinstance(data,dict):raise ValueError('Only explicit raw MIME display is used in this notebook.')
                outputs.append(dict(output_type='display_data',data=data,metadata={}))
            namespace['display']=display
            stdout=io.StringIO();stderr=io.StringIO()
            with redirect_stdout(stdout),redirect_stderr(stderr):exec(compile(''.join(cell['source']),f'notebook-cell-{count}','exec'),namespace)
            streams=[]
            if stdout.getvalue():streams.append(dict(output_type='stream',name='stdout',text=stdout.getvalue().splitlines(keepends=True)))
            if stderr.getvalue():streams.append(dict(output_type='stream',name='stderr',text=stderr.getvalue().splitlines(keepends=True)))
            cell['outputs']=streams+outputs;cell['execution_count']=count
    finally:os.chdir(previous)
    notebook=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'execution_note':'Code cells executed sequentially with Python exec, stdout capture, and raw-MIME display capture; all outputs are computed.'})
    for i,cell in enumerate(cells):cell['id']=f'split-v1-{i:02}'
    (ROOT/'notebooks/02_split_design.ipynb').write_text(json.dumps(notebook,indent=1,ensure_ascii=False)+'\n')
    print(f'Executed {count} code cells and captured three computed figures.')

if __name__=='__main__':prepare()
