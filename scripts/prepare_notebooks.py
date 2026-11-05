"""Create and execute the two data-analysis notebooks."""
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient
import sys
import os
from contextlib import redirect_stdout, redirect_stderr
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
ROOT=Path(__file__).resolve().parents[1]
md=nbf.v4.new_markdown_cell
code=nbf.v4.new_code_cell
setup='''from pathlib import Path
import subprocess
import sys

# Works from this checkout, a local Jupyter session, or a fresh Colab session.
roots = [Path.cwd(), *Path.cwd().parents]
ROOT = next((p for p in roots if (p / "data/raw/scientsbank_core.zip").exists()), None)
if ROOT is None:
    checkout = Path("/content/scientsbank_portfolio")
    if not checkout.exists():
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/Junghwan0516/science-answer-benchmark.git",
                        str(checkout)], check=True)
    ROOT = checkout
sys.path.insert(0, str(ROOT / "scripts"))
import json
import pandas as pd
from IPython.display import display, Image
from build_analysis import build
print(f"Project: {ROOT.name}")'''
audit=[md('''# SciEntsBank: Dataset Audit

**Question:** What does one record represent, and what data-quality issues must be understood before modelling?

This independent portfolio study examines public English science short answers. It starts from education experience and focuses on reproducible data inspection. One row represents an answer record, not a unique learner. This release contains data analysis only.

Source: [Dzikovska et al. (2013)](https://aclanthology.org/S13-2045/) and the [author-hosted release](https://github.com/myrosia/semeval-2013-task7). Data and data-derived outputs: CC BY-SA 3.0.'''),code(setup),md('## 1. Rebuild the tables and descriptive figures\nAll descriptive figures use the official training split. Other official splits are exported and checked for inventory and exact normalized input overlap only. No prediction model is fitted.'),code('audit = build()\ndisplay(pd.Series(audit, name="value").to_frame())'),md('## 2. Inspect one question\nRead the question and reference before interpreting its student-answer labels.'),code('train = pd.read_csv(ROOT / "data/processed/train.csv")\nsample = train[train.question_id.eq("EM_45b")]\nprint("QUESTION:", sample.question_text.iloc[0])\nprint("REFERENCE:", sample.reference_answer.iloc[0])\ndisplay(sample[["answer_id", "student_answer", "label"]].head(10))'),md('## 3. Official split inventory\nUA reuses the 135 training questions; UQ holds out questions; UD holds out three domain codes. The counts below are not performance measurements.'),code('display(pd.read_csv(ROOT / "reports/tables/split_manifest.csv"))'),md('## 4. Check missing fields and label counts'),code('display(pd.read_csv(ROOT / "reports/tables/missing_values.csv"))\ndisplay(pd.read_csv(ROOT / "reports/tables/label_distribution.csv"))'),md('## 5. Examine the minority class\nOnly 23 non-domain records occur across 19 questions and 9 domains. These small supports constrain future evaluation.'),code('display(pd.read_csv(ROOT / "reports/tables/minority_domain_distribution.csv"))\ndisplay(pd.read_csv(ROOT / "reports/tables/minority_question_distribution.csv"))\ndisplay(train.loc[train.label.eq("non_domain"), ["domain", "question_id", "student_answer"]])'),md('## 6. Inspect duplicate inputs and conflicting labels\nInput identity uses Unicode NFKC, case folding, and whitespace collapsing on the question, reference, and student answer. It preserves punctuation and word order. Original text and original labels remain unchanged. A repeated input can be a genuine separate submission; repetition is not automatically an error.'),code('duplicates = pd.read_csv(ROOT / "reports/tables/duplicate_groups.csv")\nconflicts = pd.read_csv(ROOT / "reports/tables/conflicting_labels.csv")\nprint("Duplicate groups:", len(duplicates))\nprint("Conflicting input groups:", conflicts.normalized_input_hash.nunique())\ndisplay(conflicts[["question_id", "student_answer", "label", "answer_id"]])'),md('## 7. Record split overlap\nThere are 25 UA records whose normalized complete input also occurs in training. Keep the official benchmark intact. A future model report should present an additional non-overlapping slice without choosing training changes from test outcomes.'),code('display(pd.read_csv(ROOT / "reports/tables/train_test_overlap.csv"))'),md('## Interpretation and next questions\n\n- No null or blank values were found in the seven checked training fields. This is not a claim that all labels are correct.\n- The 4,969 records reduce to 4,826 normalized input groups. There are 82 repeated groups, 143 excess rows, and 6 label-conflicting groups comprising 15 records.\n- Rare-class examples are spread across questions, but future splits still need explicit class-support checks.\n- No labels have been changed, and no records have been deduplicated.\n- Future modelling needs question-level development splits, transparent duplicate handling, and separate UA/UQ/UD reporting.\n\n**Status:** data audit complete; model development deferred.')]
figures=[
('01_class_distribution','Class imbalance','Correct answers account for 40.41% of training records. A majority-class rule would achieve this training accuracy while missing every other class. The majority/minority count ratio is about 87.3; it is not a model result.'),
('02_domain_label_counts','Domain counts','Each cell is an answer-record count. Domain codes are retained from the source rather than expanded into unverified names.'),
('03_domain_label_shares','Within-domain shares','Each row sums to 100%. This removes domain sample-size differences but does not remove uncertainty from small domains. Read it alongside the counts.'),
('04_answer_length_ecdf','Answer-length distributions','Whitespace-separated word count is a transparent length measure. Each curve uses its own class denominator; the non-domain curve has only 23 records. The pooled median is 11 words.'),
('05_answer_length_boxplot','Length and outliers','The boxes summarize quartiles; points outside the whiskers are retained observations. No record is removed simply because it is long.'),
('06_reference_overlap','Reference-answer lexical overlap','Jaccard similarity equals the intersection divided by the union of lower-case word sets. Stop words are retained. Lexical overlap is not a semantic correctness score.'),
('07_question_sample_sizes','Unequal question contributions','Questions are ranked by answer count. This is not a time series. Unequal contributions motivate an additional question-weighted analysis.'),
('08_question_label_diversity','Question-level label diversity','Each point is a question. Entropy summarizes diversity in the observed labels, not latent student ability or a validated question-difficulty scale. Class share and entropy are derived from the same labels.'),
('09_minority_domain_distribution','Minority-class concentration','Counts and within-domain rates use different denominators. Non-domain records occur in 9 of the 12 training domains and 19 of the 135 questions.'),
('10_duplicate_structure','Repeated inputs','The left panel uses a log count axis so repeated groups remain visible. The right panel separates unique normalized inputs from excess repeated rows. Repetition can reflect independent submissions.'),
('11_label_conflicts','Conflicting annotations','Six normalized complete-input groups contain more than one label. Inspect the corresponding table before considering any relabelling; the release preserves all original labels.'),
('12_negation_markers','Negation markers','The listed words and contractions are lexical flags. Scientific contradiction can occur without them, and correct explanations can contain them. This indicator is not a contradiction classifier.'),
('13_answer_vs_question_weighting','Aggregation choices','Answer-weighted shares pool all records. Question-weighted shares average within-question shares with equal weight per question. Neither estimates a unique-student distribution.'),
('14_domain_reference_overlap','Overlap across domains','Domain differences may reflect vocabulary, references, questions, or annotation. This descriptive plot cannot identify which mechanism caused them.')]
eda=[md('''# SciEntsBank: Exploratory Data Analysis

**Question:** How do class balance, question composition, answer length, and lexical relationships vary in the training data?

This notebook presents 14 measured descriptive views. It contains no AI model training or performance claims. Read the audit notebook first. All graphs use the official training data; official test records are not used for exploratory model selection.'''),code(setup),code('if not (ROOT / "reports/tables/audit_summary.json").exists():\n    build()\ntrain = pd.read_csv(ROOT / "data/processed/train.csv")\nprint(f"Training answer records: {len(train):,}")')]
for i,(name,title,interpretation) in enumerate(figures,1):
 eda.extend([md(f'## {i}. {title}\n\n{interpretation}'),code(f'display(Image(filename=str(ROOT / "reports/figures/{name}.png")))')])
eda.extend([md('## Supporting numeric tables'),code('display(pd.read_csv(ROOT / "reports/tables/text_feature_summary.csv", header=[0, 1], index_col=0))\ndisplay(pd.read_csv(ROOT / "reports/tables/weighting_comparison.csv"))'),md('## Limits and follow-up\n\nThese are associations in one public English short-answer corpus, not evidence of causal effects, learner understanding, teaching impact, or applicability to Korean responses. No claims about automatic grading accuracy have been tested. Next steps, if model development resumes, are question-grouped splitting, baseline comparison, and independent final evaluation.\n\nData source: Dzikovska et al. (2013), https://aclanthology.org/S13-2045/. Source dataset and data-derived material: CC BY-SA 3.0. See `data/README.md`.')])
for name,cells in [('00_dataset_audit',audit),('01_exploratory_analysis',eda)]:
 nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
 if '--in-process' in sys.argv:
  shell = InteractiveShell.instance()
  previous_cwd = Path.cwd()
  os.chdir(ROOT)
  count = 0
  try:
   for cell in nb.cells:
    if cell.cell_type != 'code':
     continue
    count += 1
    with capture_output() as captured:
     result = shell.run_cell(cell.source, store_history=False)
    if result.error_before_exec or result.error_in_exec:
     raise RuntimeError(captured.stdout + captured.stderr) from (result.error_before_exec or result.error_in_exec)
    outputs = []
    if captured.stdout:
     outputs.append(nbf.v4.new_output('stream', name='stdout', text=captured.stdout))
    if captured.stderr:
     outputs.append(nbf.v4.new_output('stream', name='stderr', text=captured.stderr))
    for output in captured.outputs:
     outputs.append(nbf.v4.new_output('display_data', data=output.data, metadata=output.metadata))
    cell.outputs = outputs
    cell.execution_count = count
   nb.metadata['execution_note'] = 'All code cells executed in order with IPython InteractiveShell; no network kernel was used.'
  finally:
   os.chdir(previous_cwd)
 else:
  NotebookClient(nb,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
 nbf.write(nb,ROOT/f'notebooks/{name}.ipynb')
 print('Executed',name)
