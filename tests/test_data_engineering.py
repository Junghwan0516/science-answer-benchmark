"""Integration checks for feature contracts and relational data integrity."""
from pathlib import Path
import sqlite3,sys,tempfile,unittest,hashlib
from unittest.mock import patch
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from build_splits import load_training
from build_warehouse import build,validate_features,ROOT
from analysis_extensions import make_features,cluster_label_intervals

class DataEngineeringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=load_training();cls.features=pd.read_csv(ROOT/'data/processed/training_features.csv')
        cls.tmp=tempfile.TemporaryDirectory();cls.db=Path(cls.tmp.name)/'warehouse.sqlite'
        cls.report=build(db_path=cls.db,write_reports=False)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def test_sql_matches_source_counts(self):
        with sqlite3.connect(self.db) as c:
            self.assertEqual(c.execute('SELECT SUM(answer_count) FROM v_domain_summary').fetchone()[0],4969)
            self.assertEqual(c.execute('SELECT SUM(non_domain_count) FROM v_partition_summary').fetchone()[0],23)
            self.assertEqual(c.execute('SELECT COUNT(*) FROM v_question_metrics').fetchone()[0],135)
    def test_database_rejects_orphan_and_unknown_label(self):
        for q,label in [('missing-question','correct'),(self.data.question_id.iloc[0],'unknown-label')]:
            with sqlite3.connect(self.db) as c:
                c.execute('PRAGMA foreign_keys=ON')
                with self.assertRaises(sqlite3.IntegrityError):
                    c.execute('INSERT INTO answers VALUES (?,?,?,?,?,?)',('bad-id','bad-answer',q,'Test answer',label,'0'*64))
    def test_fit_weight_cannot_be_null(self):
        with sqlite3.connect(self.db) as c:
            with self.assertRaises(sqlite3.IntegrityError):c.execute("UPDATE development_assignments SET sensitivity_fit_weight=NULL WHERE internal_split='fit'")
    def test_feature_contract_rejects_missing_coverage_and_invalid_range(self):
        with self.assertRaisesRegex(ValueError,'cover each'):validate_features(self.data,self.features.iloc[:-1])
        bad=self.features.copy();bad.loc[0,'reference_token_jaccard']=1.01
        with self.assertRaisesRegex(ValueError,'Out-of-range'):validate_features(self.data,bad)
    def test_features_depend_on_content_not_input_order(self):
        subset=self.data.iloc[:30]
        a=make_features(subset).sort_values('sample_id').reset_index(drop=True)
        b=make_features(subset.sample(frac=1,random_state=9)).sort_values('sample_id').reset_index(drop=True)
        pd.testing.assert_frame_equal(a,b)
    def test_failed_build_preserves_accepted_database(self):
        digest=hashlib.sha256(self.db.read_bytes()).hexdigest()
        original=pd.DataFrame.to_sql
        def fail_features(frame,name,*args,**kwargs):
            if name=='answer_features':raise ValueError('Injected feature write failure')
            return original(frame,name,*args,**kwargs)
        with patch.object(pd.DataFrame,'to_sql',new=fail_features):
            with self.assertRaisesRegex(ValueError,'Injected'):
                build(db_path=self.db,write_reports=False)
        self.assertEqual(digest,hashlib.sha256(self.db.read_bytes()).hexdigest())
        self.assertFalse(self.db.with_suffix('.building.sqlite').exists())
    def test_resampling_is_reproducible_and_intervals_are_bounded(self):
        a=cluster_label_intervals(self.data,replicates=100,seed=8)
        b=cluster_label_intervals(self.data,replicates=100,seed=8)
        pd.testing.assert_frame_equal(a,b)
        for col in ['question_cluster_low','question_cluster_high','row_resampling_low','row_resampling_high']:
            self.assertTrue(a[col].between(0,1).all())
        self.assertTrue(a.question_cluster_low.le(a.question_cluster_high).all())

if __name__=='__main__':unittest.main()
