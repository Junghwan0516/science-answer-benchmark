"""Checks for split reproducibility and leakage protection."""
from pathlib import Path
import sys
import unittest
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from build_splits import load_training,make_assignments,validate_assignments

class SplitDesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=load_training()
        cls.assignments,cls.candidate=make_assignments(cls.data)

    def test_reproducible_and_label_supported(self):
        other,candidate=make_assignments(self.data)
        pd.testing.assert_frame_equal(self.assignments,other)
        self.assertEqual(candidate,self.candidate)
        support=pd.crosstab(other.internal_split,other.label)
        self.assertTrue(support.loc[['development','calibration']].ge(3).all().all())
        self.assertTrue(support.loc['fit'].ge(10).all())

    def test_zero_group_overlap_and_complete_coverage(self):
        overlap=validate_assignments(self.data,self.assignments)
        self.assertTrue(overlap.iloc[:,2:].eq(0).all().all())
        self.assertEqual(len(self.assignments),4969)

    def test_reject_question_crossing_partitions(self):
        bad=self.assignments.copy()
        index=bad[bad.internal_split.eq('development')].index[0]
        bad.loc[index,'internal_split']='fit'
        with self.assertRaisesRegex(ValueError,'question crosses'):
            validate_assignments(self.data,bad)

    def test_reject_missing_or_duplicated_answers(self):
        with self.assertRaisesRegex(ValueError,'cover each'):
            validate_assignments(self.data,self.assignments.iloc[:-1])
        bad=pd.concat([self.assignments,self.assignments.iloc[[0]]],ignore_index=True)
        with self.assertRaisesRegex(ValueError,'more than once'):
            validate_assignments(self.data,bad)

    def test_reject_label_changes(self):
        bad=self.assignments.copy()
        bad.loc[0,'label']='wrong_label'
        with self.assertRaisesRegex(ValueError,'label must remain'):
            validate_assignments(self.data,bad)

    def test_reject_test_records_in_split_generation(self):
        bad=self.data.copy()
        bad.loc[0,'official_split']='test-unseen-answers'
        with self.assertRaisesRegex(ValueError,'test records are prohibited'):
            make_assignments(bad)

if __name__=='__main__':unittest.main()
