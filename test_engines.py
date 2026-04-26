# tests/test_engines.py


import pytest                         
import pandas as pd
import numpy as np
import sys
import os


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


from engines.noise_auditor import NoiseAuditor
from engines.bias_variance_engine import BiasVarianceEngine
from engines.leakage_scanner import LeakageScanner
from engines.curve_autopsy import CurveAutopsy
from core.orchestrator import Orchestrator
from core.trust_score import TrustScoreEngine




@pytest.fixture
def clean_df():
   
    np.random.seed(42)
    n = 300

    df = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, n),
        'feature_2': np.random.normal(5, 2, n),
        'feature_3': np.random.uniform(0, 100, n),
        'feature_4': np.random.exponential(2, n),
        'feature_5': np.random.normal(10, 3, n),
        'feature_6': np.random.uniform(-1, 1, n),
        'feature_7': np.random.normal(0, 0.5, n),
        'feature_8': np.random.normal(3, 1, n),
        'target':    np.random.choice([0, 1], n, p=[0.6, 0.4])
    })

    return df


@pytest.fixture
def dirty_df():

    np.random.seed(42)
    n = 400

    df = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, n),
        'feature_2': np.random.normal(5, 2, n),
        'feature_3': np.random.uniform(0, 100, n),
        'feature_4': np.random.exponential(2, n),
        'target':    np.random.choice([0, 1], n, p=[0.95, 0.05]) 
    })

    
    missing_indices = np.random.choice(n, size=int(n * 0.25), replace=False)
    df.loc[missing_indices, 'feature_1'] = np.nan

  
    outlier_indices = np.random.choice(n, size=20, replace=False)
    df.loc[outlier_indices, 'feature_2'] = 99999

 
    dupes = df.sample(50, random_state=42)
    df = pd.concat([df, dupes], ignore_index=True)

    return df



class TestNoiseAuditor:
   

    def test_returns_dict(self, clean_df):
     
        auditor = NoiseAuditor(df=clean_df, target_col='target')
        result = auditor.audit()

        
        assert isinstance(result, dict), "audit() must return a dictionary"

    def test_has_all_six_checks(self, clean_df):
       
        auditor = NoiseAuditor(df=clean_df, target_col='target')
        result = auditor.audit()

        expected_keys = [
            'missing_values',
            'outliers',
            'duplicates',
            'class_imbalance',
            'low_variance',
            'dtype_mismatches'
        ]

        for key in expected_keys:
            
            assert key in result, f"Missing key '{key}' in noise audit result"

    def test_detects_missing_values(self, dirty_df):
      
        auditor = NoiseAuditor(df=dirty_df, target_col='target')
        result = auditor.audit()

        severity = result['missing_values']['severity']
        assert severity == 'HIGH', (
            f"Expected HIGH severity for 25% missing, got {severity}"
        )

    def test_detects_class_imbalance(self, dirty_df):
        
        auditor = NoiseAuditor(df=dirty_df, target_col='target')
        result = auditor.audit()

        severity = result['class_imbalance']['severity']
        assert severity == 'HIGH', (
            f"Expected HIGH severity for 95/5 imbalance, got {severity}"
        )

    def test_severity_values_are_valid(self, clean_df):
       
        auditor = NoiseAuditor(df=clean_df, target_col='target')
        result = auditor.audit()

        valid_severities = {'HIGH', 'MEDIUM', 'LOW', 'NONE', 'UNKNOWN', 'CLEAN'}

        for check_name, check_data in result.items():
            if isinstance(check_data, dict) and 'severity' in check_data:
                sev = check_data['severity']
                assert sev in valid_severities, (
                    f"Invalid severity '{sev}' in check '{check_name}'"
                )

    def test_does_not_modify_original_df(self, clean_df):
        
        original_shape = clean_df.shape
        original_cols  = list(clean_df.columns)

        auditor = NoiseAuditor(df=clean_df, target_col='target')
        auditor.audit()

        
        assert clean_df.shape == original_shape, "Auditor modified dataframe shape"
        assert list(clean_df.columns) == original_cols, "Auditor modified dataframe columns"




class TestBiasVarianceEngine:

    def test_returns_dict_with_models(self, clean_df):
        
        engine = BiasVarianceEngine(df=clean_df, target_col='target')
        result = engine.run()

        assert isinstance(result, dict)
        assert 'models' in result, "Result must contain 'models' key"
        assert len(result['models']) == 5, (
            f"Expected 5 models, got {len(result['models'])}"
        )

    def test_each_model_has_required_fields(self, clean_df):
       
        engine = BiasVarianceEngine(df=clean_df, target_col='target')
        result = engine.run()

        required_fields = ['train_auc', 'test_auc', 'gap', 'diagnosis', 'recommendation']

        for model_name, model_data in result['models'].items():
            for field in required_fields:
                assert field in model_data, (
                    f"Model '{model_name}' missing field '{field}'"
                )

    def test_auc_values_are_in_valid_range(self, clean_df):
        
        engine = BiasVarianceEngine(df=clean_df, target_col='target')
        result = engine.run()

        for model_name, model_data in result['models'].items():
            train_auc = model_data['train_auc']
            test_auc  = model_data['test_auc']

            assert 0.0 <= train_auc <= 1.0, (
                f"{model_name} train_auc {train_auc} out of range"
            )
            assert 0.0 <= test_auc <= 1.0, (
                f"{model_name} test_auc {test_auc} out of range"
            )

    def test_gap_equals_train_minus_test(self, clean_df):
       
        engine = BiasVarianceEngine(df=clean_df, target_col='target')
        result = engine.run()

        for model_name, model_data in result['models'].items():
            expected_gap = round(
                model_data['train_auc'] - model_data['test_auc'], 4
            )
            actual_gap = model_data['gap']

            assert abs(expected_gap - actual_gap) < 0.001, (
                f"{model_name}: gap mismatch. "
                f"Expected {expected_gap}, got {actual_gap}"
            )

    def test_best_model_key_exists(self, clean_df):
        
        engine = BiasVarianceEngine(df=clean_df, target_col='target')
        result = engine.run()

        assert 'best_model' in result
        assert 'best_test_auc' in result
       
        assert result['best_model'] in result['models']




class TestLeakageScanner:

    def test_returns_dict_with_summary(self, clean_df):
        
        scanner = LeakageScanner(df=clean_df, target_col='target')
        result = scanner.scan()

        assert isinstance(result, dict)
        assert 'summary' in result, "Leakage result must contain 'summary'"

    def test_summary_has_required_fields(self, clean_df):
       
        scanner = LeakageScanner(df=clean_df, target_col='target')
        result = scanner.scan()

        summary = result['summary']
        assert 'total_sins_detected' in summary
        assert 'overall_severity' in summary
        assert 'total_sins_checked' in summary

    def test_sins_detected_is_non_negative(self, clean_df):
        
        scanner = LeakageScanner(df=clean_df, target_col='target')
        result = scanner.scan()

        sins = result['summary']['total_sins_detected']
        assert sins >= 0, f"Sins detected can't be negative, got {sins}"

    def test_sins_detected_does_not_exceed_total_checked(self, clean_df):
       
        scanner = LeakageScanner(df=clean_df, target_col='target')
        result = scanner.scan()

        detected = result['summary']['total_sins_detected']
        total    = result['summary']['total_sins_checked']

        assert detected <= total, (
            f"Detected {detected} sins but only check {total}"
        )

    def test_all_five_sins_are_present(self, clean_df):
      
        scanner = LeakageScanner(df=clean_df, target_col='target')
        result = scanner.scan()

        expected_sins = [
            'target_encoding_leak',
            'feature_from_target',
            'scaling_leak',
            'group_overlap',
            'duplicate_leakage'
        ]

        for sin in expected_sins:
            assert sin in result, f"Sin '{sin}' missing from leakage scan result"




class TestCurveAutopsy:

    def test_returns_dict_with_diagnosis(self, clean_df):
        
        autopsy = CurveAutopsy(df=clean_df, target_col='target')
        result = autopsy.run()

        assert isinstance(result, dict)
        assert 'diagnosis' in result

    def test_diagnosis_is_valid_value(self, clean_df):
       
        autopsy = CurveAutopsy(df=clean_df, target_col='target')
        result = autopsy.run()

        valid_diagnoses = {'HEALTHY', 'DATA-STARVED', 'OVER-COMPLEX', 'LEAKY'}
        assert result['diagnosis'] in valid_diagnoses, (
            f"Invalid diagnosis '{result['diagnosis']}'"
        )

    def test_curve_data_has_correct_length(self, clean_df):
        
        autopsy = CurveAutopsy(df=clean_df, target_col='target')
        result = autopsy.run()

        curve_data = result['curve_data']
        expected_length = 8   

        assert len(curve_data['train_sizes'])       == expected_length
        assert len(curve_data['train_scores_mean']) == expected_length
        assert len(curve_data['val_scores_mean'])   == expected_length

    def test_final_aucs_are_in_valid_range(self, clean_df):
       
        autopsy = CurveAutopsy(df=clean_df, target_col='target')
        result = autopsy.run()

        assert 0.0 <= result['final_train_auc'] <= 1.0
        assert 0.0 <= result['final_val_auc']   <= 1.0




class TestOrchestrator:

    def test_run_all_returns_all_four_sections(self, clean_df):
        
        orch = Orchestrator(df=clean_df, target_col='target')
        result = orch.run_all()

        required_sections = [
            'noise_audit',
            'bias_variance',
            'leakage_scan',
            'curve_autopsy',
            'meta'
        ]

        for section in required_sections:
            assert section in result, (
                f"Orchestrator result missing section '{section}'"
            )

    def test_meta_contains_correct_row_count(self, clean_df):
        
        orch = Orchestrator(df=clean_df, target_col='target')
        result = orch.run_all()

        assert result['meta']['n_rows'] == len(clean_df), (
            "Meta n_rows doesn't match actual dataframe length"
        )

    def test_meta_contains_target_column(self, clean_df):
      
        orch = Orchestrator(df=clean_df, target_col='target')
        result = orch.run_all()

        assert result['meta']['target_column'] == 'target'

    def test_full_pipeline_does_not_crash_on_dirty_data(self, dirty_df):
       
        
        orch = Orchestrator(df=dirty_df, target_col='target')
        result = orch.run_all()

        
        assert result is not None
        assert 'noise_audit' in result




class TestTrustScore:

    def test_score_is_in_valid_range(self, clean_df):
        
        orch = Orchestrator(df=clean_df, target_col='target')
        full_report = orch.run_all()

        trust = TrustScoreEngine(full_report)
        result = trust.compute()

        assert 0 <= result['score'] <= 100, (
            f"Score {result['score']} outside valid range 0–100"
        )

    def test_grade_is_valid(self, clean_df):
       
        orch = Orchestrator(df=clean_df, target_col='target')
        full_report = orch.run_all()

        trust = TrustScoreEngine(full_report)
        result = trust.compute()

        assert result['grade'] in {'A', 'B', 'C', 'D', 'F'}, (
            f"Invalid grade '{result['grade']}'"
        )

    def test_dirty_data_scores_lower_than_clean(self, clean_df, dirty_df):
       
        
        orch_clean = Orchestrator(df=clean_df, target_col='target')
        trust_clean = TrustScoreEngine(orch_clean.run_all())
        clean_score = trust_clean.compute()['score']

       
        orch_dirty = Orchestrator(df=dirty_df, target_col='target')
        trust_dirty = TrustScoreEngine(orch_dirty.run_all())
        dirty_score = trust_dirty.compute()['score']

        assert dirty_score < clean_score, (
            f"Dirty data scored {dirty_score} but clean data scored {clean_score}. "
            f"Dirty should always score lower."
        )

    def test_result_has_required_keys(self, clean_df):
      
        orch = Orchestrator(df=clean_df, target_col='target')
        full_report = orch.run_all()

        trust = TrustScoreEngine(full_report)
        result = trust.compute()

        required_keys = ['score', 'grade', 'color', 'verdict', 'breakdown']
        for key in required_keys:
            assert key in result, f"Trust score result missing key '{key}'"