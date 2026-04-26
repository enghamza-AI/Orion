# core/orchestrator.py


import pandas as pd
import time                              


from engines.noise_auditor import NoiseAuditor
from engines.bias_variance_engine import BiasVarianceEngine
from engines.leakage_scanner import LeakageScanner
from engines.curve_autopsy import CurveAutopsy


class Orchestrator:
   

    def __init__(self, df: pd.DataFrame, target_col: str, id_col: str = None):
       
        self.df = df
        self.target_col = target_col
        self.id_col = id_col
        self.full_report = {}     
        self.run_log = []         

    def _log(self, message: str):
        
        print(message)                
        self.run_log.append(message)  

    def run_all(self, progress_callback=None) -> dict:
       
        self._log(" Observatory starting full diagnostic run...")
        start_total = time.time()   

        
        self._log("  Engine 1/4: Running Noise Audit...")
        t0 = time.time()

        auditor = NoiseAuditor(df=self.df, target_col=self.target_col)
        noise_results = auditor.audit()   

        elapsed = round(time.time() - t0, 2)
        self._log(f"    Noise Audit complete in {elapsed}s")

        
        self.full_report["noise_audit"] = noise_results

        
        if progress_callback:
            progress_callback(25, "Noise Audit complete")  

        
        self._log("  Engine 2/4: Running Bias-Variance Analysis...")
        t0 = time.time()

        bv_engine = BiasVarianceEngine(df=self.df, target_col=self.target_col)
        bv_results = bv_engine.run()

        elapsed = round(time.time() - t0, 2)
        self._log(f"    Bias-Variance Analysis complete in {elapsed}s")

        self.full_report["bias_variance"] = bv_results

        if progress_callback:
            progress_callback(50, "Bias-Variance Analysis complete")  

        
        self._log("  Engine 3/4: Running Leakage Scanner...")
        t0 = time.time()

        scanner = LeakageScanner(
            df=self.df,
            target_col=self.target_col,
            id_col=self.id_col         
        )
        leakage_results = scanner.scan()

        elapsed = round(time.time() - t0, 2)
        self._log(f"    Leakage Scanner complete in {elapsed}s")

        self.full_report["leakage_scan"] = leakage_results

        if progress_callback:
            progress_callback(75, "Leakage Scanner complete")  

        
        self._log("  Engine 4/4: Running Learning Curve Autopsy...")
        t0 = time.time()

        autopsy = CurveAutopsy(df=self.df, target_col=self.target_col)
        curve_results = autopsy.run()

        elapsed = round(time.time() - t0, 2)
        self._log(f"    Curve Autopsy complete in {elapsed}s")

        self.full_report["curve_autopsy"] = curve_results

        if progress_callback:
            progress_callback(100, "All engines complete")  

        
        total_time = round(time.time() - start_total, 2)
        self._log(f"\n Full diagnostic complete in {total_time}s")

       
        self.full_report["meta"] = {
            "target_column": self.target_col,
            "n_rows": len(self.df),
            "n_columns": len(self.df.columns),
            "run_log": self.run_log,
            "total_runtime_seconds": total_time
        }

        return self.full_report

    def get_summary(self) -> dict:
     
        if not self.full_report:
            return {"error": "Run run_all() first before calling get_summary()"}

        summary = {}

      
        noise = self.full_report.get("noise_audit", {})
        summary["noise"] = {
            "missing_severity": noise.get("missing_values", {}).get("severity", "N/A"),
            "outlier_severity": noise.get("outliers", {}).get("severity", "N/A"),
        }

        
        bv = self.full_report.get("bias_variance", {})
        summary["bias_variance"] = {
            "best_model": bv.get("best_model", "N/A"),
            "best_test_auc": bv.get("best_test_auc", "N/A"),
        }

        
        leakage = self.full_report.get("leakage_scan", {})
        leakage_summary = leakage.get("summary", {})
        summary["leakage"] = {
            "sins_detected": leakage_summary.get("total_sins_detected", 0),
            "overall_severity": leakage_summary.get("overall_severity", "N/A"),
        }

       
        curve = self.full_report.get("curve_autopsy", {})
        summary["curve_autopsy"] = {
            "diagnosis": curve.get("diagnosis", "N/A"),
            "final_val_auc": curve.get("final_val_auc", "N/A"),
        }

        return summary