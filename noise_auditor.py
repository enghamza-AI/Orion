# engines/noise_auditor.py

import pandas as pd
import numpy as np

class NoiseAuditor:
    def __init__(self, df: pd.DataFrame, target_col: str):

        self.df = df.copy()
        self.target_col = target_col
        self.results = {}

    def check_missing(self):
        missing_pct = self.df.isnull().mean() * 100

        missing_pct = missing_pct[missing_pct > 0]

        missing_pct = missing_pct.sort_values(ascending=False)

        severity = "HIGH" if missing_pct.max() > 30 else \
                    "MEDIUM" if missing_pct.max() > 5 else "LOW"
        
        self.results["missing_values"] = {
            "affected_columns": missing_pct.to_dict(),
            "severity": severity,
            "n_affected": len(missing_pct)
        }

    def check_outliers(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.to_list()
        numeric_cols = [c for c in numeric_cols if c != self.target_col]

        outlier_info = {}

        for col in numeric_cols:

            col_data = self.df[col].dropna()

            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            n_outliers = ((col_data < lower_bound) | (col_data > upper_bound)).sum()

            pct_outliers = (n_outliers / len(col_data)) * 100

            if n_outliers > 0:
                outlier_info[col] = round(pct_outliers, 2)

        outlier_info = dict(sorted(outlier_info.items(), key=lambda x: x[1], reverse=True))

        worst_pct = max(outlier_info.values()) if outlier_info else 0
        severity = "HIGH" if worst_pct > 10 else \
                   "MEDIUM" if worst_pct > 2 else "LOW"

        self.results["outliers"] = {
            "affected_columns": outlier_info,
            "severity": severity,
            "n_affected": len(outlier_info)
        }

    def check_duplicates(self):

        n_dupes = self.df.duplicated().sum()

        pct_dupes = (n_dupes / len(self.df)) * 100

        severity = "HIGH" if pct_dupes > 5 else \
                    "MEDIUM" if pct_dupes > 1 else "LOW"
        
        self.results["duplicates"] = {
            "n_duplicate_rows": int(n_dupes),
            "pct_of_dataset": round(pct_dupes, 2),
            "severity": severity
        }

    def check_class_imbalance(self):

        class_counts = self.df[self.target_col].value_counts(normalize=True) * 100

       
        majority_pct = class_counts.iloc[0]   

       
        minority_pct = class_counts.iloc[-1]  

        
        ratio = majority_pct / minority_pct if minority_pct > 0 else float('int')

        severity = "HIGH" if ratio > 10 else \
                   "MEDIUM" if ratio > 3 else "LOW"

        self.results["class_imbalance"] = {
            "class_distribution": class_counts.to_dict(),  
            "imbalance_ratio": round(ratio, 2),
            "severity": severity
        }

    
    def check_low_variance(self):
       
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != self.target_col]

        low_var_cols = {}

        for col in numeric_cols:
            col_data = self.df[col].dropna()

            if len(col_data) == 0:
                continue  

            
            most_common_val = col_data.mode()[0]

            
            pct_most_common = (col_data == most_common_val).mean() * 100

            
            if pct_most_common >= 95:
                low_var_cols[col] = round(pct_most_common, 2)

        severity = "HIGH" if len(low_var_cols) > 10 else \
                   "MEDIUM" if len(low_var_cols) > 3 else "LOW"

        self.results["low_variance"] = {
            "affected_columns": low_var_cols,
            "severity": severity,
            "n_affected": len(low_var_cols)
        }

   
    def check_dtype_mismatches(self):
        
        mismatch_cols = []

        for col in self.df.columns:
            if col == self.target_col:
                continue

            
            if self.df[col].dtype == object:
               
                converted = pd.to_numeric(self.df[col], errors='coerce')

                
                pct_convertible = converted.notna().mean()
                if pct_convertible > 0.8:
                    mismatch_cols.append(col)

        severity = "HIGH" if len(mismatch_cols) > 5 else \
                   "MEDIUM" if len(mismatch_cols) > 1 else "LOW"

        self.results["dtype_mismatches"] = {
            "affected_columns": mismatch_cols,
            "severity": severity,
            "n_affected": len(mismatch_cols)
        }

    
    def audit(self) -> dict:
       
        self.check_missing()           
        self.check_outliers()          
        self.check_duplicates()        
        self.check_class_imbalance()  
        self.check_low_variance()      
        self.check_dtype_mismatches()  

        return self.results            