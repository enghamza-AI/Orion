# engines/leakage_scanner.py


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


class LeakageScanner:
  

    def __init__(self, df: pd.DataFrame, target_col: str, id_col: str = None):
       
        self.df = df.copy()
        self.target_col = target_col
        self.id_col = id_col
        self.results = {}

    def _encode_target(self):
        
        le = LabelEncoder()
        return le.fit_transform(self.df[self.target_col])

    def _get_numeric_X(self, df=None):
       
        if df is None:
            df = self.df
        X = df.drop(columns=[self.target_col]).select_dtypes(include=[np.number])
        imputer = SimpleImputer(strategy='median')
        return imputer.fit_transform(X), X.columns.tolist()

    
    def check_target_encoding_leak(self):
        
        
        cat_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        cat_cols = [c for c in cat_cols if c != self.target_col]

        
        if len(cat_cols) == 0:
            self.results["target_encoding_leak"] = {
                "detected": False,
                "reason": "No categorical columns found to encode",
                "severity": "NONE"
            }
            return

        y = self._encode_target()
        df_work = self.df.copy()
        df_work['__y__'] = y  

        
        df_leaky = df_work.copy()
        for col in cat_cols[:3]:  
            
            target_mean = df_leaky.groupby(col)['__y__'].transform('mean')
            df_leaky[col + '_encoded'] = target_mean  
            df_leaky = df_leaky.drop(columns=[col])   

        
        df_clean = df_work.copy()
        train_idx, test_idx = train_test_split(df_clean.index, test_size=0.2, random_state=42)
        for col in cat_cols[:3]:
            
            train_means = df_clean.loc[train_idx].groupby(col)['__y__'].mean()
            
            df_clean[col + '_encoded'] = df_clean[col].map(train_means).fillna(0.5)
            df_clean = df_clean.drop(columns=[col])

        
        X_leak = df_leaky.drop(columns=['__y__', self.target_col], errors='ignore')
        X_leak = X_leak.select_dtypes(include=[np.number])
        imp = SimpleImputer(strategy='median')
        X_leak = imp.fit_transform(X_leak)
        X_tr, X_te, y_tr, y_te = train_test_split(X_leak, y, test_size=0.2, random_state=42)
        model = LogisticRegression(max_iter=500)
        model.fit(X_tr, y_tr)
        auc_leaky = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])

        
        X_cln = df_clean.drop(columns=['__y__', self.target_col], errors='ignore')
        X_cln = X_cln.select_dtypes(include=[np.number])
        X_cln = imp.fit_transform(X_cln)
        X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X_cln, y, test_size=0.2, random_state=42)
        model2 = LogisticRegression(max_iter=500)
        model2.fit(X_tr2, y_tr2)
        auc_clean = roc_auc_score(y_te2, model2.predict_proba(X_te2)[:, 1])

        
        inflation = auc_leaky - auc_clean

        self.results["target_encoding_leak"] = {
            "detected": inflation > 0.01,  
            "auc_leaky": round(auc_leaky, 4),
            "auc_clean": round(auc_clean, 4),
            "auc_inflation": round(inflation, 4),
            "severity": "HIGH" if inflation > 0.05 else "MEDIUM" if inflation > 0.01 else "LOW"
        }

    
    def check_feature_from_target(self):
        
        y = self._encode_target()

       
        X_raw = self.df.drop(columns=[self.target_col]).select_dtypes(include=[np.number])
        imp = SimpleImputer(strategy='median')
        X_imp = imp.fit_transform(X_raw)

        suspicious = {}

        for i, col in enumerate(X_raw.columns):
            
            feat = X_imp[:, i].reshape(-1, 1)  
            X_tr, X_te, y_tr, y_te = train_test_split(feat, y, test_size=0.2, random_state=42)
            try:
                m = LogisticRegression(max_iter=300)
                m.fit(X_tr, y_tr)
                auc = roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])
                
                if auc > 0.90:
                    suspicious[col] = round(auc, 4)
            except Exception:
                pass  

        self.results["feature_from_target"] = {
            "detected": len(suspicious) > 0,
            "suspicious_features": suspicious,  
            "severity": "HIGH" if len(suspicious) > 0 else "NONE",
            "explanation": "Features with solo AUC > 0.90 may be derived from or directly related to the target."
        }

   
    def check_scaling_leak(self):
       
        y = self._encode_target()
        X_raw, col_names = self._get_numeric_X()

        X_tr, X_te, y_tr, y_te = train_test_split(X_raw, y, test_size=0.2, random_state=42)

        
        scaler_leaky = StandardScaler()
        X_all_scaled = scaler_leaky.fit_transform(X_raw)  
        X_tr_l = X_all_scaled[:len(X_tr)]                 
        X_te_l = X_all_scaled[len(X_tr):]                 

        model_l = LogisticRegression(max_iter=500)
        model_l.fit(X_tr_l, y_tr)
        auc_leaky = roc_auc_score(y_te, model_l.predict_proba(X_te_l)[:, 1])

        
        scaler_clean = StandardScaler()
        X_tr_c = scaler_clean.fit_transform(X_tr)    
        X_te_c = scaler_clean.transform(X_te)         

        model_c = LogisticRegression(max_iter=500)
        model_c.fit(X_tr_c, y_tr)
        auc_clean = roc_auc_score(y_te, model_c.predict_proba(X_te_c)[:, 1])

        inflation = auc_leaky - auc_clean

        self.results["scaling_leak"] = {
            "detected": abs(inflation) > 0.005,
            "auc_leaky": round(auc_leaky, 4),
            "auc_clean": round(auc_clean, 4),
            "auc_inflation": round(inflation, 4),
            "severity": "HIGH" if abs(inflation) > 0.02 else "MEDIUM" if abs(inflation) > 0.005 else "LOW"
        }

  
    def check_group_overlap(self):
        
       
        id_col = self.id_col
        if id_col is None:
           
            id_candidates = [c for c in self.df.columns
                             if 'id' in c.lower() and c != self.target_col]
            if id_candidates:
                id_col = id_candidates[0]

        if id_col and id_col in self.df.columns:
           
            train_df, test_df = train_test_split(self.df, test_size=0.2, random_state=42)
            train_ids = set(train_df[id_col].unique())
            test_ids = set(test_df[id_col].unique())
            overlap = train_ids.intersection(test_ids)
            overlap_pct = len(overlap) / len(test_ids) * 100

            self.results["group_overlap"] = {
                "detected": overlap_pct > 0,
                "id_column_used": id_col,
                "overlap_pct": round(overlap_pct, 2),
                "n_overlapping_ids": len(overlap),
                "severity": "HIGH" if overlap_pct > 10 else "MEDIUM" if overlap_pct > 1 else "LOW"
            }
        else:
           
            self.results["group_overlap"] = {
                "detected": False,
                "reason": "No ID column found. Manual review recommended.",
                "severity": "UNKNOWN",
                "recommendation": "Specify id_col parameter if your dataset has entity IDs."
            }

    
    def check_duplicate_leakage(self):
        
        train_df, test_df = train_test_split(self.df, test_size=0.2, random_state=42)

        
        feature_cols = [c for c in self.df.columns if c != self.target_col]

        train_rows = set(
            train_df[feature_cols].dropna().apply(tuple, axis=1)
            
        )
        test_rows = test_df[feature_cols].dropna().apply(tuple, axis=1)

        
        leaked_rows = test_rows[test_rows.isin(train_rows)]
        n_leaked = len(leaked_rows)
        pct_leaked = (n_leaked / len(test_df)) * 100

        self.results["duplicate_leakage"] = {
            "detected": n_leaked > 0,
            "n_leaked_rows": n_leaked,
            "pct_of_test": round(pct_leaked, 2),
            "severity": "HIGH" if pct_leaked > 5 else "MEDIUM" if pct_leaked > 0 else "NONE"
        }

    
    def scan(self) -> dict:
        
        self.check_target_encoding_leak()   
        self.check_feature_from_target()    
        self.check_scaling_leak()           
        self.check_group_overlap()          
        self.check_duplicate_leakage()      

        
        detected_count = sum(
            1 for sin, data in self.results.items()
            if isinstance(data, dict) and data.get("detected", False)
        )

        self.results["summary"] = {
            "total_sins_detected": detected_count,
            "total_sins_checked": 5,
            "overall_severity": "HIGH" if detected_count >= 3 else
                                "MEDIUM" if detected_count >= 1 else "CLEAN"
        }

        return self.results
