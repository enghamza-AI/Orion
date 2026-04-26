# engines/curve_autopsy.py


import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')


class CurveAutopsy:

   
    MIN_ROWS_REQUIRED = 80

    def __init__(self, df: pd.DataFrame, target_col: str, sample_size: int = 5000):
        self.df = df.copy()
        self.target_col = target_col
        self.sample_size = sample_size
        self.results = {}

    def _safe_fallback(self, reason: str) -> dict:
       
        return {
            "curve_data": {
                "train_sizes":        [0],
                "train_scores_mean":  [0.5],
                "train_scores_std":   [0.0],
                "val_scores_mean":    [0.5],
                "val_scores_std":     [0.0],
            },
            "diagnosis":        "DATA-STARVED",
            "color":            "red",
            "explanation":      f"Learning curve could not be computed: {reason}",
            "recommendation":   "Check that your dataset has sufficient rows, "
                                "a valid binary target column, and numeric features.",
            "final_train_auc":  0.5,
            "final_val_auc":    0.5,
            "gap":              0.0,
            "val_still_rising": False,
            "skipped":          True,  
            "skip_reason":      reason
        }

    def _prepare(self):
       

     
        if self.df is None or len(self.df) == 0:
            raise ValueError("Dataset is empty.")

       
        if self.target_col not in self.df.columns:
            raise ValueError(
                f"Target column '{self.target_col}' not found in dataset. "
                f"Available columns: {list(self.df.columns[:10])}"
            )

       
        if len(self.df) > self.sample_size:
            df = self.df.sample(n=self.sample_size, random_state=42)
        else:
            df = self.df.copy()

      
        if len(df) < self.MIN_ROWS_REQUIRED:
            raise ValueError(
                f"Dataset has only {len(df)} rows after sampling. "
                f"Minimum required: {self.MIN_ROWS_REQUIRED}."
            )

     
        df = df.dropna(subset=[self.target_col])

        if len(df) < self.MIN_ROWS_REQUIRED:
            raise ValueError(
                f"After dropping rows with missing target, only {len(df)} rows remain. "
                f"Minimum required: {self.MIN_ROWS_REQUIRED}."
            )

       
        X_raw = df.drop(columns=[self.target_col])
        y_raw = df[self.target_col]

      
        X_numeric = X_raw.select_dtypes(include=[np.number])

        if X_numeric.shape[1] == 0:
            raise ValueError(
                "No numeric feature columns found. "
                "The engine requires at least one numeric feature column."
            )

     
        all_nan_cols = X_numeric.columns[X_numeric.isnull().all()].tolist()
        if all_nan_cols:
            X_numeric = X_numeric.drop(columns=all_nan_cols)

        if X_numeric.shape[1] == 0:
            raise ValueError(
                "All numeric columns are entirely NaN. "
                "No features available for training."
            )

      
        std_vals = X_numeric.std()
        constant_cols = std_vals[std_vals == 0].index.tolist()
        if constant_cols:
            X_numeric = X_numeric.drop(columns=constant_cols)

        if X_numeric.shape[1] == 0:
            raise ValueError(
                "All numeric columns are constant (zero variance). "
                "No informative features available."
            )

   
        imputer = SimpleImputer(strategy='median')
        X_imp = imputer.fit_transform(X_numeric)

       
        if not np.isfinite(X_imp).all():
            
            X_imp = np.nan_to_num(X_imp, nan=0.0, posinf=0.0, neginf=0.0)

       
        le = LabelEncoder()
        try:
            y_enc = le.fit_transform(y_raw.astype(str))
        except Exception as e:
            raise ValueError(f"Could not encode target column: {str(e)}")

        n_classes = len(np.unique(y_enc))

        if n_classes < 2:
            raise ValueError(
                f"Target column has only 1 unique value after encoding. "
                f"Need at least 2 classes for classification."
            )

        if n_classes > 2:
          
            unique_vals, counts = np.unique(y_enc, return_counts=True)
            minority_class = unique_vals[np.argmin(counts)]
            y_enc = (y_enc == minority_class).astype(int)

        
        unique_vals, counts = np.unique(y_enc, return_counts=True)
        minority_count = counts.min()
        majority_count = counts.max()

        
        MIN_MINORITY = 15

        if minority_count < MIN_MINORITY:
            
            minority_mask = (y_enc == unique_vals[np.argmin(counts)])
            X_minority = X_imp[minority_mask]
            y_minority = y_enc[minority_mask]

            X_majority = X_imp[~minority_mask]
            y_majority = y_enc[~minority_mask]

          
            X_min_up, y_min_up = resample(
                X_minority, y_minority,
                replace=True,
                n_samples=MIN_MINORITY,
                random_state=42
            )

            X_imp  = np.vstack([X_majority, X_min_up])
            y_enc  = np.concatenate([y_majority, y_min_up])

     
        if len(y_enc) < self.MIN_ROWS_REQUIRED:
            raise ValueError(
                f"After all preprocessing only {len(y_enc)} samples remain. "
                f"Cannot run learning curve with fewer than {self.MIN_ROWS_REQUIRED} samples."
            )

        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imp)

      
        if not np.isfinite(X_scaled).all():
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        return X_scaled, y_enc

    def run(self) -> dict:
        

     
        try:
            X, y = self._prepare()
        except ValueError as e:
           
            return self._safe_fallback(str(e))
        except Exception as e:
            return self._safe_fallback(f"Unexpected preparation error: {str(e)}")

        
        minority_count = np.unique(y, return_counts=True)[1].min()
        cv_folds = min(5, minority_count // 2)
        cv_folds = max(2, cv_folds)

     
        min_train_fraction = max(0.1, (cv_folds * 4) / len(y))
       
        min_train_fraction = min(min_train_fraction, 0.5)

        train_sizes = np.linspace(min_train_fraction, 1.0, 8)

     
        if len(y) < 200:
            model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        else:
            model = GradientBoostingClassifier(n_estimators=50, random_state=42)

        
        try:
            train_sizes_abs, train_scores, val_scores = learning_curve(
                model, X, y,
                train_sizes=train_sizes,
                cv=cv_folds,
                scoring='roc_auc',
                n_jobs=-1,
                random_state=42,
                error_score=0.5   
            )
        except Exception as e:
            return self._safe_fallback(
                f"Learning curve computation failed: {str(e)}"
            )

    
        train_scores_mean = train_scores.mean(axis=1)
        train_scores_std  = train_scores.std(axis=1)
        val_scores_mean   = val_scores.mean(axis=1)
        val_scores_std    = val_scores.std(axis=1)

       
        final_train = float(train_scores_mean[-1])
        final_val   = float(val_scores_mean[-1])
        gap         = final_train - final_val

       
        val_still_rising = bool(val_scores_mean[-1] > val_scores_mean[-2]) \
                           if len(val_scores_mean) >= 2 else False

        if final_val < 0.65 and gap < 0.08:
            diagnosis      = "DATA-STARVED"
            color          = "red"
            explanation    = (
                f"Both training AUC ({final_train:.3f}) and validation AUC "
                f"({final_val:.3f}) are low and converging. The model has hit "
                f"a ceiling — more data will not help. The features or model "
                f"architecture need rethinking."
            )
            recommendation = (
                "Try feature engineering, domain knowledge features, "
                "or a fundamentally different model family."
            )

        elif gap > 0.15:
            diagnosis      = "OVER-COMPLEX"
            color          = "orange"
            explanation    = (
                f"Training AUC ({final_train:.3f}) is significantly higher than "
                f"validation AUC ({final_val:.3f}). Gap of {gap:.3f} indicates "
                f"the model is memorizing training data rather than learning patterns."
            )
            recommendation = (
                "Reduce model complexity, increase regularization, "
                "or collect more diverse training data."
            )

        elif final_val > final_train + 0.02:
            diagnosis      = "LEAKY"
            color          = "purple"
            explanation    = (
                f"Validation AUC ({final_val:.3f}) exceeds training AUC "
                f"({final_train:.3f}). This is statistically unusual and strongly "
                f"suggests data leakage in the preprocessing pipeline."
            )
            recommendation = (
                "Audit your preprocessing pipeline. Check scaling, encoding, "
                "and feature derivation steps for test-set contamination."
            )

        else:
            diagnosis      = "HEALTHY"
            color          = "green"
            explanation    = (
                f"Training AUC ({final_train:.3f}) and validation AUC "
                f"({final_val:.3f}) are both strong with a small gap ({gap:.3f}). "
                f"The model generalizes well to unseen data."
            )
            recommendation = (
                "Model looks healthy. Consider hyperparameter tuning "
                "for marginal gains."
            )

        self.results = {
            "curve_data": {
                "train_sizes":        train_sizes_abs.tolist(),
                "train_scores_mean":  train_scores_mean.tolist(),
                "train_scores_std":   train_scores_std.tolist(),
                "val_scores_mean":    val_scores_mean.tolist(),
                "val_scores_std":     val_scores_std.tolist(),
            },
            "diagnosis":        diagnosis,
            "color":            color,
            "explanation":      explanation,
            "recommendation":   recommendation,
            "final_train_auc":  round(final_train, 4),
            "final_val_auc":    round(final_val, 4),
            "gap":              round(gap, 4),
            "val_still_rising": val_still_rising,
            "cv_folds_used":    cv_folds,       
            "skipped":          False
        }

        return self.results