# engines/bias_variance_engine.py


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


class BiasVarianceEngine:

    def __init__(self, df: pd.DataFrame, target_col: str, sample_size: int = 5000):
        self.df = df.copy()
        self.target_col = target_col
        self.sample_size = sample_size
        self.results = {}

    def _prepare_data(self):
     
        if len(self.df) > self.sample_size:
            df = self.df.sample(n=self.sample_size, random_state=42)
        else:
            df = self.df.copy()

        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]

       
        X = X.select_dtypes(include=[np.number])

       
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)

      
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

       
        n_classes = len(np.unique(y_encoded))
        if n_classes > 2:
          
            unique, counts = np.unique(y_encoded, return_counts=True)
            minority_class = unique[np.argmin(counts)]
            
            y_encoded = (y_encoded == minority_class).astype(int)

        
        unique, counts = np.unique(y_encoded, return_counts=True)
        min_count = counts.min()

        if min_count < 10:
        
            self._use_stratify = False
        else:
            self._use_stratify = True

      
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)

        return X_scaled, y_encoded

    def run(self) -> dict:
        X, y = self._prepare_data()

        
        if self._use_stratify:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

        models = {
            "Logistic Regression":      LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree (depth=3)":  DecisionTreeClassifier(max_depth=3, random_state=42),
            "Decision Tree (depth=10)": DecisionTreeClassifier(max_depth=10, random_state=42),
            "Random Forest":            RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
            "Gradient Boosting":        GradientBoostingClassifier(n_estimators=50, random_state=42),
        }

        model_results = {}

        for name, model in models.items():
            try:
                model.fit(X_train, y_train)

                train_proba = model.predict_proba(X_train)

               
                if train_proba.shape[1] == 2:
                    
                    train_auc = roc_auc_score(y_train, train_proba[:, 1])
                    test_proba = model.predict_proba(X_test)[:, 1]
                    test_auc   = roc_auc_score(y_test, test_proba)
                else:
                   
                    train_auc = roc_auc_score(
                        y_train, train_proba,
                        multi_class='ovr', average='macro'
                    )
                    test_proba = model.predict_proba(X_test)
                    test_auc   = roc_auc_score(
                        y_test, test_proba,
                        multi_class='ovr', average='macro'
                    )

                gap = train_auc - test_auc

                if test_auc < 0.65:
                    diagnosis = "UNDERFIT — High Bias"
                    recommendation = "Model too simple. Try more complex model or add features."
                elif gap > 0.10:
                    diagnosis = "OVERFIT — High Variance"
                    recommendation = f"Gap of {gap:.2f}. Add regularization or more data."
                else:
                    diagnosis = "BALANCED"
                    recommendation = "Good generalization."

                model_results[name] = {
                    "train_auc":      round(train_auc, 4),
                    "test_auc":       round(test_auc, 4),
                    "gap":            round(gap, 4),
                    "diagnosis":      diagnosis,
                    "recommendation": recommendation
                }

            except Exception as e:
               
                model_results[name] = {
                    "train_auc":      0.0,
                    "test_auc":       0.0,
                    "gap":            0.0,
                    "diagnosis":      f"ERROR — {str(e)[:80]}",
                    "recommendation": "Check data quality for this model."
                }

       
        valid_models = {
            k: v for k, v in model_results.items()
            if not v["diagnosis"].startswith("ERROR")
        }

        if valid_models:
            best_model = max(valid_models, key=lambda k: valid_models[k]["test_auc"])
        else:
            best_model = list(model_results.keys())[0]

        self.results = {
            "models":          model_results,
            "best_model":      best_model,
            "best_test_auc":   model_results[best_model]["test_auc"],
            "n_features_used": X.shape[1],
            "n_samples_used":  X.shape[0]
        }

        return self.results

        

