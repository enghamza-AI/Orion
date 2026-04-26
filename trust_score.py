# core/trust_score.py



class TrustScoreEngine:
 

   
    SEVERITY_PENALTIES = {
        "HIGH":    15,   
        "MEDIUM":   8,   
        "LOW":      3,   
        "NONE":     0,   
        "CLEAN":    0,   
        "UNKNOWN":  5,   
    }

    def __init__(self, full_report: dict):
        
        self.report = full_report
        self.score = 100          
        self.penalties = {}       
        self.breakdown = {}       

   
    def _score_noise(self):
       
       
        noise = self.report.get("noise_audit", {})

        
        raw_penalty = 0
        deductions = {}

        
        missing_sev = noise.get("missing_values", {}).get("severity", "NONE")
        pts = self.SEVERITY_PENALTIES.get(missing_sev, 0)
        raw_penalty += pts
        deductions["missing_values"] = {"severity": missing_sev, "points": pts}

       
        outlier_sev = noise.get("outliers", {}).get("severity", "NONE")
        pts = self.SEVERITY_PENALTIES.get(outlier_sev, 0)
        raw_penalty += pts
        deductions["outliers"] = {"severity": outlier_sev, "points": pts}

       
        dupe_sev = noise.get("duplicates", {}).get("severity", "NONE")
        pts = self.SEVERITY_PENALTIES.get(dupe_sev, 0)
        raw_penalty += pts
        deductions["duplicates"] = {"severity": dupe_sev, "points": pts}

      
        imbalance_sev = noise.get("class_imbalance", {}).get("severity", "NONE")
        pts = self.SEVERITY_PENALTIES.get(imbalance_sev, 0)
        raw_penalty += pts
        deductions["class_imbalance"] = {"severity": imbalance_sev, "points": pts}

       
        low_var_sev = noise.get("low_variance", {}).get("severity", "NONE")
        pts = self.SEVERITY_PENALTIES.get(low_var_sev, 0)
        raw_penalty += pts
        deductions["low_variance"] = {"severity": low_var_sev, "points": pts}

        
        dtype_sev = noise.get("dtype_mismatches", {}).get("severity", "NONE")
        pts = self.SEVERITY_PENALTIES.get(dtype_sev, 0)
        raw_penalty += pts
        deductions["dtype_mismatches"] = {"severity": dtype_sev, "points": pts}

        
        final_penalty = min(raw_penalty, 25)

       
        self.score -= final_penalty

        
        self.breakdown["noise_audit"] = {
            "deductions": deductions,
            "raw_penalty": raw_penalty,
            "final_penalty": final_penalty,   
            "category_max": 25
        }

    
    def _score_bias_variance(self):
       
        bv = self.report.get("bias_variance", {})
        models = bv.get("models", {})

        raw_penalty = 0
        deductions = {}

        
        best_auc = bv.get("best_test_auc", 0.5)

        if best_auc < 0.60:
         
            auc_penalty = 20
            auc_note = "Best AUC below 0.60 — model cannot learn signal"
        elif best_auc < 0.70:
            auc_penalty = 12
            auc_note = "Best AUC below 0.70 — weak signal or poor features"
        elif best_auc < 0.80:
            auc_penalty = 6
            auc_note = "Best AUC below 0.80 — acceptable but room to improve"
        else:
            auc_penalty = 0
            auc_note = f"Best AUC {best_auc:.3f} — good performance"

        raw_penalty += auc_penalty
        deductions["best_auc"] = {"value": best_auc, "points": auc_penalty, "note": auc_note}

        
        worst_gap = 0
        worst_model = "N/A"
        for model_name, model_data in models.items():
            gap = model_data.get("gap", 0)
            if gap > worst_gap:
                worst_gap = gap
                worst_model = model_name

        if worst_gap > 0.20:
            gap_penalty = 10
            gap_note = f"{worst_model} shows severe overfit (gap={worst_gap:.3f})"
        elif worst_gap > 0.10:
            gap_penalty = 5
            gap_note = f"{worst_model} shows moderate overfit (gap={worst_gap:.3f})"
        else:
            gap_penalty = 0
            gap_note = "No severe overfit detected across models"

        raw_penalty += gap_penalty
        deductions["overfit_gap"] = {
            "worst_gap": worst_gap,
            "worst_model": worst_model,
            "points": gap_penalty,
            "note": gap_note
        }

        final_penalty = min(raw_penalty, 25)
        self.score -= final_penalty

        self.breakdown["bias_variance"] = {
            "deductions": deductions,
            "raw_penalty": raw_penalty,
            "final_penalty": final_penalty,
            "category_max": 25
        }

   
    def _score_leakage(self):
  
        leakage = self.report.get("leakage_scan", {})
        summary = leakage.get("summary", {})

        sins_detected = summary.get("total_sins_detected", 0)
        overall_severity = summary.get("overall_severity", "CLEAN")

        
        raw_penalty = sins_detected * 6
        final_penalty = min(raw_penalty, 30)

        self.score -= final_penalty

        
        sin_details = {}
        sin_names = [
            "target_encoding_leak",
            "feature_from_target",
            "scaling_leak",
            "group_overlap",
            "duplicate_leakage"
        ]

        for sin in sin_names:
            sin_data = leakage.get(sin, {})
            sin_details[sin] = {
                "detected": sin_data.get("detected", False),
                "severity": sin_data.get("severity", "NONE")
            }

        self.breakdown["leakage_scan"] = {
            "sins_detected": sins_detected,
            "sin_details": sin_details,
            "overall_severity": overall_severity,
            "raw_penalty": raw_penalty,
            "final_penalty": final_penalty,
            "category_max": 30
        }

    
    def _score_curve_autopsy(self):
       
        curve = self.report.get("curve_autopsy", {})
        diagnosis = curve.get("diagnosis", "HEALTHY")
        final_val_auc = curve.get("final_val_auc", 0.5)
        gap = curve.get("gap", 0)

       
        diagnosis_penalties = {
            "LEAKY":        20,   
            "OVER-COMPLEX": 15,   
            "DATA-STARVED": 10,   
            "HEALTHY":       0,   
        }

        base_penalty = diagnosis_penalties.get(diagnosis, 10)

       
        if final_val_auc < 0.60:
            auc_bonus_penalty = 5
        else:
            auc_bonus_penalty = 0

        raw_penalty = base_penalty + auc_bonus_penalty
        final_penalty = min(raw_penalty, 20)

        self.score -= final_penalty

        self.breakdown["curve_autopsy"] = {
            "diagnosis": diagnosis,
            "final_val_auc": final_val_auc,
            "gap": gap,
            "base_penalty": base_penalty,
            "auc_bonus_penalty": auc_bonus_penalty,
            "final_penalty": final_penalty,
            "category_max": 20
        }

  
    def _assign_grade(self):
        
        
        self.score = max(0, min(100, self.score))

        if self.score >= 90:
            grade = "A"
            color = "green"
            verdict = (
                "This model demonstrates strong data quality and generalization. "
                "Suitable for deployment with standard monitoring in place."
            )
        elif self.score >= 75:
            grade = "B"
            color = "lightgreen"
            verdict = (
                "Mostly trustworthy. Minor data quality issues detected. "
                "Address flagged items before production deployment."
            )
        elif self.score >= 60:
            grade = "C"
            color = "orange"
            verdict = (
                "Caution required. Meaningful problems detected in data quality "
                "or model behavior. Not recommended for high-stakes decisions."
            )
        elif self.score >= 40:
            grade = "D"
            color = "darkorange"
            verdict = (
                "Significant issues detected. Multiple engine findings indicate "
                "unreliable performance. Substantial rework required before deployment."
            )
        else:
            grade = "F"
            color = "red"
            verdict = (
                "Do not deploy. Critical failures detected across multiple diagnostic "
                "engines. The reported performance metrics cannot be trusted. "
                "Restart with a clean data pipeline."
            )

        return grade, color, verdict

   
    def compute(self) -> dict:
       
       
        self._score_noise()           
        self._score_bias_variance()   
        self._score_leakage()        
        self._score_curve_autopsy()   

        
        grade, color, verdict = self._assign_grade()

        return {
            "score": self.score,           
            "grade": grade,                 
            "color": color,                
            "verdict": verdict,             
            "breakdown": self.breakdown,     
            "max_possible": 100
        }