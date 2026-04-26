# core/pdf_reporter.py


from reportlab.lib.pagesizes import A4               
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm                   
from reportlab.lib import colors                     
from reportlab.platypus import (
    SimpleDocTemplate,    
    Paragraph,            
    Spacer,               
    Table,                
    TableStyle,         
    HRFlowable,          
    PageBreak             
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT   
import io                                            
from datetime import datetime                        


class PDFReporter:
  

    def __init__(self, full_report: dict, trust_score_result: dict):
       
        self.report = full_report
        self.trust = trust_score_result

        
        self.styles = getSampleStyleSheet()

       
        self._define_styles()

        
        self.story = []

    def _define_styles(self):
        
      
        self.title_style = ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            textColor=colors.HexColor('#1a2240'), 
            spaceAfter=12,
            alignment=TA_CENTER
        )

        
        self.header_style = ParagraphStyle(
            name='CustomHeader',
            parent=self.styles['Heading1'],
            fontSize=14,
            textColor=colors.HexColor('#0066cc'),
            spaceBefore=16,
            spaceAfter=8
        )

        
        self.subheader_style = ParagraphStyle(
            name='CustomSubHeader',
            parent=self.styles['Heading2'],
            fontSize=11,
            textColor=colors.HexColor('#444444'),
            spaceBefore=10,
            spaceAfter=4
        )

       
        self.body_style = ParagraphStyle(
            name='CustomBody',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#333333'),
            spaceAfter=6,
            leading=14    
        )

        
        self.verdict_style = ParagraphStyle(
            name='Verdict',
            parent=self.styles['Normal'],
            fontSize=11,
            textColor=colors.HexColor('#222222'),
            spaceAfter=8,
            leading=16,
            leftIndent=20   
        )

     
        self.score_style = ParagraphStyle(
            name='Score',
            parent=self.styles['Normal'],
            fontSize=48,
            alignment=TA_CENTER,
            spaceBefore=10,
            spaceAfter=10
        )

    def _spacer(self, height=0.3):
        
        self.story.append(Spacer(1, height * cm))

    def _divider(self):
       
        self.story.append(HRFlowable(
            width="100%",
            thickness=0.5,
            color=colors.HexColor('#cccccc')
        ))
        self._spacer(0.2)

    def _severity_color(self, severity: str):
      
        mapping = {
            "HIGH":    colors.HexColor('#ffcccc'),   
            "MEDIUM":  colors.HexColor('#fff3cc'),   
            "LOW":     colors.HexColor('#e8f5e9'),   
            "NONE":    colors.HexColor('#f0f0f0'),   
            "CLEAN":   colors.HexColor('#e8f5e9'),   
            "UNKNOWN": colors.HexColor('#e0e0ff'),   
        }
        return mapping.get(severity, colors.white)

    
    def _build_cover(self):
      
        
        self.story.append(Paragraph(
            "Clinical AI Failure Observatory",
            self.title_style
        ))
        self.story.append(Paragraph(
            "Model Diagnostic Report",
            self.styles['Heading2']
        ))
        self._spacer(0.3)
        self._divider()

      
        meta = self.report.get("meta", {})
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.story.append(Paragraph(
            f"Generated: {generated_at}", self.body_style
        ))
        self.story.append(Paragraph(
            f"Target column: <b>{meta.get('target_column', 'N/A')}</b>", self.body_style
        ))
        self.story.append(Paragraph(
            f"Dataset size: <b>{meta.get('n_rows', 'N/A'):,} rows</b> × "
            f"<b>{meta.get('n_columns', 'N/A')} columns</b>",
            self.body_style
        ))
        self._spacer(0.5)

      
        score = self.trust.get("score", 0)
        grade = self.trust.get("grade", "?")

        
        grade_colors = {
            "A": "#2e7d32",    
            "B": "#558b2f",    
            "C": "#f57f17",    
            "D": "#e65100",    
            "F": "#c62828",    
        }
        score_color = grade_colors.get(grade, "#333333")

        self.story.append(Paragraph(
            f'<font color="{score_color}"><b>{score}</b></font>',
            self.score_style
        ))
        self.story.append(Paragraph(
            f"Model Trust Score / 100 — Grade: {grade}",
            ParagraphStyle(
                name='ScoreLabel',
                parent=self.styles['Normal'],
                fontSize=13,
                alignment=TA_CENTER,
                textColor=colors.HexColor(score_color)
            )
        ))
        self._spacer(0.4)
        self._divider()

      
        verdict = self.trust.get("verdict", "")
        self.story.append(Paragraph("<b>Overall Assessment:</b>", self.subheader_style))
        self.story.append(Paragraph(verdict, self.verdict_style))

       
        self.story.append(PageBreak())

    
    def _build_noise_section(self):
        
        self.story.append(Paragraph("Section 1 — Noise Audit", self.header_style))
        self.story.append(Paragraph(
            "This section reports on data quality and corruption patterns detected "
            "in the uploaded dataset. Six corruption archetypes are checked.",
            self.body_style
        ))
        self._spacer(0.2)

        noise = self.report.get("noise_audit", {})

        
        table_data = [
            
            ["Corruption Check", "Severity", "Detail"]
        ]

        checks = {
            "missing_values":   "Missing Values (Sensor Dropout)",
            "outliers":         "Statistical Outliers",
            "duplicates":       "Duplicate Rows",
            "class_imbalance":  "Class Imbalance",
            "low_variance":     "Low-Variance Columns",
            "dtype_mismatches": "Data Type Mismatches"
        }

        for key, display_name in checks.items():
            check_data = noise.get(key, {})
            severity = check_data.get("severity", "N/A")

          
            if key == "missing_values":
                n = check_data.get("n_affected", 0)
                detail = f"{n} columns with missing values"
            elif key == "outliers":
                n = check_data.get("n_affected", 0)
                detail = f"{n} columns with outliers detected"
            elif key == "duplicates":
                n = check_data.get("n_duplicate_rows", 0)
                pct = check_data.get("pct_of_dataset", 0)
                detail = f"{n} duplicate rows ({pct}% of dataset)"
            elif key == "class_imbalance":
                ratio = check_data.get("imbalance_ratio", 1)
                detail = f"Imbalance ratio: {ratio:.1f}x"
            elif key == "low_variance":
                n = check_data.get("n_affected", 0)
                detail = f"{n} near-constant columns"
            elif key == "dtype_mismatches":
                n = check_data.get("n_affected", 0)
                detail = f"{n} columns stored as wrong type"
            else:
                detail = "N/A"

            table_data.append([display_name, severity, detail])

       
        table = Table(table_data, colWidths=[6*cm, 3*cm, 8*cm])

     
        table.setStyle(TableStyle([
          
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a2240')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),

            
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ALIGN', (1, 1), (1, -1), 'CENTER'),  
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dddddd')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1),
             [colors.white, colors.HexColor('#f8f9fa')]), 
        ]))

      
        for row_idx, (key, _) in enumerate(checks.items(), start=1):
            severity = noise.get(key, {}).get("severity", "NONE")
            bg_color = self._severity_color(severity)
           
            table.setStyle(TableStyle([
                ('BACKGROUND', (1, row_idx), (1, row_idx), bg_color)
            ]))

        self.story.append(table)

       
        breakdown = self.trust.get("breakdown", {}).get("noise_audit", {})
        penalty = breakdown.get("final_penalty", 0)
        self.story.append(Paragraph(
            f"<b>Score Impact:</b> −{penalty} points (maximum possible: −25)",
            self.body_style
        ))

        self.story.append(PageBreak())

 
    def _build_bias_variance_section(self):
       
        self.story.append(Paragraph("Section 2 — Bias-Variance Analysis", self.header_style))
        self.story.append(Paragraph(
            "Five models of increasing complexity were trained and evaluated. "
            "The gap between Train AUC and Test AUC reveals the bias-variance profile.",
            self.body_style
        ))
        self._spacer(0.2)

        bv = self.report.get("bias_variance", {})
        models = bv.get("models", {})

        table_data = [["Model", "Train AUC", "Test AUC", "Gap", "Diagnosis"]]

        for model_name, model_data in models.items():
            train_auc = model_data.get("train_auc", 0)
            test_auc = model_data.get("test_auc", 0)
            gap = model_data.get("gap", 0)
            diagnosis = model_data.get("diagnosis", "N/A")

            
            short_diag = diagnosis.split("—")[0].strip()   

            table_data.append([
                model_name,
                f"{train_auc:.4f}",
                f"{test_auc:.4f}",
                f"{gap:.4f}",
                short_diag
            ])

        table = Table(table_data, colWidths=[5.5*cm, 2.5*cm, 2.5*cm, 2*cm, 4.5*cm])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a2240')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dddddd')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1),
             [colors.white, colors.HexColor('#f8f9fa')]),
        ]))

        self.story.append(table)
        self._spacer(0.3)

        best = bv.get("best_model", "N/A")
        best_auc = bv.get("best_test_auc", 0)
        self.story.append(Paragraph(
            f"<b>Best performing model:</b> {best} (Test AUC: {best_auc:.4f})",
            self.body_style
        ))

        breakdown = self.trust.get("breakdown", {}).get("bias_variance", {})
        penalty = breakdown.get("final_penalty", 0)
        self.story.append(Paragraph(
            f"<b>Score Impact:</b> −{penalty} points (maximum possible: −25)",
            self.body_style
        ))

        self.story.append(PageBreak())

    def _build_leakage_section(self):
        
        self.story.append(Paragraph("Section 3 — Leakage Scanner", self.header_style))
        self.story.append(Paragraph(
            "Data leakage causes models to report inflated performance during development "
            "but fail catastrophically in production. Five leakage archetypes were tested.",
            self.body_style
        ))
        self._spacer(0.2)

        leakage = self.report.get("leakage_scan", {})

        sin_display_names = {
            "target_encoding_leak": "Sin 1: Target Encoding Before Split",
            "feature_from_target":  "Sin 2: Feature Derived from Target",
            "scaling_leak":         "Sin 3: Timestamp-Contaminated Scaling",
            "group_overlap":        "Sin 4: Group Overlap (Entity in Train & Test)",
            "duplicate_leakage":    "Sin 5: Duplicate ID Leakage"
        }

        table_data = [["Leakage Type", "Status", "Severity", "AUC Inflation"]]

        for key, display_name in sin_display_names.items():
            sin_data = leakage.get(key, {})
            detected = sin_data.get("detected", False)
            severity = sin_data.get("severity", "NONE")

          
            inflation = sin_data.get("auc_inflation", None)
            if inflation is not None:
                inflation_str = f"+{inflation:.4f}"
            else:
                inflation_str = "N/A"

            status_str = "⚠ DETECTED" if detected else "✓ CLEAN"

            table_data.append([display_name, status_str, severity, inflation_str])

        table = Table(table_data, colWidths=[6.5*cm, 2.5*cm, 2.5*cm, 3*cm])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a2240')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dddddd')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1),
             [colors.white, colors.HexColor('#f8f9fa')]),
        ]))

       
        for row_idx, (key, _) in enumerate(sin_display_names.items(), start=1):
            sin_data = leakage.get(key, {})
            detected = sin_data.get("detected", False)
            bg = colors.HexColor('#ffcccc') if detected else colors.HexColor('#e8f5e9')
            table.setStyle(TableStyle([
                ('BACKGROUND', (1, row_idx), (1, row_idx), bg)
            ]))

        self.story.append(table)
        self._spacer(0.3)

        summary = leakage.get("summary", {})
        sins_count = summary.get("total_sins_detected", 0)
        overall = summary.get("overall_severity", "CLEAN")

        self.story.append(Paragraph(
            f"<b>Total sins detected:</b> {sins_count}/5 — Overall severity: {overall}",
            self.body_style
        ))

        breakdown = self.trust.get("breakdown", {}).get("leakage_scan", {})
        penalty = breakdown.get("final_penalty", 0)
        self.story.append(Paragraph(
            f"<b>Score Impact:</b> −{penalty} points (maximum possible: −30)",
            self.body_style
        ))

        self.story.append(PageBreak())

    
    def _build_curve_section(self):
        
        self.story.append(Paragraph("Section 4 — Learning Curve Autopsy", self.header_style))

        curve = self.report.get("curve_autopsy", {})
        diagnosis = curve.get("diagnosis", "N/A")
        explanation = curve.get("explanation", "")
        recommendation = curve.get("recommendation", "")
        final_train = curve.get("final_train_auc", 0)
        final_val = curve.get("final_val_auc", 0)
        gap = curve.get("gap", 0)

        
        diag_colors = {
            "HEALTHY":      "#2e7d32",
            "DATA-STARVED": "#c62828",
            "OVER-COMPLEX": "#e65100",
            "LEAKY":        "#6a1b9a",
        }
        diag_color = diag_colors.get(diagnosis, "#333333")

        self.story.append(Paragraph(
            f'Diagnosis: <font color="{diag_color}"><b>{diagnosis}</b></font>',
            self.subheader_style
        ))
        self._spacer(0.2)

       
        summary_data = [
            ["Metric", "Value"],
            ["Final Train AUC", f"{final_train:.4f}"],
            ["Final Validation AUC", f"{final_val:.4f}"],
            ["Train-Validation Gap", f"{gap:.4f}"],
            ["Diagnosis", diagnosis],
        ]

        summary_table = Table(summary_data, colWidths=[7*cm, 5*cm])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a2240')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (1, 0), (1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dddddd')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1),
             [colors.white, colors.HexColor('#f8f9fa')]),
        ]))

        self.story.append(summary_table)
        self._spacer(0.3)

        
        self.story.append(Paragraph("<b>Explanation:</b>", self.subheader_style))
        self.story.append(Paragraph(explanation, self.body_style))
        self._spacer(0.2)
        self.story.append(Paragraph("<b>Recommendation:</b>", self.subheader_style))
        self.story.append(Paragraph(recommendation, self.body_style))

        breakdown = self.trust.get("breakdown", {}).get("curve_autopsy", {})
        penalty = breakdown.get("final_penalty", 0)
        self.story.append(Paragraph(
            f"<b>Score Impact:</b> −{penalty} points (maximum possible: −20)",
            self.body_style
        ))

        self.story.append(PageBreak())

   
    def _build_recommendations(self):
        
        self.story.append(Paragraph("Summary — Prioritized Recommendations", self.header_style))
        self.story.append(Paragraph(
            "The following actions are recommended based on the diagnostic findings, "
            "ordered by priority (highest impact first).",
            self.body_style
        ))
        self._spacer(0.3)

        recommendations = []

       
        leakage = self.report.get("leakage_scan", {})
        sins = leakage.get("summary", {}).get("total_sins_detected", 0)
        if sins > 0:
            recommendations.append({
                "priority": "🔴 CRITICAL",
                "area": "Data Leakage",
                "action": f"{sins} leakage sin(s) detected. Audit your preprocessing pipeline "
                          f"before any model evaluation results can be trusted."
            })

       
        noise = self.report.get("noise_audit", {})
        if noise.get("missing_values", {}).get("severity") == "HIGH":
            recommendations.append({
                "priority": "🟠 HIGH",
                "area": "Missing Data",
                "action": "High rate of missing values detected. Use domain-informed imputation "
                          "or collect additional data for affected columns."
            })

        if noise.get("class_imbalance", {}).get("severity") in ["HIGH", "MEDIUM"]:
            ratio = noise.get("class_imbalance", {}).get("imbalance_ratio", 1)
            recommendations.append({
                "priority": "🟠 HIGH",
                "area": "Class Imbalance",
                "action": f"Imbalance ratio of {ratio:.1f}x detected. Use SMOTE, class weights, "
                          f"or threshold optimization. Never report accuracy on imbalanced data."
            })

        
        bv = self.report.get("bias_variance", {})
        best_auc = bv.get("best_test_auc", 1.0)
        if best_auc < 0.70:
            recommendations.append({
                "priority": "🟠 HIGH",
                "area": "Model Performance",
                "action": f"Best test AUC is only {best_auc:.3f}. Consider feature engineering, "
                          f"domain expert input, or a more expressive model family."
            })

       
        curve = self.report.get("curve_autopsy", {})
        if curve.get("diagnosis") == "OVER-COMPLEX":
            recommendations.append({
                "priority": "🟡 MEDIUM",
                "area": "Overfitting",
                "action": curve.get("recommendation", "Reduce model complexity.")
            })
        elif curve.get("diagnosis") == "DATA-STARVED":
            recommendations.append({
                "priority": "🟡 MEDIUM",
                "area": "Data Volume",
                "action": curve.get("recommendation", "Collect more training data.")
            })

       
        if not recommendations:
            recommendations.append({
                "priority": "🟢 NONE",
                "area": "All Clear",
                "action": "No critical issues detected. Continue with standard monitoring "
                          "and periodic revalidation."
            })

      
        rec_data = [["Priority", "Area", "Recommended Action"]]
        for rec in recommendations:
            rec_data.append([rec["priority"], rec["area"], rec["action"]])

        rec_table = Table(rec_data, colWidths=[2.5*cm, 3.5*cm, 11*cm])
        rec_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a2240')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dddddd')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1),
             [colors.white, colors.HexColor('#f8f9fa')]),
            ('WORDWRAP', (2, 1), (2, -1), True),   
        ]))

        self.story.append(rec_table)
        self._spacer(0.5)
        self._divider()

        
        self.story.append(Paragraph(
            "Clinical AI Failure Observatory — enghamza-AI",
            ParagraphStyle(
                name='Footer',
                parent=self.styles['Normal'],
                fontSize=8,
                textColor=colors.grey,
                alignment=TA_CENTER
            )
        ))

   
    def generate(self) -> bytes:
     
      
        buffer = io.BytesIO()

      
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            leftMargin=2*cm,
            rightMargin=2*cm,
            topMargin=2*cm,
            bottomMargin=2*cm
        )

     
        self._build_cover()
        self._build_noise_section()
        self._build_bias_variance_section()
        self._build_leakage_section()
        self._build_curve_section()
        self._build_recommendations()

        
        doc.build(self.story)

        
        buffer.seek(0)

        
        return buffer.read()