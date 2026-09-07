from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from reportlab.lib.colors import HexColor, black
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer

OUTPUT_DIR = Path("output")
ORANGE = HexColor("#E67E22")


class ReportService:
    """Generate UI PDFs: detailed conversation analysis report + SOAP note."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_reports(self, result: dict[str, Any]) -> dict[str, str]:
        analysis = self.save_analysis_report(result)
        soap = self.save_soap_note(result)
        return {**analysis, **soap}

    def save_analysis_report(self, result: dict[str, Any]) -> dict[str, str]:
        visit_id = result.get("visit_id") or "visit"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{stamp}_{visit_id}_analysis_report.pdf"
        path = self.output_dir / filename

        styles = self._styles()
        doc = SimpleDocTemplate(
            str(path),
            pagesize=A4,
            leftMargin=0.75 * inch,
            rightMargin=0.75 * inch,
            topMargin=0.7 * inch,
            bottomMargin=0.7 * inch,
        )
        story = []
        insights = result.get("report_insights") or {}

        story.append(Paragraph("Medical Conversation Analysis Report", styles["title"]))
        story.append(
            Paragraph(
                f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                styles["body"],
            )
        )
        story.append(Paragraph(f"Visit ID: {self._escape(visit_id)}", styles["body"]))
        if result.get("primary_guideline_used"):
            story.append(
                Paragraph(
                    f"Primary guideline: {self._escape(result['primary_guideline_used'])}",
                    styles["body"],
                )
            )
        story.append(Spacer(1, 8))

        story.append(Paragraph("Conversation History", styles["heading"]))
        blocks = self._conversation_history_blocks(result)
        if blocks:
            for role, text in blocks:
                story.append(
                    Paragraph(
                        f"<b>{self._escape(role)}:</b> {self._escape(text)}",
                        styles["body"],
                    )
                )
        else:
            story.append(Paragraph("No conversation available.", styles["body"]))

        story.append(Paragraph("Analysis and Insights", styles["heading"]))

        story.append(Paragraph("Key Points", styles["subheading"]))
        story.extend(self._bullet_list(insights.get("key_points") or self._fallback_key_points(result), styles["body"]))

        story.append(Paragraph("Action Items", styles["subheading"]))
        story.extend(self._bullet_list(insights.get("action_items") or self._fallback_action_items(result), styles["body"]))

        story.append(Paragraph("Patient Concerns", styles["subheading"]))
        story.extend(self._bullet_list(insights.get("patient_concerns") or [], styles["body"]))

        story.append(Paragraph("Medical Terms", styles["subheading"]))
        story.extend(self._bullet_list(insights.get("medical_terms") or [], styles["body"]))

        story.append(Paragraph("Follow Up Questions", styles["subheading"]))
        story.extend(self._bullet_list(insights.get("follow_up_questions") or [], styles["body"]))

        story.append(Paragraph("Symptoms", styles["subheading"]))
        story.extend(self._bullet_list(insights.get("symptoms") or [], styles["body"]))

        story.append(Paragraph("Medications", styles["subheading"]))
        story.extend(self._bullet_list(insights.get("medications") or [], styles["body"]))

        story.append(Paragraph("Diagnosis Considerations", styles["subheading"]))
        story.extend(self._bullet_list(insights.get("diagnosis_considerations") or [], styles["body"]))

        story.append(Paragraph("Risk Factors", styles["subheading"]))
        story.extend(self._bullet_list(insights.get("risk_factors") or [], styles["body"]))

        story.append(Paragraph("Differential Diagnosis", styles["subheading"]))
        ranked = result.get("differential_diagnosis") or []
        if ranked:
            for dx in ranked:
                name = dx.get("name", "")
                conf = dx.get("confidence")
                pct = self._to_percent(conf)
                label = f"{name} ({pct}%)" if pct is not None else name
                story.append(Paragraph(f"• {self._escape(label)}", styles["body"]))
                if dx.get("reason"):
                    story.append(
                        Paragraph(
                            f"&nbsp;&nbsp;&nbsp;Rationale: {self._escape(dx['reason'])}",
                            styles["body"],
                        )
                    )
        else:
            story.extend(
                self._bullet_list(insights.get("differential_diagnosis_list") or [], styles["body"])
            )

        doc.build(story)
        return {
            "report_pdf": str(path),
            "report_filename": filename,
        }

    def save_soap_note(self, result: dict[str, Any]) -> dict[str, str]:
        visit_id = result.get("visit_id") or "visit"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{stamp}_{visit_id}_soap_note.pdf"
        path = self.output_dir / filename

        styles = self._styles()
        doc = SimpleDocTemplate(
            str(path),
            pagesize=A4,
            leftMargin=0.85 * inch,
            rightMargin=0.85 * inch,
            topMargin=0.8 * inch,
            bottomMargin=0.8 * inch,
        )
        story = []
        insights = result.get("report_insights") or {}
        soap = insights.get("soap") or {}

        story.append(Paragraph("SOAP Note", styles["soap_title"]))
        story.append(Spacer(1, 18))

        for label in ("Subjective", "Objective", "Assessment", "Plan"):
            key = label.lower()
            value = soap.get(key) or "Not documented"
            story.append(Paragraph(label, styles["soap_section"]))
            story.append(Paragraph(self._escape(str(value)), styles["soap_body"]))
            story.append(Spacer(1, 14))

        story.append(Paragraph("Differential Diagnosis Suggestions", styles["soap_dx_title"]))
        story.append(
            Paragraph(
                "Supportive suggestions only; requires clinician verification and approval.",
                styles["soap_disclaimer"],
            )
        )
        story.append(Spacer(1, 8))

        diagnoses = result.get("differential_diagnosis") or []
        if diagnoses:
            for index, dx in enumerate(diagnoses[:5], start=1):
                name = dx.get("name", f"Diagnosis {index}")
                pct = self._to_percent(dx.get("confidence"))
                if pct is None:
                    pct = self._to_percent(dx.get("llm_confidence")) or 0
                story.append(
                    Paragraph(
                        f"<b>{index}. {self._escape(name)} ({pct}%)</b>",
                        styles["soap_body"],
                    )
                )
                rationale = dx.get("reason") or ""
                if not rationale:
                    criteria = dx.get("criteria_matched") or []
                    rationale = ", ".join(str(c) for c in criteria) if criteria else "Based on encounter findings"
                story.append(
                    Paragraph(
                        f"Rationale: {self._escape(rationale)}",
                        styles["soap_body"],
                    )
                )
                story.append(Spacer(1, 8))
        else:
            story.append(Paragraph("No differential suggestions available.", styles["soap_body"]))

        story.append(Spacer(1, 28))
        story.append(
            Paragraph(
                "<b>Clinician signature:</b> ________________________________",
                styles["soap_body"],
            )
        )

        doc.build(story)
        return {
            "soap_pdf": str(path),
            "soap_filename": filename,
        }

    def _styles(self) -> dict:
        base = getSampleStyleSheet()
        return {
            "title": ParagraphStyle(
                "ReportTitle",
                parent=base["Heading1"],
                fontSize=16,
                textColor=black,
                spaceAfter=6,
                alignment=TA_LEFT,
            ),
            "heading": ParagraphStyle(
                "ReportHeading",
                parent=base["Heading2"],
                fontSize=12,
                textColor=black,
                spaceBefore=14,
                spaceAfter=6,
            ),
            "subheading": ParagraphStyle(
                "ReportSubHeading",
                parent=base["Heading3"],
                fontSize=11,
                textColor=black,
                spaceBefore=10,
                spaceAfter=4,
            ),
            "body": ParagraphStyle(
                "ReportBody",
                parent=base["BodyText"],
                fontSize=10,
                leading=14,
                spaceAfter=3,
            ),
            "soap_title": ParagraphStyle(
                "SoapTitle",
                parent=base["Heading1"],
                fontName="Helvetica-Bold",
                fontSize=22,
                textColor=black,
                spaceAfter=10,
            ),
            "soap_section": ParagraphStyle(
                "SoapSection",
                parent=base["Heading2"],
                fontName="Helvetica-Bold",
                fontSize=12,
                textColor=ORANGE,
                spaceBefore=4,
                spaceAfter=4,
            ),
            "soap_body": ParagraphStyle(
                "SoapBody",
                parent=base["BodyText"],
                fontName="Helvetica",
                fontSize=10.5,
                leading=15,
                textColor=black,
                spaceAfter=2,
            ),
            "soap_dx_title": ParagraphStyle(
                "SoapDxTitle",
                parent=base["Heading2"],
                fontName="Helvetica-Bold",
                fontSize=13,
                textColor=black,
                spaceBefore=8,
                spaceAfter=4,
            ),
            "soap_disclaimer": ParagraphStyle(
                "SoapDisclaimer",
                parent=base["BodyText"],
                fontName="Helvetica-Oblique",
                fontSize=9,
                textColor=ORANGE,
                spaceAfter=6,
            ),
        }

    def _conversation_history_blocks(self, result: dict[str, Any]) -> list:
        conversation = (
            result.get("doctor_patient_conversation")
            or result.get("conversation")
            or []
        )
        if not conversation:
            return []

        blocks = []
        current_role = ""
        current_parts = []

        for turn in conversation:
            if not isinstance(turn, dict):
                continue
            role = self._normalize_role(str(turn.get("role") or turn.get("speaker") or "Speaker"))
            text = (turn.get("text") or "").strip()
            if not text:
                continue
            if role == current_role:
                current_parts.append(text)
            else:
                if current_role and current_parts:
                    blocks.append((current_role, self._join_utterances(current_parts)))
                current_role = role
                current_parts = [text]

        if current_role and current_parts:
            blocks.append((current_role, self._join_utterances(current_parts)))
        return blocks

    def _fallback_key_points(self, result: dict[str, Any]) -> list[str]:
        points = []
        summary = result.get("summary") or {}
        if isinstance(summary, dict):
            if summary.get("chief_complaint"):
                points.append(f"Chief complaint: {summary['chief_complaint']}")
            if summary.get("assessment"):
                points.append(f"Assessment: {summary['assessment']}")
        entities = result.get("entities") or {}
        if isinstance(entities, dict):
            points.extend(str(s) for s in (entities.get("symptoms") or []))
        diagnoses = result.get("differential_diagnosis") or []
        if diagnoses:
            top = diagnoses[0]
            points.append(
                f"Top diagnosis: {top.get('name', '')} (confidence {top.get('confidence', '')})"
            )
        return points or ["Analysis completed."]

    def _fallback_action_items(self, result: dict[str, Any]) -> list[str]:
        items = []
        diagnoses = result.get("differential_diagnosis") or []
        if diagnoses:
            if diagnoses[0].get("recommended_plan"):
                items.append(str(diagnoses[0]["recommended_plan"]))
            for missing in diagnoses[0].get("missing_information") or []:
                items.append(f"Obtain: {missing}")
        return items or ["Review findings with clinician.", "Schedule follow-up as needed."]

    def _bullet_list(self, items: list[Any], body_style: ParagraphStyle) -> list:
        cleaned = [str(item).strip() for item in (items or []) if str(item).strip()]
        if not cleaned:
            return [Paragraph("• None documented.", body_style)]
        flowables = []
        for item in cleaned:
            flowables.append(
                ListItem(Paragraph(self._escape(item), body_style), leftIndent=10)
            )
        return [ListFlowable(flowables, bulletType="bullet", start="•")]

    @staticmethod
    def _to_percent(value: Any) -> Optional[int]:
        if value is None or value == "":
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if number <= 1.0:
            number *= 100
        return round(number)

    @staticmethod
    def _normalize_role(role: str) -> str:
        lowered = role.strip().lower()
        if "doctor" in lowered or "physician" in lowered or "clinician" in lowered:
            return "doctor"
        if "patient" in lowered:
            return "patient"
        if lowered.startswith("speaker 0") or lowered.endswith("0"):
            return "patient"
        if lowered.startswith("speaker 1") or lowered.endswith("1"):
            return "doctor"
        return lowered or "speaker"

    @staticmethod
    def _join_utterances(parts: list[str]) -> str:
        merged = " ".join(part.strip() for part in parts if part and part.strip())
        return " ".join(merged.split())

    @staticmethod
    def _escape(text: str) -> str:
        return (
            str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
