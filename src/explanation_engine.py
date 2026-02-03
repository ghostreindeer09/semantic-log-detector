"""
Explanation Engine Module - Generate human-readable explanations.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class Explanation:
    """Human-readable explanation for an anomaly."""
    log_id: str
    anomaly_score: float
    summary: str
    details: List[str]
    similar_logs: List[Dict]
    recommendations: List[str]


class ExplanationEngine:
    """Generates explanations for detected anomalies."""
    
    SEVERITY_LEVELS = {
        (0.0, 0.3): ("Low", "This log shows minor deviation from normal patterns."),
        (0.3, 0.6): ("Medium", "This log shows moderate deviation from established patterns."),
        (0.6, 0.8): ("High", "This log shows significant deviation that warrants investigation."),
        (0.8, 1.0): ("Critical", "This log is highly anomalous and requires immediate attention.")
    }
    
    def __init__(self, vector_db=None, preprocessor=None):
        self.vector_db = vector_db
        self.preprocessor = preprocessor
    
    def get_severity(self, score: float) -> tuple:
        """Get severity level and description."""
        for (low, high), (level, desc) in self.SEVERITY_LEVELS.items():
            if low <= score < high:
                return level, desc
        return "Critical", "Extremely anomalous pattern detected."
    
    def explain(
        self,
        log_id: str,
        log_text: str,
        anomaly_score: float,
        neighbor_ids: List[str],
        distances: List[float]
    ) -> Explanation:
        """Generate explanation for a single anomaly."""
        severity, severity_desc = self.get_severity(anomaly_score)
        
        # Build similar logs info
        similar_logs = []
        for i, (nid, dist) in enumerate(zip(neighbor_ids[:3], distances[:3])):
            meta = self.vector_db.get_metadata(nid) if self.vector_db else {}
            similar_logs.append({
                'id': nid, 'distance': dist, 'rank': i + 1,
                'text': meta.get('cleaned_text', 'N/A')[:100]
            })
        
        # Build details
        details = [
            f"Severity Level: {severity}",
            f"Anomaly Score: {anomaly_score:.2%}",
            f"Mean Distance to Neighbors: {sum(distances)/len(distances):.4f}" if distances else "No neighbors found"
        ]
        
        # Build recommendations
        recommendations = self._generate_recommendations(severity, log_text)
        
        summary = f"{severity} severity anomaly detected. {severity_desc}"
        
        return Explanation(
            log_id=log_id, anomaly_score=anomaly_score,
            summary=summary, details=details,
            similar_logs=similar_logs, recommendations=recommendations
        )
    
    def _generate_recommendations(self, severity: str, log_text: str) -> List[str]:
        """Generate recommendations based on severity and content."""
        base_recs = []
        
        if severity in ["Critical", "High"]:
            base_recs.extend([
                "Investigate this log immediately",
                "Check related system components",
                "Review recent changes in the affected service"
            ])
        elif severity == "Medium":
            base_recs.extend([
                "Monitor for recurring patterns",
                "Add to watchlist for future occurrences"
            ])
        else:
            base_recs.append("Continue monitoring; no immediate action required")
        
        # Content-specific recommendations
        log_lower = log_text.lower()
        if "error" in log_lower or "fail" in log_lower:
            base_recs.append("Check error handling and recovery mechanisms")
        if "timeout" in log_lower:
            base_recs.append("Review network connectivity and service dependencies")
        if "memory" in log_lower or "cpu" in log_lower:
            base_recs.append("Monitor resource utilization trends")
        if "auth" in log_lower or "login" in log_lower:
            base_recs.append("Review authentication logs for security concerns")
        
        return base_recs
    
    def batch_explain(
        self,
        anomaly_scores: List,
        log_texts: Dict[str, str]
    ) -> List[Explanation]:
        """Generate explanations for multiple anomalies."""
        explanations = []
        for score in anomaly_scores:
            if score.is_anomaly:
                text = log_texts.get(score.log_id, "")
                exp = self.explain(
                    score.log_id, text, score.score,
                    score.nearest_neighbors, score.distances
                )
                explanations.append(exp)
        return explanations
    
    def format_explanation(self, explanation: Explanation) -> str:
        """Format explanation as readable text."""
        lines = [
            f"╔══ ANOMALY REPORT ══╗",
            f"Log ID: {explanation.log_id}",
            f"Score: {explanation.anomaly_score:.2%}",
            f"\n{explanation.summary}",
            f"\n--- Details ---"
        ]
        lines.extend([f"  • {d}" for d in explanation.details])
        
        if explanation.similar_logs:
            lines.append(f"\n--- Similar Logs ---")
            for sim in explanation.similar_logs:
                lines.append(f"  [{sim['rank']}] {sim['id']} (dist: {sim['distance']:.4f})")
        
        lines.append(f"\n--- Recommendations ---")
        lines.extend([f"  → {r}" for r in explanation.recommendations])
        lines.append(f"╚════════════════════╝")
        
        return "\n".join(lines)
