import uuid
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from datetime import datetime
from .severity_estimator import SeverityReport
from .incident_classifier import IncidentClassification

logger = logging.getLogger(__name__)

@dataclass
class AlertPacket:
    """Structured Emergency Alert Packet."""
    incident_id: str
    timestamp: str
    incident_type: str
    severity_level: int
    estimated_origin: str
    propagation_path: List[str]
    affected_zones: List[str]
    crowd_estimate: int
    recommended_action: str
    confidence_score: float
    description: str


class AlertGenerator:
    """Generates structured emergency alerts and keeps track of alert history."""

    def __init__(self):
        """Initialize the AlertGenerator."""
        self.alert_history: List[AlertPacket] = []

    def generate_alert(self, severity_report: SeverityReport, incident_classification: IncidentClassification, correlations: Dict) -> Optional[AlertPacket]:
        """
        Generate a structured alert from severity and classification data.
        
        Args:
            severity_report (SeverityReport): The severity report for the incident.
            incident_classification (IncidentClassification): The classification details.
            correlations (Dict): External correlations like crowd estimates.
            
        Returns:
            AlertPacket: The generated alert packet, or None on failure.
        """
        try:
            affected = incident_classification.affected_cameras
            origin = affected[0] if affected else "UNKNOWN"
            
            alert = AlertPacket(
                incident_id=str(uuid.uuid4()),
                timestamp=datetime.utcnow().isoformat(),
                incident_type=incident_classification.incident_type.name,
                severity_level=severity_report.level,
                estimated_origin=origin,
                propagation_path=affected,
                affected_zones=severity_report.affected_zones,
                crowd_estimate=correlations.get('crowd_estimate', 0),
                recommended_action=severity_report.recommended_action,
                confidence_score=incident_classification.confidence,
                description=f"[{severity_report.trend.value}] {incident_classification.description}"
            )
            
            # Simple deduplication could be implemented here
            self.alert_history.append(alert)
            return alert
        except Exception as e:
            logger.error(f"Failed to generate alert: {e}")
            return None
