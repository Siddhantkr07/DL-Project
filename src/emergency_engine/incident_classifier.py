import logging
from enum import Enum
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)

class IncidentType(Enum):
    """Enumeration of incident types with inherent base severity values (1-5)."""
    FIRE_SMOKE = 5
    CROWD_SURGE = 4
    VIOLENT_ACTIVITY = 4
    SUSPICIOUS_MOVEMENT = 3
    UNAUTHORIZED_ACCESS = 3
    ABANDONED_OBJECT = 2
    NORMAL = 1


@dataclass
class IncidentClassification:
    """Result of classifying an incident."""
    incident_type: IncidentType
    confidence: float
    severity_level: int
    affected_cameras: List[str]
    description: str


class IncidentClassifier:
    """
    Classifies incidents into types using rule-based and simulated ML heuristics.
    """

    def __init__(self):
        """Initialize the IncidentClassifier."""
        pass

    def classify(self, global_scene_model: dict) -> IncidentClassification:
        """
        Classifies the current global scene state.
        
        Args:
            global_scene_model: Dictionary containing aggregated scene information.
                               Expected keys: fire_smoke_count, crowd_density, motion_anomaly, active_cameras.
                               
        Returns:
            IncidentClassification object describing the incident.
        """
        try:
            fire_smoke = global_scene_model.get('fire_smoke_count', 0)
            crowd_density = global_scene_model.get('crowd_density', 0.0)
            motion_anomaly = global_scene_model.get('motion_anomaly', 0.0)
            cameras = global_scene_model.get('active_cameras', [])

            incident_type = IncidentType.NORMAL
            confidence = 1.0
            description = "Normal activity."

            # Priority-based rule evaluation
            if fire_smoke > 0:
                incident_type = IncidentType.FIRE_SMOKE
                confidence = min(1.0, fire_smoke * 0.4)
                description = f"Fire/Smoke detected in {fire_smoke} area(s)."
            elif crowd_density > 0.8:
                incident_type = IncidentType.CROWD_SURGE
                confidence = crowd_density
                description = "High crowd density indicating potential surge."
            elif motion_anomaly > 0.7:
                incident_type = IncidentType.VIOLENT_ACTIVITY
                confidence = motion_anomaly
                description = "Severe motion anomaly detected, possible violence."
            elif motion_anomaly > 0.4:
                incident_type = IncidentType.SUSPICIOUS_MOVEMENT
                confidence = motion_anomaly
                description = "Suspicious or unusual movement detected."
                
            return IncidentClassification(
                incident_type=incident_type,
                confidence=confidence,
                severity_level=incident_type.value,
                affected_cameras=cameras,
                description=description
            )
        except Exception as e:
            logger.error(f"Error during incident classification: {e}")
            return IncidentClassification(
                incident_type=IncidentType.NORMAL, 
                confidence=1.0, 
                severity_level=1, 
                affected_cameras=[], 
                description="Error classifying incident."
            )
