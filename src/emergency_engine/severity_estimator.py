import logging
from enum import Enum
from dataclasses import dataclass
from typing import List
from .incident_classifier import IncidentClassification, IncidentType

logger = logging.getLogger(__name__)

class Trend(Enum):
    ESCALATING = "ESCALATING"
    STABLE = "STABLE"
    DEESCALATING = "DEESCALATING"

@dataclass
class SeverityReport:
    """Report detailing the severity of an incident."""
    level: int
    score: float
    trend: Trend
    affected_zones: List[str]
    recommended_action: str

class SeverityEstimator:
    """Estimates the severity of an incident considering history and cross-camera spread."""

    def __init__(self):
        """Initialize the SeverityEstimator."""
        pass

    def estimate(self, incident_classification: IncidentClassification, scene_history: List[dict]) -> SeverityReport:
        """
        Estimate the overall severity of an incident.
        
        Args:
            incident_classification (IncidentClassification): The output from the IncidentClassifier.
            scene_history (List[dict]): Running timeline of previous events.
            
        Returns:
            SeverityReport containing severity score, level, and trend.
        """
        try:
            base_score = incident_classification.severity_level
            
            # Trend analysis
            trend = Trend.STABLE
            if len(scene_history) > 1:
                past_severity = scene_history[-2].get('severity_level', 1)
                if base_score > past_severity:
                    trend = Trend.ESCALATING
                elif base_score < past_severity:
                    trend = Trend.DEESCALATING
                    
            # Compute a continuous score factoring spread and confidence
            spread_factor = min(1.0, len(incident_classification.affected_cameras) * 0.2)
            continuous_score = float(base_score) + spread_factor * incident_classification.confidence
            
            if trend == Trend.ESCALATING:
                continuous_score *= 1.1
            elif trend == Trend.DEESCALATING:
                continuous_score *= 0.9
                
            level = min(5, max(1, int(round(continuous_score))))
            
            actions = {
                1: "No action required.",
                2: "Monitor situation closely.",
                3: "Dispatch security personnel to area.",
                4: "Initiate partial evacuation and alert authorities.",
                5: "Immediate full evacuation and emergency response."
            }
            
            return SeverityReport(
                level=level,
                score=continuous_score,
                trend=trend,
                affected_zones=incident_classification.affected_cameras,
                recommended_action=actions.get(level, "Monitor.")
            )
        except Exception as e:
            logger.error(f"Error estimating severity: {e}")
            return SeverityReport(
                level=1, 
                score=1.0, 
                trend=Trend.STABLE, 
                affected_zones=[], 
                recommended_action="Error estimating severity."
            )
