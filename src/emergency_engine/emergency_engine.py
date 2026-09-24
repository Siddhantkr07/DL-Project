import logging
from typing import Optional, Dict
from .incident_classifier import IncidentClassifier
from .severity_estimator import SeverityEstimator
from .alert_generator import AlertGenerator, AlertPacket

logger = logging.getLogger(__name__)

class EmergencyEngine:
    """Main orchestrator for Sentinel AI Emergency capabilities."""

    def __init__(self):
        """Initialize the EmergencyEngine and its subcomponents."""
        self.classifier = IncidentClassifier()
        self.severity_estimator = SeverityEstimator()
        self.alert_generator = AlertGenerator()
        self.incident_timeline = []

    def process(self, global_scene_model: dict, correlations: dict) -> Optional[AlertPacket]:
        """
        Process the global scene model to detect incidents and generate alerts if necessary.
        
        Args:
            global_scene_model (dict): The aggregated scene data across cameras.
            correlations (dict): Additional correlations and insights.
            
        Returns:
            AlertPacket if an alert is warranted (severity >= 2), otherwise None.
        """
        try:
            classification = self.classifier.classify(global_scene_model)
            
            # Maintain a running timeline
            history_record = {
                'severity_level': classification.severity_level,
                'timestamp': global_scene_model.get('timestamp', '')
            }
            self.incident_timeline.append(history_record)
            if len(self.incident_timeline) > 100:
                self.incident_timeline.pop(0)
                
            severity_report = self.severity_estimator.estimate(classification, self.incident_timeline)
            
            # Trigger alert for severity >= 2
            if severity_report.level >= 2:
                alert = self.alert_generator.generate_alert(severity_report, classification, correlations)
                return alert
                
            return None
        except Exception as e:
            logger.error(f"Emergency Engine process failed: {e}")
            return None
