import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple
import networkx as nx

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None
    logging.warning("PyTorch not installed. GNNReasoner will not perform actual ML reasoning.")

@dataclass
class IncidentGraph:
    """Represents the spatial-temporal graph of an incident."""
    nodes: List[str]  # camera_ids
    edges: List[Tuple[str, str]] # pairs of camera_ids connected by entity correlation
    incident_propagation_path: List[str] # Predicted sequence of cameras

class GNNReasoner(nn.Module) if nn is not None else object:
    """
    Graph Neural Network spatial-temporal reasoning module.
    Models cameras as nodes and entities as edges.
    """
    def __init__(self, feature_dim: int = 256):
        super().__init__()
        self.feature_dim = feature_dim
        
        if nn is not None:
            # Simple Message Passing Neural Network parameters
            self.W_msg = nn.Linear(feature_dim, feature_dim)
            self.W_upd = nn.Linear(feature_dim * 2, feature_dim)
            self.path_predictor = nn.Linear(feature_dim, 1) # Predicts likelihood of being in propagation path
        
        logging.info("Initialized GNNReasoner.")

    def build_graph(self, correlations: Dict[str, List[str]]) -> nx.Graph:
        """
        Builds a networkx graph based on entity correlations.
        
        Args:
            correlations: Dict mapping entity_id to list of camera_ids.
            
        Returns:
            A networkx Graph where nodes are cameras and edges represent shared entities.
        """
        G = nx.Graph()
        
        # Build edges based on shared entities
        for ent_id, cam_list in correlations.items():
            if len(cam_list) < 2:
                continue
            # Connect all cameras that saw this entity (clique)
            for i in range(len(cam_list)):
                for j in range(i + 1, len(cam_list)):
                    u, v = cam_list[i], cam_list[j]
                    if not G.has_edge(u, v):
                        G.add_edge(u, v, weight=1)
                    else:
                        G[u][v]['weight'] += 1
                        
        return G

    def reason(self, graph: nx.Graph, node_features: Dict[str, 'torch.Tensor'] = None) -> IncidentGraph:
        """
        Performs GNN message passing on the graph to reason about spatial relationships.
        
        Args:
            graph: The networkx graph built from correlations.
            node_features: Optional dictionary of node embeddings.
            
        Returns:
            IncidentGraph detailing the reasoning output.
        """
        nodes = list(graph.nodes())
        edges = list(graph.edges())
        path = []
        
        if torch is None or not nodes:
            return IncidentGraph(nodes, edges, [])
            
        try:
            # Dummy logic for reasoning if node features aren't provided
            if node_features is None:
                # Initialize random features for demonstration
                node_features = {node: torch.randn(self.feature_dim) for node in nodes}
                
            # One step of message passing
            new_features = {}
            for node in nodes:
                # Aggregate messages from neighbors
                neighbors = list(graph.neighbors(node))
                if not neighbors:
                    new_features[node] = node_features[node]
                    continue
                    
                messages = torch.stack([self.W_msg(node_features[nb]) for nb in neighbors])
                agg_message = torch.mean(messages, dim=0)
                
                # Update node state
                concat_state = torch.cat([node_features[node], agg_message])
                new_state = torch.relu(self.W_upd(concat_state))
                new_features[node] = new_state
                
            # Predict propagation path based on updated node scores
            scores = {}
            for node in nodes:
                score = torch.sigmoid(self.path_predictor(new_features[node])).item()
                scores[node] = score
                
            # Simple heuristic: sort nodes by score to form a path
            path = [node for node, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
            
        except Exception as e:
            logging.error(f"Error during GNN reasoning: {e}")
            
        return IncidentGraph(nodes, edges, path)

if __name__ == "__main__":
    reasoner = GNNReasoner()
    # Mock correlations
    corrs = {
        "ent1": ["camA", "camB"],
        "ent2": ["camB", "camC"],
        "ent3": ["camA"]
    }
    g = reasoner.build_graph(corrs)
    print(f"Built graph with {g.number_of_nodes()} nodes and {g.number_of_edges()} edges.")
    
    incident_graph = reasoner.reason(g)
    print(f"Predicted path: {incident_graph.incident_propagation_path}")
