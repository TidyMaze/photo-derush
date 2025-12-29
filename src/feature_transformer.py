"""Feature interaction transformer for model inference."""

from __future__ import annotations

import numpy as np


class FeatureInteractionTransformer:
    """Transformer to add feature interactions during inference."""
    
    def __init__(self, interaction_pairs: list[tuple[int, int]], ratio_pairs: list[tuple[int, int]] = None):
        self.interaction_pairs = interaction_pairs
        self.ratio_pairs = ratio_pairs or []
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Add interactions and ratios to feature matrix."""
        X_out = X.copy()
        
        # Add interactions
        interactions = []
        for i, j in self.interaction_pairs:
            interaction = X[:, i] * X[:, j]
            interactions.append(interaction)
        
        if interactions:
            X_out = np.hstack([X_out, np.column_stack(interactions)])
        
        # Add ratios
        ratios = []
        for i, j in self.ratio_pairs:
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.divide(X[:, i], X[:, j], 
                                out=np.zeros_like(X[:, i]), 
                                where=X[:, j]!=0)
                ratios.append(ratio)
        
        if ratios:
            X_out = np.hstack([X_out, np.column_stack(ratios)])
        
        return X_out


