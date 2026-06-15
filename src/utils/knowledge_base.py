import numpy as np
from src.utils.logger import setup_production_logger

logger = setup_production_logger("knowledge_base")

class FactoryKnowledgeBase:
    """
    Enterprise-grade vector index storage simulator.
    Uses Cosine Similarity to cross-reference runtime neural network alerts
    against proprietary corporate engineering manuals.
    """
    def __init__(self):
        self.vector_database_index = []
        self._initialize_and_index_vault()

    def _generate_semantic_embedding(self, text_input: str) -> np.ndarray:
        """
        Simulates a text-embedding model vectorization.
        Converts text blocks into normalized mathematical coordinate matrices.
        """
        hash_seed = sum(ord(char) for char in text_input)
        np.random.seed(hash_seed)
        raw_vector = np.random.randn(128)
        return raw_vector / np.linalg.norm(raw_vector)

    def _initialize_and_index_vault(self):
        """Populates and caches the initial local vector index with repair manuals."""
        raw_manuals = [
            {
                "error_signature": "CRITICAL_ANOMALY at 1200Hz friction signature strike",
                "root_cause": "Severe dry bearing degradation. Protective lubricant seal failure causing metal-on-metal wear.",
                "resolution_protocol": "DE-ENERGIZE IMMEDIATELY. Apply Lube-X High-Viscosity Grade 4 Grease to bearing housing. Swap unit if deep scoring is present."
            },
            {
                "error_signature": "CRITICAL_ANOMALY at 60Hz baseline motor hum failure",
                "root_cause": "Electrical phase unbalance or loose anchor mount bolts causing severe structural wobble.",
                "resolution_protocol": "Isolate power. Check mounting plates. Retighten all Grade-8 anchor bolts using a torque wrench to exactly 85 ft-lbs."
            }
        ]

        logger.info("🗄️ [KNOWLEDGE BASE] Initializing corporate vector repository indexing...")
        for manual in raw_manuals:
            vector_coordinates = self._generate_semantic_embedding(manual["error_signature"])
            self.vector_database_index.append({
                "vector": vector_coordinates,
                "metadata": manual
            })
        logger.info(f"✅ [KNOWLEDGE BASE] Indexing complete. Cached {len(self.vector_database_index)} manual blueprints.")

    def query_closest_manual(self, anomaly_alert_string: str) -> dict:
        """Executes geometric dot-product calculations to pull the best matching documentation."""
        query_vector = self._generate_semantic_embedding(anomaly_alert_string)
        highest_similarity = -1.0
        best_match = None

        for record in self.vector_database_index:
            stored_vector = record["vector"]
            # Cosine similarity calculation line
            similarity = np.dot(query_vector, stored_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(stored_vector))
            
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = record["metadata"]

        if best_match:
            logger.info(f"🎯 [KNOWLEDGE BASE] Semantic match resolved. Cosine Score: {highest_similarity:.4f}")
        return best_match
