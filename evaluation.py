import numpy as np

class RAGEvaluator:
    @staticmethod
    def calculate_similarity_score(query_embedding, result_embeddings):
        scores = []
        for res_emb in result_embeddings:
            dot_product = np.dot(query_embedding, res_emb)
            norm_q = np.linalg.norm(query_embedding)
            norm_r = np.linalg.norm(res_emb)
            scores.append(dot_product / (norm_q * norm_r))
        return float(np.mean(scores))