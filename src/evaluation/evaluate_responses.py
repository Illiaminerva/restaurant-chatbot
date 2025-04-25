"""
Evaluation metrics and testing utilities for restaurant chatbot responses.
Implements various metrics to assess response quality and relevance.
"""

from typing import Dict, List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Standard test queries for consistent evaluation
TEST_QUERIES = [
    "What's a good Italian restaurant in San Francisco?",
    "I'm looking for a cheap sushi place",
    "Where should I go for a romantic dinner?",
    "Any good vegetarian restaurants downtown?",
    "What's the best Mexican food in the area?"
]

# Initialize sentence embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def compute_metrics(response: str, query: str = "") -> Dict[str, float]:
    """
    Compute quality metrics for a response.
    Returns dictionary of scores for different quality aspects.
    """
    metrics = {}

    # Sentence embedding similarity (semantic match)
    embeddings = embedder.encode([query, response])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    metrics["semantic_similarity"] = similarity

    # Response length score
    words = response.split()
    length = len(words)
    metrics['length_score'] = 1.0 if 20 <= length <= 100 else 0.5 if 10 <= length < 20 else 0.0

    # Restaurant relevance score
    restaurant_keywords = ['restaurant', 'food', 'menu', 'dish', 'cuisine', 'dining', 'eat']
    relevant_words = sum(1 for word in words if word.lower() in restaurant_keywords)
    metrics['relevance_score'] = min(relevant_words / 3, 1.0)

    # Information completeness score
    info_keywords = {
        'price': ['$', 'price', 'cheap', 'expensive', 'affordable', 'cost'],
        'location': ['located', 'address', 'street', 'avenue', 'downtown', 'neighborhood'],
        'cuisine': ['cuisine', 'food', 'dish', 'style', 'cooking'],
        'quality': ['good', 'great', 'best', 'amazing', 'excellent', 'favorite']
    }

    info_scores = []
    for category, keywords in info_keywords.items():
        has_info = any(word in response.lower() for word in keywords)
        info_scores.append(1.0 if has_info else 0.0)

    metrics['completeness_score'] = sum(info_scores) / len(info_scores)

    # Penalty for generic/template phrases
    templated_phrases = ["I recommend trying", "one of the best", "you should check out"]
    penalty = any(phrase.lower() in response.lower() for phrase in templated_phrases)
    metrics["template_penalty"] = 0.0 if not penalty else -0.2

    # Overall score (average + penalty)
    base_score = np.mean([
        metrics['length_score'],
        metrics['relevance_score'],
        metrics['completeness_score'],
        metrics['semantic_similarity']
    ])
    metrics['overall_score'] = base_score + metrics["template_penalty"]

    return metrics

def evaluate_responses(model, queries: List[str] = TEST_QUERIES) -> Dict[str, float]:
    """
    Evaluate model responses on test queries.
    Returns averaged metrics across all test cases.
    """
    all_metrics = []
    responses = []

    for query in queries:
        response = model.generate_response(query)
        metrics = compute_metrics(response, query)
        all_metrics.append(metrics)
        responses.append(response)

    # Average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)

    return avg_metrics, responses

def print_evaluation(metrics: Dict[str, float], responses: List[str], queries: List[str] = TEST_QUERIES):
    """
    Print evaluation results in a readable format.
    Displays metrics and sample responses.
    """
    print("\nEvaluation Results:")
    print("=" * 50)
    print(f"Average Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.3f}")

    print("\nSample Responses:")
    print("=" * 50)
    for query, response in zip(queries, responses):
        print(f"\nQuery: {query}")
        print(f"Response: {response}")
        print("-" * 50)
