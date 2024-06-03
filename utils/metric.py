from collections import Counter
from typing import List, Tuple

def type_token_ratio(sentence):
    words = sentence.split()
    word_count = Counter(words)
    unique_words = len(word_count)
    total_words = len(words)
    return unique_words / total_words

def apply_diversity_metric(data: List[Tuple[str, str]], top_percentage: float = 0.5) -> List[Tuple[str, str]]:
    scores = [(type_token_ratio(src), (src, tgt)) for src, tgt in data]
    scores.sort(key=lambda x: x[0], reverse=True)
    top_n = int(len(scores) * top_percentage)
    selected_data = [sentence for _, sentence in scores[:top_n]]
    return selected_data