"""
Text preprocessing and tokenization utilities supporting English (MIND) and Danish (EB-NeRD).
"""
import re
from typing import List, Set

# Standard English stopwords
ENGLISH_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so",
    "some", "such", "than", "that", "that's", "the", "their", "theirs", "them",
    "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll",
    "they're", "they've", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've",
    "were", "weren't", "what", "what's", "when", "when's", "where", "where's",
    "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't",
    "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your",
    "yours", "yourself", "yourselves"
}

# Standard Danish stopwords
DANISH_STOPWORDS: Set[str] = {
    "og", "i", "jeg", "det", "at", "en", "den", "til", "er", "som", "paa", "på",
    "de", "med", "han", "af", "for", "ikke", "der", "var", "mig", "sig", "men",
    "et", "har", "om", "vi", "min", "havde", "ham", "hun", "nu", "over", "da",
    "fra", "du", "ud", "sin", "dem", "os", "op", "man", "hans", "hvor", "eller",
    "hvad", "skal", "selv", "her", "alle", "vil", "blev", "foer", "før", "saa",
    "så", "under", "have", "nogen", "noget", "efter", "ned", "da", "lille",
    "din", "mit", "alt", "mod", "hvis", "hver", "dette", "disse", "selv"
}

# Regex pattern to match alphanumeric tokens (including Danish characters æ, ø, å)
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9æøåÆØÅ]+")

def tokenize(text: str, language: str = "english", remove_stopwords: bool = True) -> List[str]:
    """
    Tokenizes input text into normalized lowercase tokens, optionally filtering stopwords.
    
    Args:
        text: Input string.
        language: 'english' or 'danish'.
        remove_stopwords: Whether to remove language-specific stopwords.
        
    Returns:
        List of tokens.
    """
    if not text:
        return []
    
    tokens = TOKEN_PATTERN.findall(text.lower())
    if not remove_stopwords:
        return tokens
    
    stopwords = DANISH_STOPWORDS if language.lower().startswith("dan") else ENGLISH_STOPWORDS
    return [t for t in tokens if t not in stopwords and len(t) > 1]
