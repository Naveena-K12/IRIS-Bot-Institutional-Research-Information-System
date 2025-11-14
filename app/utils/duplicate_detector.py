"""
Duplicate Detection Service
Detects duplicate papers using multiple similarity metrics: title normalization, author overlap, and semantic embeddings.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np

from .semantic_embedder import semantic_embedder

logger = logging.getLogger(__name__)


class DuplicateDetector:
    """Service for detecting duplicate papers."""
    
    def __init__(self, 
                 title_similarity_threshold: float = 0.85,
                 author_overlap_threshold: float = 0.5,
                 embedding_similarity_threshold: float = 0.90):
        """
        Initialize the duplicate detector.
        
        Args:
            title_similarity_threshold: Minimum title similarity to consider duplicate (0-1)
            author_overlap_threshold: Minimum author overlap ratio to consider duplicate (0-1)
            embedding_similarity_threshold: Minimum embedding similarity to consider duplicate (0-1)
        """
        self.title_similarity_threshold = title_similarity_threshold
        self.author_overlap_threshold = author_overlap_threshold
        self.embedding_similarity_threshold = embedding_similarity_threshold
    
    def normalize_title(self, title: str) -> str:
        """
        Normalize title for comparison.
        
        Args:
            title: Original title
            
        Returns:
            Normalized title
        """
        if not title:
            return ""
        
        # Lowercase
        normalized = title.lower()
        
        # Remove punctuation except spaces
        normalized = re.sub(r'[^\w\s]', '', normalized)
        
        # Remove extra whitespace
        normalized = " ".join(normalized.split())
        
        return normalized
    
    def jaccard_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate Jaccard similarity between two strings.
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Jaccard similarity score (0-1)
        """
        if not str1 or not str2:
            return 0.0
        
        set1 = set(str1.split())
        set2 = set(str2.split())
        
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def levenshtein_distance(self, str1: str, str2: str) -> int:
        """
        Calculate Levenshtein distance between two strings.
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Levenshtein distance
        """
        if not str1 or not str2:
            return max(len(str1), len(str2))
        
        m, n = len(str1), len(str2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if str1[i - 1] == str2[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + cost
                )
        
        return dp[m][n]
    
    def fuzzy_title_match(self, title1: str, title2: str) -> float:
        """
        Calculate fuzzy match score between two titles.
        
        Args:
            title1: First title
            title2: Second title
            
        Returns:
            Similarity score (0-1)
        """
        norm1 = self.normalize_title(title1)
        norm2 = self.normalize_title(title2)
        
        # Exact match
        if norm1 == norm2:
            return 1.0
        
        # Jaccard similarity
        jaccard = self.jaccard_similarity(norm1, norm2)
        
        # Levenshtein-based similarity
        max_len = max(len(norm1), len(norm2))
        if max_len == 0:
            return 0.0
        
        levenshtein = self.levenshtein_distance(norm1, norm2)
        levenshtein_sim = 1.0 - (levenshtein / max_len)
        
        # Weighted combination
        similarity = 0.7 * jaccard + 0.3 * levenshtein_sim
        
        return similarity
    
    def parse_authors(self, authors_str: str) -> Set[str]:
        """
        Parse authors string into set of normalized author names.
        
        Args:
            authors_str: Authors string (comma-separated or other formats)
            
        Returns:
            Set of normalized author names
        """
        if not authors_str:
            return set()
        
        # Split by common delimiters
        authors = re.split(r'[,;]|\band\b', authors_str)
        
        # Normalize and clean
        normalized = set()
        for author in authors:
            author = author.strip()
            if author:
                # Normalize: lowercase, remove extra spaces
                author = " ".join(author.lower().split())
                normalized.add(author)
        
        return normalized
    
    def author_overlap(self, authors1: str, authors2: str) -> float:
        """
        Calculate author overlap ratio.
        
        Args:
            authors1: First authors string
            authors2: Second authors string
            
        Returns:
            Overlap ratio (0-1)
        """
        set1 = self.parse_authors(authors1)
        set2 = self.parse_authors(authors2)
        
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        # Return ratio of intersection to union (Jaccard)
        return intersection / union if union > 0 else 0.0
    
    def is_duplicate(self, paper1: Dict[str, Any], paper2: Dict[str, Any]) -> Tuple[bool, float, str]:
        """
        Check if two papers are duplicates.
        
        Args:
            paper1: First paper dictionary
            paper2: Second paper dictionary
            
        Returns:
            Tuple of (is_duplicate, similarity_score, reason)
        """
        # Extract fields
        title1 = paper1.get('title', '')
        title2 = paper2.get('title', '')
        authors1 = paper1.get('authors', '')
        authors2 = paper2.get('authors', '')
        abstract1 = paper1.get('abstract', '')
        abstract2 = paper2.get('abstract', '')
        doi1 = paper1.get('doi', '')
        doi2 = paper2.get('doi', '')
        
        # Check DOI match (strongest indicator)
        if doi1 and doi2 and doi1.lower().strip() == doi2.lower().strip():
            return True, 1.0, "DOI match"
        
        # Check title similarity
        title_sim = self.fuzzy_title_match(title1, title2)
        
        # Check author overlap
        author_overlap_ratio = self.author_overlap(authors1, authors2)
        
        # Calculate combined score
        # If title is very similar and authors overlap, likely duplicate
        if title_sim >= self.title_similarity_threshold and author_overlap_ratio >= self.author_overlap_threshold:
            similarity = (title_sim * 0.6 + author_overlap_ratio * 0.4)
            return True, similarity, f"Title similarity: {title_sim:.2f}, Author overlap: {author_overlap_ratio:.2f}"
        
        # Check embedding similarity if abstracts are available
        if abstract1 and abstract2:
            try:
                # Generate embeddings
                embedding1 = semantic_embedder.generate_embedding(f"{title1} {abstract1}")
                embedding2 = semantic_embedder.generate_embedding(f"{title2} {abstract2}")
                
                # Calculate cosine similarity
                embedding_sim = semantic_embedder.cosine_similarity(embedding1, embedding2)
                
                if embedding_sim >= self.embedding_similarity_threshold:
                    return True, embedding_sim, f"Embedding similarity: {embedding_sim:.2f}"
            except Exception as e:
                logger.warning(f"Error calculating embedding similarity: {e}")
        
        # Not duplicate
        max_similarity = max(title_sim, author_overlap_ratio)
        return False, max_similarity, f"Title similarity: {title_sim:.2f}, Author overlap: {author_overlap_ratio:.2f}"
    
    def find_duplicates(self, paper: Dict[str, Any], existing_papers: List[Dict[str, Any]]) -> List[Tuple[int, float, str]]:
        """
        Find duplicates of a paper among existing papers.
        
        Args:
            paper: Paper to check for duplicates
            existing_papers: List of existing papers to check against
            
        Returns:
            List of tuples (paper_id, similarity_score, reason)
        """
        duplicates = []
        
        for existing in existing_papers:
            # Skip if comparing to itself
            if paper.get('id') == existing.get('id'):
                continue
            
            # Skip if existing paper is already marked as duplicate
            if existing.get('is_duplicate'):
                continue
            
            is_dup, similarity, reason = self.is_duplicate(paper, existing)
            
            if is_dup:
                duplicates.append((existing.get('id'), similarity, reason))
        
        # Sort by similarity (descending)
        duplicates.sort(key=lambda x: x[1], reverse=True)
        
        return duplicates
    
    def mark_duplicate(self, paper_id: int, original_paper_id: int, similarity_score: float) -> bool:
        """
        Mark a paper as duplicate (to be called by repository).
        
        Args:
            paper_id: ID of duplicate paper
            original_paper_id: ID of original paper
            similarity_score: Similarity score
            
        Returns:
            True if successful
        """
        # This will be called by the repository to update the database
        # The actual database update is handled by the repository
        return True


# Global instance
duplicate_detector = DuplicateDetector()

