"""
Google Scholar Validator
Validates papers and fetches metadata using Google Scholar when DOI/ISSN fails.
"""

import logging
import requests
import re
from typing import Optional, Dict, List
from dataclasses import dataclass
import time
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)


@dataclass
class ScholarMetadata:
    """Container for Google Scholar metadata."""
    title: str = ""
    authors: str = ""
    year: int = 0
    journal: str = ""
    publisher: str = ""
    citations: int = 0
    pdf_link: str = ""
    scholar_link: str = ""
    abstract: str = ""
    success: bool = False
    error: str = ""


class GoogleScholarValidator:
    """Validate papers using Google Scholar."""
    
    def __init__(self):
        """Initialize Google Scholar validator."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.base_url = "https://scholar.google.com/scholar"
        self.rate_limit_delay = 2.0  # Be respectful to Google
        self.last_request_time = 0
    
    def search_paper(self, title: str, authors: str = "", year: int = 0) -> ScholarMetadata:
        """
        Search for a paper in Google Scholar.
        
        Args:
            title: Paper title
            authors: Author names (optional)
            year: Publication year (optional)
            
        Returns:
            ScholarMetadata object
        """
        if not title or len(title) < 5:
            return ScholarMetadata(error="Title too short")
        
        try:
            self._respect_rate_limit()
            
            # Build search query
            query = title
            if authors:
                query += f" {authors.split(',')[0]}"  # Add first author
            
            logger.info(f"Searching Google Scholar for: {query[:50]}...")
            
            # Search Google Scholar
            params = {
                'q': query,
                'hl': 'en',
                'as_sdt': '0,5'  # Include patents and citations
            }
            
            response = self.session.get(
                self.base_url,
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                return self._parse_results(response.text, title, year)
            elif response.status_code == 429:
                logger.warning("Google Scholar rate limit hit")
                return ScholarMetadata(error="Rate limit exceeded")
            else:
                logger.error(f"Google Scholar error: {response.status_code}")
                return ScholarMetadata(error=f"HTTP {response.status_code}")
                
        except requests.Timeout:
            logger.error("Google Scholar timeout")
            return ScholarMetadata(error="Timeout")
        except Exception as e:
            logger.error(f"Google Scholar error: {e}")
            return ScholarMetadata(error=str(e))
    
    def validate_by_title(self, title: str, authors: str = "") -> bool:
        """
        Validate if a paper exists in Google Scholar.
        
        Args:
            title: Paper title
            authors: Author names (optional)
            
        Returns:
            True if paper found with reasonable confidence
        """
        result = self.search_paper(title, authors)
        return result.success
    
    def _respect_rate_limit(self):
        """Respect rate limits to avoid blocking."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            logger.debug(f"Rate limiting: sleeping {sleep_time:.1f}s")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _parse_results(self, html: str, query_title: str, query_year: int = 0) -> ScholarMetadata:
        """Parse Google Scholar search results."""
        try:
            # Extract first result
            # Google Scholar HTML structure (simplified parsing)
            
            # Find first result div
            result_pattern = r'<div class="gs_ri">(.*?)</div>\s*</div>'
            matches = re.findall(result_pattern, html, re.DOTALL)
            
            if not matches:
                return ScholarMetadata(error="No results found")
            
            first_result = matches[0]
            
            metadata = ScholarMetadata()
            metadata.success = True
            
            # Extract title
            title_pattern = r'<h3[^>]*><a[^>]*>(.*?)</a>'
            title_match = re.search(title_pattern, first_result)
            if title_match:
                title = re.sub(r'<[^>]+>', '', title_match.group(1))
                metadata.title = self._clean_text(title)
            
            # Extract authors and publication info
            info_pattern = r'<div class="gs_a">(.*?)</div>'
            info_match = re.search(info_pattern, first_result)
            if info_match:
                info_text = re.sub(r'<[^>]+>', '', info_match.group(1))
                parts = info_text.split(' - ')
                
                if len(parts) >= 1:
                    metadata.authors = self._clean_text(parts[0])
                
                if len(parts) >= 2:
                    pub_info = parts[1]
                    # Try to extract year
                    year_match = re.search(r'\b(19|20)\d{2}\b', pub_info)
                    if year_match:
                        metadata.year = int(year_match.group(0))
                    
                    # Extract journal/publisher
                    metadata.journal = self._clean_text(pub_info.split(',')[0])
                
                if len(parts) >= 3:
                    metadata.publisher = self._clean_text(parts[2])
            
            # Extract snippet/abstract
            snippet_pattern = r'<div class="gs_rs">(.*?)</div>'
            snippet_match = re.search(snippet_pattern, first_result)
            if snippet_match:
                snippet = re.sub(r'<[^>]+>', '', snippet_match.group(1))
                metadata.abstract = self._clean_text(snippet)
            
            # Extract citation count
            citation_pattern = r'Cited by (\d+)'
            citation_match = re.search(citation_pattern, first_result)
            if citation_match:
                metadata.citations = int(citation_match.group(1))
            
            # Calculate match score
            if metadata.title:
                match_score = self._calculate_match_score(query_title, metadata.title)
                
                if match_score < 0.5:
                    logger.warning(f"Low match score: {match_score:.2f}")
                    metadata.success = False
                    metadata.error = "Low match confidence"
                else:
                    logger.info(f"Google Scholar match found (score: {match_score:.2f})")
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error parsing Google Scholar results: {e}")
            return ScholarMetadata(error=f"Parse error: {e}")
    
    def _clean_text(self, text: str) -> str:
        """Clean HTML text."""
        # Remove HTML entities
        text = re.sub(r'&[a-z]+;', ' ', text)
        text = re.sub(r'&#\d+;', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _calculate_match_score(self, query_title: str, result_title: str) -> float:
        """Calculate how well result matches query."""
        if not query_title or not result_title:
            return 0.0
        
        # Normalize
        query = query_title.lower()
        result = result_title.lower()
        
        # Remove common words
        stop_words = {'a', 'an', 'the', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'and', 'or'}
        
        query_words = set(query.split()) - stop_words
        result_words = set(result.split()) - stop_words
        
        if not query_words or not result_words:
            return 0.0
        
        # Jaccard similarity
        intersection = len(query_words.intersection(result_words))
        union = len(query_words.union(result_words))
        
        return intersection / union if union > 0 else 0.0


# Global instance
google_scholar_validator = GoogleScholarValidator()


def validate_paper_scholar(title: str, authors: str = "", year: int = 0) -> ScholarMetadata:
    """
    Convenience function to validate paper via Google Scholar.
    
    Args:
        title: Paper title
        authors: Author names
        year: Publication year
        
    Returns:
        ScholarMetadata object
    """
    return google_scholar_validator.search_paper(title, authors, year)






















