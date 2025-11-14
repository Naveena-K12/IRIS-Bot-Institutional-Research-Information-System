"""
Paper Type Detection
Automatically detects the type of research paper from PDF content.
"""

import re
import logging
from typing import Optional, Dict, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class PaperTypeDetector:
    """Detects paper type from PDF content."""
    
    # Paper type categories
    JOURNAL_ARTICLE = "Journal Article"
    CONFERENCE_PAPER = "Conference Paper"
    BOOK_CHAPTER = "Book Chapter"
    THESIS = "Thesis/Dissertation"
    TECHNICAL_REPORT = "Technical Report"
    PREPRINT = "Preprint"
    REVIEW_ARTICLE = "Review Article"
    CASE_STUDY = "Case Study"
    SHORT_PAPER = "Short Paper"
    POSTER = "Poster"
    WORKSHOP_PAPER = "Workshop Paper"
    WHITE_PAPER = "White Paper"
    UNKNOWN = "Unknown"
    
    def __init__(self):
        """Initialize paper type detector with patterns."""
        self.patterns = self._load_patterns()
    
    def _load_patterns(self) -> Dict[str, List[Dict]]:
        """Load detection patterns for each paper type."""
        patterns = {}
        
        # Journal Article patterns
        patterns[self.JOURNAL_ARTICLE] = [
            {'pattern': r'(?:International|European|American|Asian|Journal)\s+(?:Journal|Review)\s+(?:of|for)', 'weight': 10, 'location': 'title'},
            {'pattern': r'ISSN[:\s]+\d{4}-\d{3}[\dXx]', 'weight': 8, 'location': 'anywhere'},
            {'pattern': r'Volume\s+\d+[,\s]+(?:Issue|Number)\s+\d+', 'weight': 7, 'location': 'anywhere'},
            {'pattern': r'(?:Published|Accepted|Received)\s+(?:in|by|online)', 'weight': 5, 'location': 'anywhere'},
            {'pattern': r'©\s*\d{4}.*(?:Publisher|Publishing|Press)', 'weight': 6, 'location': 'anywhere'},
        ]
        
        # Conference Paper patterns
        patterns[self.CONFERENCE_PAPER] = [
            {'pattern': r'(?:Proceedings|Conference|Symposium|Workshop)\s+(?:of|on)', 'weight': 10, 'location': 'title'},
            {'pattern': r'(?:IEEE|ACM|AAAI|CVPR|NeurIPS|ICML|ICLR|IJCAI)', 'weight': 9, 'location': 'anywhere'},
            {'pattern': r'(?:International|Annual|Regional)\s+Conference', 'weight': 8, 'location': 'anywhere'},
            {'pattern': r'Presented\s+at', 'weight': 7, 'location': 'anywhere'},
            {'pattern': r'Session\s+\d+', 'weight': 5, 'location': 'anywhere'},
        ]
        
        # Book Chapter patterns
        patterns[self.BOOK_CHAPTER] = [
            {'pattern': r'Chapter\s+\d+', 'weight': 10, 'location': 'title'},
            {'pattern': r'ISBN[:\s]+[\d-]+', 'weight': 9, 'location': 'anywhere'},
            {'pattern': r'(?:In|From):\s+[A-Z][^,\n]+\(Ed[s]?\.\)', 'weight': 8, 'location': 'anywhere'},
            {'pattern': r'Pages?\s+\d+\s*[-–]\s*\d+', 'weight': 6, 'location': 'anywhere'},
            {'pattern': r'(?:Springer|Elsevier|Wiley|Cambridge|Oxford)\s+(?:University\s+)?Press', 'weight': 7, 'location': 'anywhere'},
        ]
        
        # Thesis/Dissertation patterns
        patterns[self.THESIS] = [
            {'pattern': r'(?:PhD|Ph\.D\.|Master[\'s]*|M\.S\.|M\.Sc\.|Bachelor[\'s]*|B\.S\.|B\.Sc\.)\s+(?:Thesis|Dissertation)', 'weight': 15, 'location': 'anywhere'},
            {'pattern': r'(?:Doctoral|Graduate)\s+(?:Thesis|Dissertation)', 'weight': 15, 'location': 'anywhere'},
            {'pattern': r'(?:Submitted|Presented)\s+(?:to|in\s+partial\s+fulfillment)', 'weight': 10, 'location': 'anywhere'},
            {'pattern': r'Department\s+of\s+[A-Z][^\n,]+,\s+(?:University|Institute)', 'weight': 8, 'location': 'anywhere'},
            {'pattern': r'(?:Advisor|Supervisor|Committee):', 'weight': 7, 'location': 'anywhere'},
        ]
        
        # Technical Report patterns
        patterns[self.TECHNICAL_REPORT] = [
            {'pattern': r'Technical\s+Report', 'weight': 15, 'location': 'anywhere'},
            {'pattern': r'TR[:\-\s]+\d+', 'weight': 12, 'location': 'anywhere'},
            {'pattern': r'(?:Report|Memo|Memorandum)\s+(?:No\.|Number|#)\s*\d+', 'weight': 10, 'location': 'anywhere'},
            {'pattern': r'(?:Internal|Research|Lab|Laboratory)\s+Report', 'weight': 9, 'location': 'anywhere'},
        ]
        
        # Preprint patterns
        patterns[self.PREPRINT] = [
            {'pattern': r'arXiv:\d+\.\d+', 'weight': 15, 'location': 'anywhere'},
            {'pattern': r'bioRxiv|medRxiv|ChemRxiv', 'weight': 15, 'location': 'anywhere'},
            {'pattern': r'Preprint', 'weight': 12, 'location': 'anywhere'},
            {'pattern': r'(?:Not\s+)?(?:peer[\-\s])?reviewed', 'weight': 5, 'location': 'anywhere'},
        ]
        
        # Review Article patterns
        patterns[self.REVIEW_ARTICLE] = [
            {'pattern': r'(?:Systematic|Literature|Comprehensive)\s+Review', 'weight': 12, 'location': 'title'},
            {'pattern': r'(?:Survey|Overview)\s+(?:of|on)', 'weight': 10, 'location': 'title'},
            {'pattern': r'State[\-\s]of[\-\s]the[\-\s]Art', 'weight': 10, 'location': 'title'},
            {'pattern': r'Meta[\-\s]Analysis', 'weight': 12, 'location': 'title'},
        ]
        
        # Case Study patterns
        patterns[self.CASE_STUDY] = [
            {'pattern': r'Case\s+Study', 'weight': 15, 'location': 'title'},
            {'pattern': r'A\s+Case\s+(?:of|from|in)', 'weight': 10, 'location': 'title'},
            {'pattern': r'(?:Real[\-\s]world|Industrial)\s+(?:Application|Example)', 'weight': 8, 'location': 'anywhere'},
        ]
        
        # Short Paper patterns
        patterns[self.SHORT_PAPER] = [
            {'pattern': r'Short\s+Paper', 'weight': 15, 'location': 'anywhere'},
            {'pattern': r'Brief\s+(?:Communication|Report|Note)', 'weight': 12, 'location': 'anywhere'},
            {'pattern': r'Letter\s+to\s+(?:the\s+)?Editor', 'weight': 10, 'location': 'anywhere'},
        ]
        
        # Poster patterns
        patterns[self.POSTER] = [
            {'pattern': r'Poster\s+(?:Presentation|Session|Abstract)', 'weight': 15, 'location': 'anywhere'},
            {'pattern': r'Extended\s+Abstract', 'weight': 10, 'location': 'anywhere'},
        ]
        
        # Workshop Paper patterns
        patterns[self.WORKSHOP_PAPER] = [
            {'pattern': r'Workshop\s+(?:on|Paper)', 'weight': 15, 'location': 'anywhere'},
            {'pattern': r'(?:Co[\-\s]located|Associated)\s+with', 'weight': 8, 'location': 'anywhere'},
        ]
        
        # White Paper patterns
        patterns[self.WHITE_PAPER] = [
            {'pattern': r'White\s+Paper', 'weight': 15, 'location': 'anywhere'},
            {'pattern': r'Position\s+Paper', 'weight': 12, 'location': 'anywhere'},
            {'pattern': r'Policy\s+(?:Brief|Paper)', 'weight': 10, 'location': 'anywhere'},
        ]
        
        return patterns
    
    def detect_paper_type(self, text: str, title: str = "", doi: str = "") -> str:
        """
        Detect paper type from PDF content.
        
        Args:
            text: Full or partial PDF text content
            title: Paper title (if known)
            doi: DOI (if known)
            
        Returns:
            Detected paper type
        """
        # Search in first 3000 characters for efficiency
        search_text = text[:3000]
        
        # Check DOI prefix for quick detection
        if doi:
            paper_type_from_doi = self._detect_from_doi(doi)
            if paper_type_from_doi:
                logger.info(f"Detected paper type from DOI: {paper_type_from_doi}")
                return paper_type_from_doi
        
        # Score each paper type
        scores = {}
        
        for paper_type, pattern_list in self.patterns.items():
            score = 0
            
            for pattern_info in pattern_list:
                pattern = pattern_info['pattern']
                weight = pattern_info['weight']
                location = pattern_info['location']
                
                # Determine where to search
                if location == 'title' and title:
                    search_area = title
                elif location == 'anywhere':
                    search_area = search_text
                else:
                    search_area = search_text
                
                # Check for pattern match
                if re.search(pattern, search_area, re.IGNORECASE):
                    score += weight
                    logger.debug(f"Matched pattern for {paper_type}: {pattern} (weight: {weight})")
            
            scores[paper_type] = score
        
        # Get the highest scoring type
        if scores:
            max_score = max(scores.values())
            
            # Require minimum score of 8 to be confident
            if max_score >= 8:
                detected_type = max(scores, key=scores.get)
                logger.info(f"Detected paper type: {detected_type} (score: {max_score})")
                return detected_type
        
        # Default to unknown
        logger.info("Could not confidently detect paper type")
        return self.UNKNOWN
    
    def _detect_from_doi(self, doi: str) -> Optional[str]:
        """Try to detect paper type from DOI prefix."""
        doi_lower = doi.lower()
        
        # arXiv preprints
        if 'arxiv' in doi_lower:
            return self.PREPRINT
        
        # Common conference publishers
        if any(conf in doi_lower for conf in ['ieee', 'acm']):
            # Could be conference or journal, return None to let pattern matching decide
            return None
        
        # Springer book chapters
        if '978-' in doi or 'isbn' in doi_lower:
            return self.BOOK_CHAPTER
        
        return None
    
    def get_all_types(self) -> List[str]:
        """Get list of all supported paper types."""
        return [
            self.JOURNAL_ARTICLE,
            self.CONFERENCE_PAPER,
            self.BOOK_CHAPTER,
            self.THESIS,
            self.TECHNICAL_REPORT,
            self.PREPRINT,
            self.REVIEW_ARTICLE,
            self.CASE_STUDY,
            self.SHORT_PAPER,
            self.POSTER,
            self.WORKSHOP_PAPER,
            self.WHITE_PAPER,
            self.UNKNOWN,
        ]
    
    def validate_type(self, paper_type: str) -> bool:
        """Check if a paper type is valid."""
        return paper_type in self.get_all_types()


# Global instance
paper_type_detector = PaperTypeDetector()


def detect_paper_type(text: str, title: str = "", doi: str = "") -> str:
    """
    Convenience function to detect paper type.
    
    Args:
        text: PDF text content
        title: Paper title (optional)
        doi: DOI (optional)
        
    Returns:
        Detected paper type
    """
    return paper_type_detector.detect_paper_type(text, title, doi)


def get_supported_paper_types() -> List[str]:
    """Get list of all supported paper types."""
    return paper_type_detector.get_all_types()






















