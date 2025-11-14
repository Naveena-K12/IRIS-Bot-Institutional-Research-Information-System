"""
Indexing Status Validator
Determines if papers are indexed in SCI (Science Citation Index) or Scopus databases.
"""

import re
import logging
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import requests
import time

logger = logging.getLogger(__name__)


@dataclass
class IndexingStatus:
    """Container for indexing status information."""
    sci_indexed: bool = False
    scopus_indexed: bool = False
    indexing_label: str = ""  # "SCI", "Scopus", "SCI + Scopus", "Non-SCI/Non-Scopus"
    confidence: float = 0.0
    journal_name: str = ""
    issn: str = ""
    publisher: str = ""
    source: str = ""  # Where the validation came from
    error: str = ""


class IndexingValidator:
    """Validates indexing status in SCI and Scopus databases."""
    
    # SCI (Science Citation Index) indicators
    SCI_PUBLISHERS = [
        'Nature Publishing Group',
        'Springer Nature',
        'Wiley',
        'Elsevier',
        'IEEE',
        'ACM',
        'Taylor & Francis',
        'SAGE',
        'Oxford University Press',
        'Cambridge University Press',
        'American Chemical Society',
        'American Physical Society',
        'Royal Society of Chemistry',
        'American Mathematical Society',
        'American Society for Microbiology',
        'American Heart Association',
        'American Cancer Society',
        'Cell Press',
        'PLOS',
        'MDPI',
        'Frontiers',
        'BioMed Central',
        'Hindawi',
    ]
    
    # High-impact SCI journal prefixes
    SCI_JOURNAL_PREFIXES = [
        r'Nature\s+',
        r'Science\s+',
        r'Cell\s+',
        r'Physical\s+Review',
        r'Journal\s+of\s+the\s+American\s+Chemical\s+Society',
        r'Proceedings\s+of\s+the\s+National\s+Academy\s+of\s+Sciences',
        r'The\s+Lancet',
        r'New\s+England\s+Journal\s+of\s+Medicine',
        r'IEEE\s+Transactions',
        r'ACM\s+Computing\s+Surveys',
        r'Journal\s+of\s+Machine\s+Learning\s+Research',
    ]
    
    # Scopus indicators
    SCOPUS_PUBLISHERS = [
        'Elsevier',
        'Springer',
        'Wiley',
        'Taylor & Francis',
        'SAGE',
        'IEEE',
        'ACM',
        'MDPI',
        'Frontiers',
        'PLOS',
        'BioMed Central',
        'Hindawi',
        'Nature Publishing',
        'Oxford University Press',
        'Cambridge University Press',
    ]
    
    # Non-indexed indicators
    NON_INDEXED_INDICATORS = [
        'Preprint',
        'Working Paper',
        'Technical Report',
        'Internal Report',
        'Student Paper',
        'Conference Abstract',
        'Poster',
    ]
    
    def __init__(self):
        """Initialize the indexing validator."""
        self.cache = {}  # Simple cache for repeated lookups
        
    def validate_indexing_status(self, metadata: Dict) -> IndexingStatus:
        """
        Validate indexing status based on available metadata.
        
        Args:
            metadata: Dictionary containing paper metadata
            
        Returns:
            IndexingStatus object with validation results
        """
        status = IndexingStatus()
        
        try:
            # Extract key information
            journal = metadata.get('journal', '').strip()
            publisher = metadata.get('publisher', '').strip()
            issn = metadata.get('issn', '').strip()
            doi = metadata.get('doi', '').strip()
            paper_type = metadata.get('paper_type', '').strip()
            
            # Set basic info
            status.journal_name = journal
            status.issn = issn
            status.publisher = publisher
            
            # Check for non-indexed indicators first
            if self._is_non_indexed(paper_type, journal):
                status.indexing_label = "Non-SCI/Non-Scopus"
                status.confidence = 0.9
                status.source = "Non-indexed indicators"
                return status
            
            # Check SCI indexing
            sci_status = self._check_sci_indexing(journal, publisher, issn, doi)
            status.sci_indexed = sci_status['indexed']
            sci_confidence = sci_status['confidence']
            
            # Check Scopus indexing
            scopus_status = self._check_scopus_indexing(journal, publisher, issn, doi)
            status.scopus_indexed = scopus_status['indexed']
            scopus_confidence = scopus_status['confidence']
            
            # Determine final label and confidence
            if status.sci_indexed and status.scopus_indexed:
                status.indexing_label = "SCI + Scopus"
                status.confidence = min(sci_confidence, scopus_confidence)
                status.source = "Both SCI and Scopus indicators"
            elif status.sci_indexed:
                status.indexing_label = "SCI"
                status.confidence = sci_confidence
                status.source = "SCI indicators"
            elif status.scopus_indexed:
                status.indexing_label = "Scopus"
                status.confidence = scopus_confidence
                status.source = "Scopus indicators"
            else:
                status.indexing_label = "Non-SCI/Non-Scopus"
                status.confidence = 0.6  # Medium confidence for non-indexed
                status.source = "No indexing indicators found"
            
            logger.info(f"Indexing validation: {status.indexing_label} ({status.confidence:.1%} confidence)")
            
        except Exception as e:
            logger.error(f"Error validating indexing status: {e}")
            status.error = str(e)
            status.indexing_label = "Non-SCI/Non-Scopus"
            status.confidence = 0.5
        
        return status
    
    def _is_non_indexed(self, paper_type: str, journal: str) -> bool:
        """Check if paper is clearly non-indexed."""
        paper_type_lower = paper_type.lower()
        journal_lower = journal.lower()
        
        # Check paper type
        for indicator in self.NON_INDEXED_INDICATORS:
            if indicator.lower() in paper_type_lower:
                return True
        
        # Check for thesis/dissertation
        if any(word in paper_type_lower for word in ['thesis', 'dissertation']):
            return True
        
        # Check for technical reports
        if any(word in paper_type_lower for word in ['report', 'memo', 'technical']):
            return True
        
        return False
    
    def _check_sci_indexing(self, journal: str, publisher: str, issn: str, doi: str) -> Dict:
        """Check if journal/paper is likely SCI indexed."""
        confidence = 0.0
        
        # Check publisher
        if publisher:
            for sci_publisher in self.SCI_PUBLISHERS:
                if sci_publisher.lower() in publisher.lower():
                    confidence += 0.4
                    break
        
        # Check journal name patterns
        if journal:
            for pattern in self.SCI_JOURNAL_PREFIXES:
                if re.search(pattern, journal, re.IGNORECASE):
                    confidence += 0.3
                    break
            
            # Check for high-impact journal indicators
            high_impact_indicators = [
                'nature', 'science', 'cell', 'lancet', 'nejm',  # NEJM = New England Journal of Medicine
                'ieee transactions', 'acm computing', 'jmlr',    # JMLR = Journal of Machine Learning Research
                'physical review', 'jacs',                       # JACS = Journal of American Chemical Society
            ]
            
            journal_lower = journal.lower()
            for indicator in high_impact_indicators:
                if indicator in journal_lower:
                    confidence += 0.5
                    break
        
        # Check DOI prefix for major publishers
        if doi:
            sci_doi_prefixes = [
                '10.1038',  # Nature
                '10.1126',  # Science
                '10.1016',  # Elsevier (Cell Press, etc.)
                '10.1109',  # IEEE
                '10.1145',  # ACM
                '10.1007',  # Springer (Nature journals)
            ]
            
            for prefix in sci_doi_prefixes:
                if doi.startswith(prefix):
                    confidence += 0.3
                    break
        
        # Check ISSN (some ISSN ranges are known SCI journals)
        if issn:
            # This is a simplified check - in practice, you'd have a full ISSN database
            confidence += 0.1  # Base confidence for having ISSN
        
        # Determine if SCI indexed
        indexed = confidence >= 0.4  # Threshold for SCI indexing
        
        return {
            'indexed': indexed,
            'confidence': min(confidence, 0.95)  # Cap at 95%
        }
    
    def _check_scopus_indexing(self, journal: str, publisher: str, issn: str, doi: str) -> Dict:
        """Check if journal/paper is likely Scopus indexed."""
        confidence = 0.0
        
        # Check publisher
        if publisher:
            for scopus_publisher in self.SCOPUS_PUBLISHERS:
                if scopus_publisher.lower() in publisher.lower():
                    confidence += 0.4
                    break
        
        # Scopus is more inclusive than SCI, so more journals are indexed
        if journal:
            # Most academic journals with ISSN are in Scopus
            if re.search(r'\bISSN\b', journal, re.IGNORECASE):
                confidence += 0.2
            
            # Check for academic journal indicators
            academic_indicators = [
                'journal', 'review', 'transactions', 'proceedings',
                'international', 'european', 'american', 'asian',
                'quarterly', 'monthly', 'annual'
            ]
            
            journal_lower = journal.lower()
            for indicator in academic_indicators:
                if indicator in journal_lower:
                    confidence += 0.1
                    break
        
        # Check DOI prefix
        if doi:
            scopus_doi_prefixes = [
                '10.1016',  # Elsevier (most journals)
                '10.1007',  # Springer
                '10.1109',  # IEEE
                '10.1145',  # ACM
                '10.1038',  # Nature
                '10.1126',  # Science
                '10.1080',  # Taylor & Francis
                '10.1177',  # SAGE
                '10.3390',  # MDPI
            ]
            
            for prefix in scopus_doi_prefixes:
                if doi.startswith(prefix):
                    confidence += 0.3
                    break
        
        # Scopus includes most journals with ISSN
        if issn:
            confidence += 0.3  # High confidence for ISSN journals
        
        # Determine if Scopus indexed
        indexed = confidence >= 0.3  # Lower threshold than SCI
        
        return {
            'indexed': indexed,
            'confidence': min(confidence, 0.95)
        }
    
    def get_indexing_labels(self) -> List[str]:
        """Get list of all supported indexing labels."""
        return [
            "SCI",
            "Scopus", 
            "SCI + Scopus",
            "Non-SCI/Non-Scopus"
        ]
    
    def validate_label(self, label: str) -> bool:
        """Check if an indexing label is valid."""
        return label in self.get_indexing_labels()


# Global instance
indexing_validator = IndexingValidator()


def validate_indexing_status(metadata: Dict) -> IndexingStatus:
    """
    Convenience function to validate indexing status.
    
    Args:
        metadata: Paper metadata dictionary
        
    Returns:
        IndexingStatus object
    """
    return indexing_validator.validate_indexing_status(metadata)


def get_indexing_labels() -> List[str]:
    """Get list of supported indexing labels."""
    return indexing_validator.get_indexing_labels()





















