"""
Post-Import Verification System
Verifies papers after import using DOI, ISSN, and author+title validation.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .crossref_fetcher import CrossrefAPIFetcher, CrossrefMetadata
from .issn_validator import ISSNValidator, ISSNMetadata
from .google_scholar_validator import GoogleScholarValidator, ScholarMetadata
from .unified_classifier import UnifiedPaperClassifier
from .citation_fetcher import CitationFetcher

logger = logging.getLogger(__name__)


class VerificationStatus(Enum):
    """Verification status enumeration."""
    PENDING = "pending"
    VERIFIED = "verified"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class VerificationResult:
    """Result of post-import verification."""
    paper_id: int
    status: VerificationStatus
    method_used: str  # "doi", "issn", "author_title", "google_scholar"
    confidence_score: float  # 0.0 to 1.0
    verified_metadata: Dict[str, Any]
    errors: List[str]
    suggestions: List[str]
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.suggestions is None:
            self.suggestions = []


class PostImportVerifier:
    """Post-import verification system for papers."""
    
    def __init__(self):
        """Initialize the verifier with all validation tools."""
        self.crossref_fetcher = CrossrefAPIFetcher()
        self.issn_validator = ISSNValidator()
        self.scholar_validator = GoogleScholarValidator()
        self.citation_fetcher = CitationFetcher()
        self.classifier = UnifiedPaperClassifier()
        
        # Verification thresholds
        self.min_confidence = 0.6
        self.high_confidence = 0.8

        # Control whether Google Scholar is used.
        # For large/batch operations we typically disable it to avoid
        # hitting rate limits or CAPTCHA / HTML changes that cause errors.
        self.enable_google_scholar: bool = True
    
    def verify_paper(self, paper: Dict[str, Any]) -> VerificationResult:
        """
        Verify a single paper using multiple methods.
        
        Args:
            paper: Paper dictionary with metadata
            
        Returns:
            VerificationResult object
        """
        paper_id = paper.get('id', 0)
        result = VerificationResult(
            paper_id=paper_id,
            status=VerificationStatus.PENDING,
            method_used="",
            confidence_score=0.0,
            verified_metadata={},
            errors=[],
            suggestions=[]
        )
        
        logger.info(f"Starting verification for paper {paper_id}: {paper.get('title', 'Unknown')}")
        
        # Try verification methods in order of preference
        verification_methods = [
            self._verify_by_doi,
            self._verify_by_issn,
            self._verify_by_author_title,
        ]

        # Google Scholar is an unofficial/fragile source; only include it
        # when explicitly enabled (e.g. for single‑paper checks or very
        # small batches). For large batches, callers should set
        # self.enable_google_scholar = False to avoid repeated errors.
        if self.enable_google_scholar:
            verification_methods.append(self._verify_by_google_scholar)
        
        for method in verification_methods:
            try:
                method_result = method(paper)
                if method_result and method_result.confidence_score >= self.min_confidence:
                    result = method_result
                    result.paper_id = paper_id
                    break
                elif method_result and method_result.confidence_score > result.confidence_score:
                    result = method_result
                    result.paper_id = paper_id
            except Exception as e:
                logger.warning(f"Verification method {method.__name__} failed: {e}")
                result.errors.append(f"{method.__name__}: {str(e)}")
        
        # Determine indexing status
        if result.verified_metadata:
            result.verified_metadata['indexing_status'] = self._determine_indexing_status(result.verified_metadata)
        
        # Determine final status
        if result.confidence_score >= self.high_confidence:
            result.status = VerificationStatus.VERIFIED
        elif result.confidence_score >= self.min_confidence:
            result.status = VerificationStatus.PARTIAL
        else:
            result.status = VerificationStatus.FAILED
        
        logger.info(f"Verification complete for paper {paper_id}: {result.status.value} (confidence: {result.confidence_score:.2f})")
        return result
    
    def _verify_by_doi(self, paper: Dict[str, Any]) -> Optional[VerificationResult]:
        """Verify paper using DOI."""
        doi = paper.get('doi', '').strip()
        if not doi:
            return None
        
        logger.info(f"Verifying by DOI: {doi}")
        
        try:
            crossref_metadata = self.crossref_fetcher.fetch_by_doi(doi)
            
            if not crossref_metadata.success:
                return VerificationResult(
                    paper_id=0,
                    status=VerificationStatus.FAILED,
                    method_used="doi",
                    confidence_score=0.0,
                    verified_metadata={},
                    errors=[crossref_metadata.error],
                    suggestions=[]
                )
            
            # Calculate confidence based on title match
            confidence = self._calculate_title_confidence(
                paper.get('title', ''),
                crossref_metadata.title
            )
            
            # Only use verified title if confidence is high enough
            title_to_use = crossref_metadata.title if confidence >= 0.8 else paper.get('title', '')
            
            verified_metadata = {
                'doi': crossref_metadata.doi,
                'title': title_to_use,
                'authors': crossref_metadata.authors,
                'journal': crossref_metadata.journal,
                'publisher': crossref_metadata.publisher,
                'issn': crossref_metadata.issn,
                'year': crossref_metadata.year,
                'abstract': crossref_metadata.abstract,
                'url': crossref_metadata.url,
                'type': crossref_metadata.type
            }
            
            return VerificationResult(
                paper_id=0,
                status=VerificationStatus.VERIFIED if confidence >= self.high_confidence else VerificationStatus.PARTIAL,
                method_used="doi",
                confidence_score=confidence,
                verified_metadata=verified_metadata,
                errors=[],
                suggestions=self._generate_suggestions(paper, verified_metadata)
            )
            
        except Exception as e:
            logger.error(f"DOI verification error: {e}")
            return VerificationResult(
                paper_id=0,
                status=VerificationStatus.FAILED,
                method_used="doi",
                confidence_score=0.0,
                verified_metadata={},
                errors=[f"DOI verification failed: {str(e)}"],
                suggestions=[]
            )
    
    def _verify_by_issn(self, paper: Dict[str, Any]) -> Optional[VerificationResult]:
        """Verify paper using ISSN."""
        # Extract ISSN from paper metadata
        issn = self._extract_issn_from_paper(paper)
        if not issn:
            return None
        
        logger.info(f"Verifying by ISSN: {issn}")
        
        try:
            issn_metadata = self.issn_validator.validate_by_issn(issn)
            
            if not issn_metadata.success:
                return VerificationResult(
                    paper_id=0,
                    status=VerificationStatus.FAILED,
                    method_used="issn",
                    confidence_score=0.0,
                    verified_metadata={},
                    errors=[issn_metadata.error],
                    suggestions=[]
                )
            
            # Calculate confidence based on journal match
            confidence = self._calculate_journal_confidence(
                paper.get('journal', ''),
                issn_metadata.title
            )
            
            verified_metadata = {
                'issn': issn_metadata.issn,
                'journal': issn_metadata.title,
                'publisher': issn_metadata.publisher,
                'country': issn_metadata.country,
                'language': issn_metadata.language,
                'subjects': issn_metadata.subjects,
                'is_open_access': issn_metadata.is_open_access,
                'license': issn_metadata.license,
                'apc_charges': issn_metadata.apc_charges
            }
            
            return VerificationResult(
                paper_id=0,
                status=VerificationStatus.VERIFIED if confidence >= self.high_confidence else VerificationStatus.PARTIAL,
                method_used="issn",
                confidence_score=confidence,
                verified_metadata=verified_metadata,
                errors=[],
                suggestions=self._generate_issn_suggestions(paper, verified_metadata)
            )
            
        except Exception as e:
            logger.error(f"ISSN verification error: {e}")
            return VerificationResult(
                paper_id=0,
                status=VerificationStatus.FAILED,
                method_used="issn",
                confidence_score=0.0,
                verified_metadata={},
                errors=[f"ISSN verification failed: {str(e)}"],
                suggestions=[]
            )
    
    def _verify_by_author_title(self, paper: Dict[str, Any]) -> Optional[VerificationResult]:
        """Verify paper using author and title search via Crossref."""
        title = paper.get('title', '').strip()
        authors = paper.get('authors', '').strip()
        
        if not title or len(title) < 10:
            return None
        
        logger.info(f"Verifying by author+title: {title[:50]}...")
        
        try:
            # Search Crossref by title and author
            results = self.crossref_fetcher.search_by_title_and_author(title, authors, limit=3)
            
            if not results:
                return VerificationResult(
                    paper_id=0,
                    status=VerificationStatus.FAILED,
                    method_used="author_title",
                    confidence_score=0.0,
                    verified_metadata={},
                    errors=["No results found in Crossref"],
                    suggestions=[]
                )
            
            # Use the best match
            best_match = results[0]
            match_score = getattr(best_match, 'match_score', 0.0)
            
            # Only use verified title if confidence is high enough
            title_to_use = best_match.title if match_score >= 0.8 else title
            
            verified_metadata = {
                'doi': best_match.doi,
                'title': title_to_use,
                'authors': best_match.authors,
                'journal': best_match.journal,
                'publisher': best_match.publisher,
                'issn': best_match.issn,
                'year': best_match.year,
                'abstract': best_match.abstract,
                'url': best_match.url,
                'type': best_match.type
            }
            
            return VerificationResult(
                paper_id=0,
                status=VerificationStatus.VERIFIED if match_score >= self.high_confidence else VerificationStatus.PARTIAL,
                method_used="author_title",
                confidence_score=match_score,
                verified_metadata=verified_metadata,
                errors=[],
                suggestions=self._generate_suggestions(paper, verified_metadata)
            )
            
        except Exception as e:
            logger.error(f"Author+title verification error: {e}")
            return VerificationResult(
                paper_id=0,
                status=VerificationStatus.FAILED,
                method_used="author_title",
                confidence_score=0.0,
                verified_metadata={},
                errors=[f"Author+title verification failed: {str(e)}"],
                suggestions=[]
            )
    
    def _verify_by_google_scholar(self, paper: Dict[str, Any]) -> Optional[VerificationResult]:
        """Verify paper using Google Scholar."""
        title = paper.get('title', '').strip()
        authors = paper.get('authors', '').strip()
        year = paper.get('year', 0)
        
        if not title or len(title) < 10:
            return None
        
        logger.info(f"Verifying by Google Scholar: {title[:50]}...")
        
        try:
            scholar_metadata = self.scholar_validator.search_paper(title, authors, year)
            
            if not scholar_metadata.success:
                return VerificationResult(
                    paper_id=0,
                    status=VerificationStatus.FAILED,
                    method_used="google_scholar",
                    confidence_score=0.0,
                    verified_metadata={},
                    errors=[scholar_metadata.error],
                    suggestions=[]
                )
            
            # Calculate confidence based on title match
            confidence = self._calculate_title_confidence(
                title,
                scholar_metadata.title
            )
            
            # Only use verified title if confidence is high enough
            title_to_use = scholar_metadata.title if confidence >= 0.8 else title
            
            verified_metadata = {
                'title': title_to_use,
                'authors': scholar_metadata.authors,
                'journal': scholar_metadata.journal,
                'publisher': scholar_metadata.publisher,
                'year': scholar_metadata.year,
                'abstract': scholar_metadata.abstract,
                'citations': scholar_metadata.citations,
                'pdf_link': scholar_metadata.pdf_link,
                'scholar_link': scholar_metadata.scholar_link
            }
            
            return VerificationResult(
                paper_id=0,
                status=VerificationStatus.VERIFIED if confidence >= self.high_confidence else VerificationStatus.PARTIAL,
                method_used="google_scholar",
                confidence_score=confidence,
                verified_metadata=verified_metadata,
                errors=[],
                suggestions=self._generate_suggestions(paper, verified_metadata)
            )
            
        except Exception as e:
            logger.error(f"Google Scholar verification error: {e}")
            return VerificationResult(
                paper_id=0,
                status=VerificationStatus.FAILED,
                method_used="google_scholar",
                confidence_score=0.0,
                verified_metadata={},
                errors=[f"Google Scholar verification failed: {str(e)}"],
                suggestions=[]
            )
    
    def _extract_issn_from_paper(self, paper: Dict[str, Any]) -> Optional[str]:
        """Extract ISSN from paper metadata."""
        # Check if ISSN is already in metadata
        metadata = paper.get('metadata', {})
        if isinstance(metadata, dict):
            issn = metadata.get('issn', '')
            if issn:
                return issn
        
        # Check journal field
        journal = paper.get('journal', '')
        if journal:
            issns = self.issn_validator.extract_issn_from_text(journal)
            if issns:
                return issns[0]
        
        # Check abstract
        abstract = paper.get('abstract', '')
        if abstract:
            issns = self.issn_validator.extract_issn_from_text(abstract)
            if issns:
                return issns[0]
        
        return None
    
    def _calculate_title_confidence(self, original_title: str, verified_title: str) -> float:
        """Calculate confidence based on title similarity."""
        if not original_title or not verified_title:
            return 0.0
        
        # Simple word overlap similarity
        orig_words = set(original_title.lower().split())
        verified_words = set(verified_title.lower().split())
        
        # Remove common words
        stop_words = {'a', 'an', 'the', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'and', 'or'}
        orig_words = orig_words - stop_words
        verified_words = verified_words - stop_words
        
        if not orig_words or not verified_words:
            return 0.0
        
        intersection = len(orig_words.intersection(verified_words))
        union = len(orig_words.union(verified_words))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_journal_confidence(self, original_journal: str, verified_journal: str) -> float:
        """Calculate confidence based on journal similarity."""
        return self._calculate_title_confidence(original_journal, verified_journal)
    
    def _generate_suggestions(self, original: Dict[str, Any], verified: Dict[str, Any]) -> List[str]:
        """Generate suggestions for metadata improvements."""
        suggestions = []
        
        # Title suggestions
        if verified.get('title') and verified['title'] != original.get('title', ''):
            suggestions.append(f"Consider updating title: '{verified['title']}'")
        
        # Author suggestions
        if verified.get('authors') and verified['authors'] != original.get('authors', ''):
            suggestions.append(f"Consider updating authors: '{verified['authors']}'")
        
        # DOI suggestions
        if verified.get('doi') and not original.get('doi'):
            suggestions.append(f"Add DOI: {verified['doi']}")
        
        # Journal suggestions
        if verified.get('journal') and verified['journal'] != original.get('journal', ''):
            suggestions.append(f"Consider updating journal: '{verified['journal']}'")
        
        # Year suggestions
        if verified.get('year') and verified['year'] != original.get('year', 0):
            suggestions.append(f"Consider updating year: {verified['year']}")
        
        return suggestions
    
    def _generate_issn_suggestions(self, original: Dict[str, Any], verified: Dict[str, Any]) -> List[str]:
        """Generate ISSN-specific suggestions."""
        suggestions = []
        
        # Journal title
        if verified.get('journal') and verified['journal'] != original.get('journal', ''):
            suggestions.append(f"Journal verified: '{verified['journal']}'")
        
        # Publisher
        if verified.get('publisher'):
            suggestions.append(f"Publisher: {verified['publisher']}")
        
        # Open access status
        if verified.get('is_open_access'):
            suggestions.append("This is an open access journal")
        
        # License information
        if verified.get('license'):
            suggestions.append(f"License: {verified['license']}")
        
        # APC charges
        if verified.get('apc_charges'):
            suggestions.append(f"APC charges: {verified['apc_charges']}")
        
        return suggestions
    
    def _determine_indexing_status(self, verified_metadata: Dict[str, Any]) -> str:
        """Determine indexing status using unified classifier."""
        classification = self.classifier.classify_paper(verified_metadata)
        return classification['indexing_status']
    
    def verify_papers_batch(self, papers: List[Dict[str, Any]]) -> List[VerificationResult]:
        """
        Verify multiple papers in batch.
        
        Args:
            papers: List of paper dictionaries
            
        Returns:
            List of VerificationResult objects
        """
        results = []
        
        logger.info(f"Starting batch verification of {len(papers)} papers")
        
        for i, paper in enumerate(papers):
            logger.info(f"Verifying paper {i+1}/{len(papers)}")
            result = self.verify_paper(paper)
            results.append(result)
        
        # Summary statistics
        verified_count = sum(1 for r in results if r.status == VerificationStatus.VERIFIED)
        partial_count = sum(1 for r in results if r.status == VerificationStatus.PARTIAL)
        failed_count = sum(1 for r in results if r.status == VerificationStatus.FAILED)
        
        logger.info(f"Batch verification complete: {verified_count} verified, {partial_count} partial, {failed_count} failed")
        
        return results


# Global instance
post_import_verifier = PostImportVerifier()


def verify_paper_post_import(paper: Dict[str, Any]) -> VerificationResult:
    """
    Convenience function to verify a single paper.
    
    Args:
        paper: Paper dictionary
        
    Returns:
        VerificationResult object
    """
    return post_import_verifier.verify_paper(paper)


def verify_papers_batch(papers: List[Dict[str, Any]]) -> List[VerificationResult]:
    """
    Convenience function to verify multiple papers.
    
    Args:
        papers: List of paper dictionaries
        
    Returns:
        List of VerificationResult objects
    """
    return post_import_verifier.verify_papers_batch(papers)
