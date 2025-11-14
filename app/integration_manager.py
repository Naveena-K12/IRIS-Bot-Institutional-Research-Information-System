"""
Integration Manager for Research Paper Browser v2.0
Coordinates all components and provides unified interface.
"""

import logging
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import json

from .database_unified import get_unified_database_manager, get_unified_paper_repository
from .utils.enhanced_pdf_extractor import extract_paper_metadata, get_extraction_stats
from .utils.metadata_enricher import enrich_paper_metadata, metadata_enricher
from .utils.duplicate_detector import duplicate_detector
from .config import DB_BACKEND, POSTGRES_DSN

logger = logging.getLogger(__name__)


class IntegrationManager:
    """Main integration manager for the Research Paper Browser v2.0."""
    
    def __init__(self):
        self.db_manager = get_unified_database_manager()
        self.paper_repo = get_unified_paper_repository()
        self.is_initialized = False
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize the system components."""
        try:
            # Initialize database
            logger.info("Initializing database...")
            
            # Train ML classifiers if we have existing data
            self._train_ml_classifiers()
            
            self.is_initialized = True
            logger.info("System initialization completed")
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            raise
    
    def _train_ml_classifiers(self):
        """Train ML classifiers on existing data."""
        try:
            # Get existing papers for training
            papers = self.paper_repo.search_papers("", limit=1000)
            
            if len(papers) < 10:  # Need minimum data for training
                logger.info("Insufficient data for ML training, skipping...")
                return
            
            # Prepare training data
            training_data = []
            for paper in papers:
                metadata = paper.get('metadata', {})
                abstract = paper.get('abstract', '')
                department = metadata.get('department', 'Unknown')
                domain = metadata.get('research_domain', 'Unknown')
                
                if abstract and department != 'Unknown' and domain != 'Unknown':
                    training_data.append((abstract, department, domain))
            
            if training_data:
                metadata_enricher.ml_tagger.train_classifiers(training_data)
                logger.info(f"Trained ML classifiers on {len(training_data)} samples")
            
        except Exception as e:
            logger.warning(f"ML training failed: {e}")
    
    def _check_for_duplicates(self, paper_data: Dict[str, Any]) -> List[Tuple[int, float, str]]:
        """
        Check if a paper has duplicates in the database.
        
        Args:
            paper_data: Paper data dictionary
            
        Returns:
            List of tuples (paper_id, similarity_score, reason) for duplicates found
        """
        try:
            # Get existing papers to check against
            existing_papers = self.paper_repo.search_papers("", limit=1000)
            
            if not existing_papers:
                return []
            
            # Find duplicates
            duplicates = duplicate_detector.find_duplicates(paper_data, existing_papers)
            
            return duplicates
            
        except Exception as e:
            logger.warning(f"Error checking for duplicates: {e}")
            return []
    
    def process_pdf_file(self, file_path: str, auto_import: bool = False) -> Dict[str, Any]:
        """
        Process a single PDF file through the complete pipeline.
        
        Args:
            file_path: Path to PDF file
            auto_import: Whether to automatically import without user review
            
        Returns:
            Dictionary with processing results
        """
        try:
            logger.info(f"Processing PDF: {file_path}")
            
            # Step 1: Extract metadata
            extracted = extract_paper_metadata(file_path)
            if not extracted or extracted.confidence < 0.1:
                return {
                    'success': False,
                    'error': 'Failed to extract meaningful metadata from PDF',
                    'extracted': extracted
                }
            
            # Step 2: Enrich metadata
            enriched = enrich_paper_metadata(
                extracted.title, extracted.authors, extracted.abstract,
                extracted.doi, extracted.journal, extracted.year
            )
            
            # Step 3: Prepare data for database
            paper_data = {
                'title': extracted.title,
                'authors': extracted.authors,
                'year': extracted.year,
                'abstract': extracted.abstract,
                'doi': extracted.doi,
                'journal': extracted.journal or enriched.journal_name,
                'publisher': extracted.publisher or enriched.publisher,
                'file_path': file_path,
                'full_text': extracted.full_text,
                'department': enriched.department,
                'research_domain': enriched.research_domain,
                'indexing_status': enriched.indexing_status,
                'keywords': extracted.keywords,
                'confidence': (extracted.confidence + enriched.confidence) / 2,
            }
            
            result = {
                'success': True,
                'extracted': extracted,
                'enriched': enriched,
                'paper_data': paper_data,
                'file_path': file_path
            }
            
            # Step 4: Check for duplicates before importing
            duplicate_info = self._check_for_duplicates(paper_data)
            if duplicate_info:
                result['duplicates_found'] = duplicate_info
                result['has_duplicates'] = True
                logger.info(f"Found {len(duplicate_info)} potential duplicate(s) for: {extracted.title}")
            
            # Step 5: Auto-import if requested (and no duplicates blocking)
            if auto_import:
                try:
                    # If duplicates found, mark the new paper as duplicate
                    if duplicate_info and len(duplicate_info) > 0:
                        # Mark as duplicate of the first (highest similarity) match
                        original_id, similarity, reason = duplicate_info[0]
                        paper_data['is_duplicate'] = True
                        paper_data['duplicate_of_id'] = original_id
                        paper_data['similarity_score'] = similarity
                        logger.info(f"Marking paper as duplicate of {original_id} (similarity: {similarity:.2f})")
                    
                    paper_id = self.paper_repo.add_paper(paper_data)
                    result['paper_id'] = paper_id
                    result['imported'] = True
                    logger.info(f"Auto-imported paper {paper_id}: {extracted.title}")
                except Exception as e:
                    result['import_error'] = str(e)
                    result['imported'] = False
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            return {
                'success': False,
                'error': str(e),
                'file_path': file_path
            }
    
    def process_multiple_pdfs(self, file_paths: List[str], 
                             auto_import: bool = False) -> Dict[str, Any]:
        """
        Process multiple PDF files.
        
        Args:
            file_paths: List of PDF file paths
            auto_import: Whether to automatically import without user review
            
        Returns:
            Dictionary with batch processing results
        """
        results = {
            'total_files': len(file_paths),
            'successful': 0,
            'failed': 0,
            'imported': 0,
            'errors': [],
            'papers': []
        }
        
        for file_path in file_paths:
            try:
                result = self.process_pdf_file(file_path, auto_import)
                
                if result['success']:
                    results['successful'] += 1
                    results['papers'].append(result)
                    
                    if result.get('imported', False):
                        results['imported'] += 1
                else:
                    results['failed'] += 1
                    results['errors'].append({
                        'file': file_path,
                        'error': result.get('error', 'Unknown error')
                    })
                    
            except Exception as e:
                results['failed'] += 1
                results['errors'].append({
                    'file': file_path,
                    'error': str(e)
                })
                logger.error(f"Error processing {file_path}: {e}")
        
        return results
    
    def search_papers(self, query: str = "", filters: Optional[Dict] = None, 
                     limit: int = 50) -> List[Dict[str, Any]]:
        """
        Search papers with advanced filtering.
        
        Args:
            query: Search query
            filters: Dictionary of filters
            limit: Maximum number of results
            
        Returns:
            List of paper dictionaries
        """
        try:
            return self.paper_repo.search_papers(query, filters, limit)
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def get_paper_by_id(self, paper_id: int) -> Optional[Dict[str, Any]]:
        """Get paper by ID."""
        try:
            return self.paper_repo.get_paper_by_id(paper_id)
        except Exception as e:
            logger.error(f"Error getting paper {paper_id}: {e}")
            return None
    
    def update_paper_metadata(self, paper_id: int, metadata: Dict[str, Any]) -> bool:
        """Update paper metadata."""
        try:
            return self.paper_repo.update_paper_metadata(paper_id, metadata)
        except Exception as e:
            logger.error(f"Error updating paper {paper_id}: {e}")
            return False
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        try:
            # Get basic counts
            all_papers = self.paper_repo.search_papers("", limit=10000)
            
            # Count by indexing status
            indexing_stats = {}
            department_stats = {}
            domain_stats = {}
            
            for paper in all_papers:
                metadata = paper.get('metadata', {})
                
                # Indexing status
                status = metadata.get('indexing_status', 'Unknown')
                indexing_stats[status] = indexing_stats.get(status, 0) + 1
                
                # Department
                dept = metadata.get('department', 'Unknown')
                department_stats[dept] = department_stats.get(dept, 0) + 1
                
                # Research domain
                domain = metadata.get('research_domain', 'Unknown')
                domain_stats[domain] = domain_stats.get(domain, 0) + 1
            
            return {
                'total_papers': len(all_papers),
                'indexing_status': indexing_stats,
                'departments': department_stats,
                'research_domains': domain_stats,
                'database_backend': DB_BACKEND,
                'ml_trained': metadata_enricher.ml_tagger.is_trained,
            }
            
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {'error': str(e)}
    
    def export_to_csv(self, output_path: str, filters: Optional[Dict] = None) -> bool:
        """
        Export papers to CSV file.
        
        Args:
            output_path: Path for output CSV file
            filters: Optional filters for papers to export
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import pandas as pd
            
            # Get papers
            papers = self.search_papers("", filters, limit=10000)
            
            if not papers:
                logger.warning("No papers to export")
                return False
            
            # Prepare data for CSV
            csv_data = []
            for paper in papers:
                metadata = paper.get('metadata', {})
                
                row = {
                    'id': paper.get('id'),
                    'title': paper.get('title'),
                    'authors': paper.get('authors'),
                    'year': paper.get('year'),
                    'abstract': paper.get('abstract'),
                    'doi': paper.get('doi'),
                    'journal': paper.get('journal'),
                    'publisher': paper.get('publisher'),
                    'file_path': paper.get('file_path'),
                    'department': metadata.get('department', ''),
                    'research_domain': metadata.get('research_domain', ''),
                    'indexing_status': metadata.get('indexing_status', ''),
                    'keywords': ', '.join(metadata.get('keywords', [])),
                    'confidence': metadata.get('confidence', 0.0),
                }
                csv_data.append(row)
            
            # Create DataFrame and save
            df = pd.DataFrame(csv_data)
            df.to_csv(output_path, index=False)
            
            logger.info(f"Exported {len(csv_data)} papers to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return False
    
    def detect_duplicates_batch(self, limit: int = 1000) -> Dict[str, Any]:
        """
        Detect and mark duplicates in the entire database.
        
        Args:
            limit: Maximum number of papers to check
            
        Returns:
            Dictionary with detection results
        """
        try:
            logger.info("Starting batch duplicate detection...")
            
            # Get all papers
            all_papers = self.paper_repo.search_papers("", limit=limit)
            
            if len(all_papers) < 2:
                logger.info("Not enough papers for duplicate detection")
                return {
                    'total_papers': len(all_papers),
                    'duplicates_found': 0,
                    'papers_checked': 0
                }
            
            duplicates_marked = 0
            papers_checked = 0
            
            for i, paper in enumerate(all_papers):
                # Skip if already marked as duplicate
                if paper.get('is_duplicate'):
                    continue
                
                # Check against remaining papers
                remaining_papers = all_papers[i+1:]
                duplicates = duplicate_detector.find_duplicates(paper, remaining_papers)
                
                papers_checked += 1
                
                # Mark the first duplicate found (highest similarity)
                if duplicates:
                    duplicate_candidate_id, similarity, reason = duplicates[0]
                    current_paper_id = paper.get('id')
                    
                    # Determine which one to mark as duplicate (keep the one with lower ID, i.e., older one)
                    if current_paper_id < duplicate_candidate_id:
                        # Current paper is older, mark the candidate as duplicate
                        update_data = {
                            'is_duplicate': True,
                            'duplicate_of_id': current_paper_id,
                            'similarity_score': similarity
                        }
                        self.paper_repo.update_paper_metadata(duplicate_candidate_id, update_data)
                        logger.info(f"Marked paper {duplicate_candidate_id} as duplicate of {current_paper_id} (similarity: {similarity:.2f})")
                    else:
                        # Candidate is older, mark the current paper as duplicate
                        update_data = {
                            'is_duplicate': True,
                            'duplicate_of_id': duplicate_candidate_id,
                            'similarity_score': similarity
                        }
                        self.paper_repo.update_paper_metadata(current_paper_id, update_data)
                        logger.info(f"Marked paper {current_paper_id} as duplicate of {duplicate_candidate_id} (similarity: {similarity:.2f})")
                    
                    duplicates_marked += 1
                
                # Progress logging
                if papers_checked % 100 == 0:
                    logger.info(f"Checked {papers_checked}/{len(all_papers)} papers, found {duplicates_marked} duplicates")
            
            logger.info(f"Batch duplicate detection completed: {duplicates_marked} duplicates found among {papers_checked} papers checked")
            
            return {
                'total_papers': len(all_papers),
                'duplicates_found': duplicates_marked,
                'papers_checked': papers_checked
            }
            
        except Exception as e:
            logger.error(f"Error in batch duplicate detection: {e}")
            return {
                'error': str(e),
                'total_papers': 0,
                'duplicates_found': 0,
                'papers_checked': 0
            }
    
    def delete_duplicates(self, keep_original: bool = True) -> Dict[str, Any]:
        """
        Delete papers marked as duplicates.
        
        Args:
            keep_original: If True, keep original papers and delete duplicates. If False, delete originals and keep duplicates.
            
        Returns:
            Dictionary with deletion results
        """
        try:
            logger.info("Starting duplicate deletion...")
            
            # Get all duplicate papers
            all_papers = self.paper_repo.search_papers("", limit=10000)
            
            duplicates_to_delete = []
            for paper in all_papers:
                if paper.get('is_duplicate'):
                    duplicates_to_delete.append(paper.get('id'))
            
            if not duplicates_to_delete:
                logger.info("No duplicates found to delete")
                return {
                    'duplicates_found': 0,
                    'deleted': 0,
                    'failed': 0
                }
            
            deleted_count = 0
            failed_count = 0
            
            for paper_id in duplicates_to_delete:
                try:
                    if self.paper_repo.delete_paper(paper_id):
                        deleted_count += 1
                        logger.info(f"Deleted duplicate paper {paper_id}")
                    else:
                        failed_count += 1
                        logger.warning(f"Failed to delete duplicate paper {paper_id}")
                except Exception as e:
                    failed_count += 1
                    logger.error(f"Error deleting duplicate paper {paper_id}: {e}")
            
            logger.info(f"Duplicate deletion completed: {deleted_count} deleted, {failed_count} failed")
            
            return {
                'duplicates_found': len(duplicates_to_delete),
                'deleted': deleted_count,
                'failed': failed_count
            }
            
        except Exception as e:
            logger.error(f"Error in duplicate deletion: {e}")
            return {
                'error': str(e),
                'duplicates_found': 0,
                'deleted': 0,
                'failed': 0
            }
    
    def close(self):
        """Close database connections."""
        if self.db_manager:
            self.db_manager.close()


# Global integration manager
integration_manager = IntegrationManager()


def get_integration_manager() -> IntegrationManager:
    """Get global integration manager instance."""
    return integration_manager

