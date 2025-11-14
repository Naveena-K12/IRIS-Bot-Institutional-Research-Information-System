"""
Paper Deleter
Provides functionality to delete papers from the database and optionally remove PDF files.
"""

import logging
import os
from typing import List, Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class PaperDeleter:
    """Handles paper deletion operations."""
    
    def __init__(self, paper_repository):
        """Initialize with paper repository."""
        self.paper_repo = paper_repository
    
    def delete_paper_by_id(self, paper_id: int, delete_pdf_file: bool = False) -> bool:
        """
        Delete a paper by its ID.
        
        Args:
            paper_id: ID of the paper to delete
            delete_pdf_file: Whether to also delete the PDF file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get paper details first
            paper = self.paper_repo.get_paper_by_id(paper_id)
            if not paper:
                logger.error(f"Paper with ID {paper_id} not found")
                return False
            
            file_path = paper.get('file_path', '')
            
            # Delete from database
            success = self._delete_from_database(paper_id)
            
            if success and delete_pdf_file and file_path:
                # Delete PDF file if requested
                self._delete_pdf_file(file_path)
            
            if success:
                logger.info(f"Successfully deleted paper ID {paper_id}")
            else:
                logger.error(f"Failed to delete paper ID {paper_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error deleting paper {paper_id}: {e}")
            return False
    
    def delete_papers_by_ids(self, paper_ids: List[int], delete_pdf_files: bool = False) -> Dict[str, int]:
        """
        Delete multiple papers by their IDs.
        
        Args:
            paper_ids: List of paper IDs to delete
            delete_pdf_files: Whether to also delete PDF files
            
        Returns:
            Dictionary with success/failure counts
        """
        results = {
            'success': 0,
            'failed': 0,
            'errors': []
        }
        
        for paper_id in paper_ids:
            try:
                if self.delete_paper_by_id(paper_id, delete_pdf_files):
                    results['success'] += 1
                else:
                    results['failed'] += 1
                    results['errors'].append(f"Failed to delete paper ID {paper_id}")
            except Exception as e:
                results['failed'] += 1
                results['errors'].append(f"Error deleting paper ID {paper_id}: {e}")
        
        logger.info(f"Batch delete completed: {results['success']} successful, {results['failed']} failed")
        return results
    
    def delete_papers_by_criteria(self, 
                                 title_contains: Optional[str] = None,
                                 author_contains: Optional[str] = None,
                                 department: Optional[str] = None,
                                 research_domain: Optional[str] = None,
                                 year: Optional[int] = None,
                                 delete_pdf_files: bool = False) -> Dict[str, int]:
        """
        Delete papers based on search criteria.
        
        Args:
            title_contains: Delete papers with title containing this text
            author_contains: Delete papers with author containing this text
            department: Delete papers from this department
            research_domain: Delete papers from this research domain
            year: Delete papers from this year
            delete_pdf_files: Whether to also delete PDF files
            
        Returns:
            Dictionary with success/failure counts
        """
        try:
            # Search for papers matching criteria
            papers = self.paper_repo.search_papers(
                query="",
                title=title_contains,
                authors=author_contains,
                department=department,
                research_domain=research_domain,
                year=year
            )
            
            if not papers:
                logger.info("No papers found matching deletion criteria")
                return {'success': 0, 'failed': 0, 'errors': []}
            
            # Get paper IDs
            paper_ids = [paper['id'] for paper in papers if paper.get('id')]
            
            logger.info(f"Found {len(paper_ids)} papers matching criteria")
            
            # Delete papers
            return self.delete_papers_by_ids(paper_ids, delete_pdf_files)
            
        except Exception as e:
            logger.error(f"Error deleting papers by criteria: {e}")
            return {'success': 0, 'failed': 1, 'errors': [str(e)]}
    
    def delete_all_papers(self, delete_pdf_files: bool = False) -> Dict[str, int]:
        """
        Delete all papers from the database.
        
        Args:
            delete_pdf_files: Whether to also delete PDF files
            
        Returns:
            Dictionary with success/failure counts
        """
        try:
            # Get all papers
            all_papers = self.paper_repo.search_papers("", limit=10000)
            
            if not all_papers:
                logger.info("No papers found in database")
                return {'success': 0, 'failed': 0, 'errors': []}
            
            paper_ids = [paper['id'] for paper in all_papers if paper.get('id')]
            
            logger.info(f"Deleting all {len(paper_ids)} papers")
            
            return self.delete_papers_by_ids(paper_ids, delete_pdf_files)
            
        except Exception as e:
            logger.error(f"Error deleting all papers: {e}")
            return {'success': 0, 'failed': 1, 'errors': [str(e)]}
    
    def _delete_from_database(self, paper_id: int) -> bool:
        """Delete paper from database using repository method."""
        try:
            # Use the repository's delete_paper method
            if hasattr(self.paper_repo, 'delete_paper'):
                return self.paper_repo.delete_paper(paper_id)
            else:
                logger.error("Repository does not have delete_paper method")
                return False
                
        except Exception as e:
            logger.error(f"Database deletion error: {e}")
            return False
    
    def _delete_pdf_file(self, file_path: str) -> bool:
        """Delete PDF file from filesystem."""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Deleted PDF file: {file_path}")
                return True
            else:
                logger.warning(f"PDF file not found: {file_path}")
                return False
        except Exception as e:
            logger.error(f"Error deleting PDF file {file_path}: {e}")
            return False
    
    def get_paper_info_for_deletion(self, paper_id: int) -> Optional[Dict[str, Any]]:
        """
        Get paper information for deletion confirmation.
        
        Args:
            paper_id: ID of the paper
            
        Returns:
            Dictionary with paper information or None if not found
        """
        try:
            paper = self.paper_repo.get_paper_by_id(paper_id)
            if not paper:
                return None
            
            file_path = paper.get('file_path', '')
            file_exists = os.path.exists(file_path) if file_path else False
            file_size = os.path.getsize(file_path) if file_exists else 0
            
            return {
                'id': paper_id,
                'title': paper.get('title', 'Unknown Title'),
                'authors': paper.get('authors', 'Unknown Authors'),
                'year': paper.get('year', 'Unknown Year'),
                'department': paper.get('department', 'Unknown Department'),
                'research_domain': paper.get('research_domain', 'Unknown Domain'),
                'file_path': file_path,
                'file_exists': file_exists,
                'file_size': file_size,
                'file_size_mb': round(file_size / (1024 * 1024), 2) if file_size > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting paper info for deletion: {e}")
            return None
    
    def cleanup_orphaned_files(self, papers_directory: str = "data/papers") -> Dict[str, int]:
        """
        Find and optionally delete PDF files that are not referenced in the database.
        
        Args:
            papers_directory: Directory containing PDF files
            
        Returns:
            Dictionary with cleanup results
        """
        try:
            # Get all papers from database
            all_papers = self.paper_repo.search_papers("", limit=10000)
            referenced_files = {paper.get('file_path', '') for paper in all_papers}
            
            # Get all PDF files in directory
            papers_dir = Path(papers_directory)
            if not papers_dir.exists():
                return {'orphaned_files': 0, 'errors': ['Papers directory not found']}
            
            all_pdf_files = list(papers_dir.glob("*.pdf"))
            orphaned_files = []
            
            for pdf_file in all_pdf_files:
                file_path = str(pdf_file.absolute())
                if file_path not in referenced_files:
                    orphaned_files.append(file_path)
            
            return {
                'orphaned_files': len(orphaned_files),
                'total_pdf_files': len(all_pdf_files),
                'referenced_files': len(referenced_files),
                'orphaned_file_list': orphaned_files
            }
            
        except Exception as e:
            logger.error(f"Error cleaning up orphaned files: {e}")
            return {'orphaned_files': 0, 'errors': [str(e)]}


def create_paper_deleter(paper_repository):
    """Create a PaperDeleter instance."""
    return PaperDeleter(paper_repository)
