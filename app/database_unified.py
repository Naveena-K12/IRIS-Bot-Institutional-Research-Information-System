"""
Unified Database Management for Research Paper Browser v3.0
Clean, normalized database structure with proper data organization.
"""

import logging
from typing import Optional, Union, Dict, Any, List, Tuple
from pathlib import Path
import json
from datetime import datetime

try:
    import psycopg2
    import psycopg2.extras
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

try:
    from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, Text, DateTime, Boolean, Float, ForeignKey
    from sqlalchemy.orm import sessionmaker, relationship
    from sqlalchemy.dialects.postgresql import TSVECTOR
    from sqlalchemy.exc import OperationalError
    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False

from .config import DB_BACKEND, POSTGRES_DSN, SQLITE_DB_PATH, ensure_directories_exist

logger = logging.getLogger(__name__)


class UnifiedDatabaseManager:
    """Unified database manager with clean, normalized structure."""
    
    def __init__(self, dsn: Optional[str] = None):
        self.dsn = dsn or POSTGRES_DSN
        self.engine = None
        self.session_factory = None
        self._setup_database()
    
    def _setup_database(self):
        """Setup database connection and schema."""
        if DB_BACKEND == "postgres":
            self._setup_postgresql()
        else:
            self._setup_sqlite()
    
    def _setup_postgresql(self):
        """Setup PostgreSQL database with normalized structure."""
        if not HAS_PSYCOPG2 or not HAS_SQLALCHEMY:
            raise ImportError("psycopg2 and SQLAlchemy are required for PostgreSQL backend")
        
        try:
            # Parse psycopg2 DSN format and convert to SQLAlchemy URL
            import re
            from urllib.parse import quote_plus
            
            dsn_parts = {}
            pattern = r'(\w+)=([^\s]+(?:\s+(?!\w+=)[^\s]+)*)'
            matches = re.findall(pattern, self.dsn)
            for key, value in matches:
                dsn_parts[key] = value.strip()
            
            password = quote_plus(dsn_parts.get('password', ''))
            db_url = f"postgresql+psycopg2://{dsn_parts.get('user', 'postgres')}:{password}@{dsn_parts.get('host', 'localhost')}:{dsn_parts.get('port', '5432')}/{dsn_parts.get('dbname', 'research_papers')}"
            self.engine = create_engine(db_url, echo=False, connect_args={"client_encoding": "utf8"})
            self.session_factory = sessionmaker(bind=self.engine)
            
            # Enable pgvector extension
            self._enable_pgvector()
            
            # Create tables
            self._create_postgresql_tables()
            
            logger.info("PostgreSQL unified database setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up PostgreSQL: {e}")
            raise
    
    def _setup_sqlite(self):
        """Setup SQLite database with normalized structure."""
        ensure_directories_exist()
        self.engine = create_engine(f"sqlite:///{SQLITE_DB_PATH}")
        self.session_factory = sessionmaker(bind=self.engine)
        self._create_sqlite_tables()
        logger.info("SQLite unified database setup completed")
    
    def _enable_pgvector(self):
        """Enable pgvector extension for vector operations."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.commit()
                logger.info("pgvector extension enabled successfully")
        except Exception as e:
            logger.warning(f"Could not enable pgvector extension: {e}")
            logger.info("Continuing without pgvector support...")
    
    def _create_postgresql_tables(self):
        """Create PostgreSQL tables with normalized structure."""
        metadata = MetaData()
        
        # Check if unified tables exist
        with self.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'papers_unified'
                )
            """))
            table_exists = result.fetchone()[0]
            
            if table_exists:
                logger.info("Unified tables already exist. Skipping creation.")
                return
        
        # Core papers table - only essential fields
        papers_table = Table(
            'papers_unified',
            metadata,
            Column('id', Integer, primary_key=True, autoincrement=True),
            Column('title', String(500), nullable=False),
            Column('authors', Text, nullable=False),
            Column('year', Integer, nullable=False),
            Column('abstract', Text),
            Column('doi', String(255)),
            Column('journal', String(255)),
            Column('publisher', String(255)),
            Column('file_path', String(500), nullable=False),
            Column('full_text', Text),
            
            # Full-text search
            Column('search_vector', TSVECTOR),
            
            # Vector embeddings for semantic search
            Column('abstract_embedding', String),
            
            # Duplicate detection
            Column('is_duplicate', Boolean, default=False),
            Column('duplicate_of_id', Integer),
            Column('similarity_score', Float),
            
            # Verification status
            Column('verification_status', String(50), default='pending'),
            Column('verification_method', String(50)),
            Column('verification_confidence', Float),
            Column('verification_date', DateTime),
            Column('last_verification_attempt', DateTime),
            
            # Timestamps
            Column('created_at', DateTime, server_default=text('NOW()')),
            Column('updated_at', DateTime, server_default=text('NOW()')),
        )
        
        # Paper metadata table - normalized metadata
        paper_metadata_table = Table(
            'paper_metadata',
            metadata,
            Column('id', Integer, primary_key=True, autoincrement=True),
            Column('paper_id', Integer, ForeignKey('papers_unified.id'), nullable=False),
            Column('department', String(255)),
            Column('research_domain', String(255)),
            Column('paper_type', String(100)),
            Column('student', String(100)),
            Column('review_status', String(100)),
            Column('indexing_status', String(100)),
            Column('issn', String(20)),
            Column('published_month', String(50)),
            Column('created_at', DateTime, server_default=text('NOW()')),
            Column('updated_at', DateTime, server_default=text('NOW()')),
        )
        
        # Citation data table - normalized citation information
        citation_data_table = Table(
            'citation_data',
            metadata,
            Column('id', Integer, primary_key=True, autoincrement=True),
            Column('paper_id', Integer, ForeignKey('papers_unified.id'), nullable=False),
            Column('citation_count', Integer, default=0),
            Column('scimago_quartile', String(10)),
            Column('impact_factor', Float, default=0.0),
            Column('h_index', Integer, default=0),
            Column('citation_source', String(100)),
            Column('citation_updated_at', DateTime),
            Column('created_at', DateTime, server_default=text('NOW()')),
            Column('updated_at', DateTime, server_default=text('NOW()')),
        )
        
        # Create tables
        metadata.create_all(self.engine)
        
        # Create indexes
        self._create_postgresql_indexes()
    
    def _create_sqlite_tables(self):
        """Create SQLite tables with normalized structure."""
        metadata = MetaData()
        
        # Core papers table
        papers_table = Table(
            'papers_unified',
            metadata,
            Column('id', Integer, primary_key=True, autoincrement=True),
            Column('title', String(500), nullable=False),
            Column('authors', Text, nullable=False),
            Column('year', Integer, nullable=False),
            Column('abstract', Text),
            Column('doi', String(255)),
            Column('journal', String(255)),
            Column('publisher', String(255)),
            Column('file_path', String(500), nullable=False),
            Column('full_text', Text),
            Column('is_duplicate', Boolean, default=False),
            Column('duplicate_of_id', Integer),
            Column('similarity_score', Float),
            Column('created_at', DateTime),
            Column('updated_at', DateTime),
        )
        
        # Paper metadata table
        paper_metadata_table = Table(
            'paper_metadata',
            metadata,
            Column('id', Integer, primary_key=True, autoincrement=True),
            Column('paper_id', Integer, ForeignKey('papers_unified.id'), nullable=False),
            Column('department', String(255)),
            Column('research_domain', String(255)),
            Column('paper_type', String(100)),
            Column('student', String(100)),
            Column('review_status', String(100)),
            Column('indexing_status', String(100)),
            Column('issn', String(20)),
            Column('published_month', String(50)),
            Column('created_at', DateTime),
            Column('updated_at', DateTime),
        )
        
        # Citation data table
        citation_data_table = Table(
            'citation_data',
            metadata,
            Column('id', Integer, primary_key=True, autoincrement=True),
            Column('paper_id', Integer, ForeignKey('papers_unified.id'), nullable=False),
            Column('citation_count', Integer, default=0),
            Column('scimago_quartile', String(10)),
            Column('impact_factor', Float, default=0.0),
            Column('h_index', Integer, default=0),
            Column('citation_source', String(100)),
            Column('citation_updated_at', DateTime),
            Column('created_at', DateTime),
            Column('updated_at', DateTime),
        )
        
        metadata.create_all(self.engine)
    
    def _create_postgresql_indexes(self):
        """Create PostgreSQL indexes for performance."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_papers_unified_year ON papers_unified(year)",
            "CREATE INDEX IF NOT EXISTS idx_papers_unified_journal ON papers_unified(journal)",
            "CREATE INDEX IF NOT EXISTS idx_papers_unified_doi ON papers_unified(doi)",
            "CREATE INDEX IF NOT EXISTS idx_papers_unified_duplicates ON papers_unified(is_duplicate)",
            "CREATE INDEX IF NOT EXISTS idx_papers_unified_search_vector ON papers_unified USING GIN(search_vector)",
            "CREATE INDEX IF NOT EXISTS idx_papers_unified_created_at ON papers_unified(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_paper_metadata_paper_id ON paper_metadata(paper_id)",
            "CREATE INDEX IF NOT EXISTS idx_paper_metadata_department ON paper_metadata(department)",
            "CREATE INDEX IF NOT EXISTS idx_paper_metadata_research_domain ON paper_metadata(research_domain)",
            "CREATE INDEX IF NOT EXISTS idx_citation_data_paper_id ON citation_data(paper_id)",
        ]
        
        try:
            with self.engine.connect() as conn:
                for index_sql in indexes:
                    conn.execute(text(index_sql))
                conn.commit()
        except Exception as e:
            logger.warning(f"Error creating indexes: {e}")
    
    def get_session(self):
        """Get database session."""
        return self.session_factory()
    
    def close(self):
        """Close database connections."""
        if self.engine:
            self.engine.dispose()


class UnifiedPaperRepository:
    """Unified paper repository with normalized data access."""
    
    def __init__(self, db_manager: UnifiedDatabaseManager):
        self.db_manager = db_manager
    
    def add_paper(self, paper_data: Dict[str, Any]) -> int:
        """
        Add a new paper to the unified database structure.
        
        Args:
            paper_data: Dictionary containing paper information
            
        Returns:
            ID of the created paper
        """
        with self.db_manager.get_session() as session:
            try:
                # Insert main paper record
                if DB_BACKEND == "postgres":
                    result = session.execute(text("""
                        INSERT INTO papers_unified (title, authors, year, abstract, doi, journal, publisher, 
                                                  file_path, full_text, search_vector, is_duplicate, 
                                                  duplicate_of_id, similarity_score)
                        VALUES (:title, :authors, :year, :abstract, :doi, :journal, :publisher,
                                :file_path, :full_text, to_tsvector('english', :search_text),
                                :is_duplicate, :duplicate_of_id, :similarity_score)
                        RETURNING id
                    """), {
                        'title': paper_data.get('title'),
                        'authors': paper_data.get('authors'),
                        'year': paper_data.get('year'),
                        'abstract': paper_data.get('abstract'),
                        'doi': paper_data.get('doi'),
                        'journal': paper_data.get('journal'),
                        'publisher': paper_data.get('publisher'),
                        'file_path': paper_data.get('file_path'),
                        'full_text': paper_data.get('full_text'),
                        'search_text': ' '.join([
                            paper_data.get('title', ''),
                            paper_data.get('authors', ''),
                            paper_data.get('abstract', ''),
                            paper_data.get('full_text', ''),
                        ]),
                        'is_duplicate': paper_data.get('is_duplicate', False),
                        'duplicate_of_id': paper_data.get('duplicate_of_id'),
                        'similarity_score': paper_data.get('similarity_score'),
                    })
                    paper_id = result.fetchone()[0]
                else:
                    # SQLite version
                    result = session.execute(text("""
                        INSERT INTO papers_unified (title, authors, year, abstract, doi, journal, publisher,
                                                  file_path, full_text, is_duplicate, duplicate_of_id, similarity_score)
                        VALUES (:title, :authors, :year, :abstract, :doi, :journal, :publisher,
                                :file_path, :full_text, :is_duplicate, :duplicate_of_id, :similarity_score)
                    """), {
                        'title': paper_data.get('title'),
                        'authors': paper_data.get('authors'),
                        'year': paper_data.get('year'),
                        'abstract': paper_data.get('abstract'),
                        'doi': paper_data.get('doi'),
                        'journal': paper_data.get('journal'),
                        'publisher': paper_data.get('publisher'),
                        'file_path': paper_data.get('file_path'),
                        'full_text': paper_data.get('full_text'),
                        'is_duplicate': paper_data.get('is_duplicate', False),
                        'duplicate_of_id': paper_data.get('duplicate_of_id'),
                        'similarity_score': paper_data.get('similarity_score'),
                    })
                    paper_id = result.lastrowid
                
                # Insert metadata
                self._insert_paper_metadata(session, paper_id, paper_data)
                
                # Insert citation data
                self._insert_citation_data(session, paper_id, paper_data)
                
                session.commit()
                return paper_id
                
            except Exception as e:
                session.rollback()
                logger.error(f"Error adding paper: {e}")
                raise
    
    def _insert_paper_metadata(self, session, paper_id: int, paper_data: Dict[str, Any]):
        """Insert paper metadata into normalized table."""
        metadata_fields = {
            'department': paper_data.get('department', ''),
            'research_domain': paper_data.get('research_domain', ''),
            'paper_type': paper_data.get('paper_type', ''),
            'student': paper_data.get('student', ''),
            'review_status': paper_data.get('review_status', ''),
            'indexing_status': paper_data.get('indexing_status', ''),
            'issn': paper_data.get('issn', ''),
            'published_month': paper_data.get('published_month', ''),
        }
        
        # Only insert if there's actual metadata
        if any(metadata_fields.values()):
            session.execute(text("""
                INSERT INTO paper_metadata (paper_id, department, research_domain, paper_type, 
                                          student, review_status, indexing_status, issn, published_month)
                VALUES (:paper_id, :department, :research_domain, :paper_type, :student, 
                        :review_status, :indexing_status, :issn, :published_month)
            """), {'paper_id': paper_id, **metadata_fields})
    
    def _insert_citation_data(self, session, paper_id: int, paper_data: Dict[str, Any]):
        """Insert citation data into normalized table."""
        citation_fields = {
            'citation_count': paper_data.get('citation_count', 0),
            'scimago_quartile': paper_data.get('scimago_quartile', ''),
            'impact_factor': paper_data.get('impact_factor', 0.0),
            'h_index': paper_data.get('h_index', 0),
            'citation_source': paper_data.get('citation_source', ''),
            'citation_updated_at': paper_data.get('citation_updated_at'),
        }
        
        # Only insert if there's actual citation data
        if any(citation_fields.values()):
            session.execute(text("""
                INSERT INTO citation_data (paper_id, citation_count, scimago_quartile, impact_factor, 
                                         h_index, citation_source, citation_updated_at)
                VALUES (:paper_id, :citation_count, :scimago_quartile, :impact_factor, 
                        :h_index, :citation_source, :citation_updated_at)
            """), {'paper_id': paper_id, **citation_fields})
    
    def search_papers(self, query: str, filters: Optional[Dict] = None, 
                     limit: int = 50) -> List[Dict[str, Any]]:
        """
        Search papers with full-text search and metadata filtering.
        
        Args:
            query: Search query
            filters: Dictionary of filters (year, journal, indexing_status, etc.)
            limit: Maximum number of results
            
        Returns:
            List of paper dictionaries with normalized data
        """
        with self.db_manager.get_session() as session:
            try:
                # Build search query with joins
                search_conditions = []
                params = {'query': query, 'limit': limit}
                
                # Full-text search
                if query:
                    if DB_BACKEND == "postgres":
                        search_conditions.append("p.search_vector @@ plainto_tsquery('english', :query)")
                    else:
                        search_conditions.append("(p.title LIKE :query OR p.authors LIKE :query OR p.abstract LIKE :query OR p.full_text LIKE :query)")
                        params['query'] = f"%{query}%"
                
                # Metadata filters
                if filters:
                    if 'year' in filters:
                        search_conditions.append("p.year = :year")
                        params['year'] = filters['year']
                    
                    if 'journal' in filters:
                        search_conditions.append("p.journal ILIKE :journal")
                        params['journal'] = f"%{filters['journal']}%"
                    
                    if 'indexing_status' in filters:
                        search_conditions.append("pm.indexing_status = :indexing_status")
                        params['indexing_status'] = filters['indexing_status']
                    
                    if 'department' in filters:
                        search_conditions.append("pm.department = :department")
                        params['department'] = filters['department']
                    if 'published_month' in filters:
                        search_conditions.append("pm.published_month = :published_month")
                        params['published_month'] = filters['published_month']
                
                # Build final query with joins
                where_clause = " AND ".join(search_conditions) if search_conditions else "1=1"
                
                if DB_BACKEND == "postgres":
                    sql = f"""
                        SELECT p.id, p.title, p.authors, p.year, p.abstract, p.doi, p.journal, p.publisher,
                               p.file_path, p.full_text, p.is_duplicate, p.duplicate_of_id, p.similarity_score,
                               p.verification_status, p.verification_method, p.verification_confidence, 
                               p.verification_date, p.last_verification_attempt,
                               pm.department, pm.research_domain, pm.paper_type, pm.student, pm.review_status, 
                               pm.indexing_status, pm.issn, pm.published_month,
                               cd.citation_count, cd.scimago_quartile, cd.impact_factor, cd.h_index, 
                               cd.citation_source, cd.citation_updated_at,
                               ts_rank(p.search_vector, plainto_tsquery('english', :query)) as rank
                        FROM papers_unified p
                        LEFT JOIN paper_metadata pm ON p.id = pm.paper_id
                        LEFT JOIN citation_data cd ON p.id = cd.paper_id
                        WHERE {where_clause}
                        ORDER BY rank DESC, p.year DESC
                        LIMIT :limit
                    """
                else:
                    sql = f"""
                        SELECT p.id, p.title, p.authors, p.year, p.abstract, p.doi, p.journal, p.publisher,
                               p.file_path, p.full_text, p.is_duplicate, p.duplicate_of_id, p.similarity_score,
                               pm.department, pm.research_domain, pm.paper_type, pm.student, pm.review_status, 
                               pm.indexing_status, pm.issn, pm.published_month,
                               cd.citation_count, cd.scimago_quartile, cd.impact_factor, cd.h_index, 
                               cd.citation_source, cd.citation_updated_at
                        FROM papers_unified p
                        LEFT JOIN paper_metadata pm ON p.id = pm.paper_id
                        LEFT JOIN citation_data cd ON p.id = cd.paper_id
                        WHERE {where_clause}
                        ORDER BY p.year DESC
                        LIMIT :limit
                    """
                
                result = session.execute(text(sql), params)
                rows = result.fetchall()
                
                # Convert to dictionaries
                papers = []
                for row in rows:
                    paper_dict = dict(row._mapping)
                    
                    # Handle None values for numeric fields that might be used in comparisons
                    if paper_dict.get('citation_count') is None:
                        paper_dict['citation_count'] = 0
                    if paper_dict.get('impact_factor') is None:
                        paper_dict['impact_factor'] = 0.0
                    if paper_dict.get('h_index') is None:
                        paper_dict['h_index'] = 0
                    if paper_dict.get('verification_confidence') is None:
                        paper_dict['verification_confidence'] = 0.0
                    
                    papers.append(paper_dict)
                
                return papers
                
            except Exception as e:
                logger.error(f"Error searching papers: {e}")
                return []
    
    def get_paper_by_id(self, paper_id: int) -> Optional[Dict[str, Any]]:
        """Get paper by ID with all related data."""
        with self.db_manager.get_session() as session:
            try:
                if DB_BACKEND == "postgres":
                    result = session.execute(text("""
                        SELECT p.id, p.title, p.authors, p.year, p.abstract, p.doi, p.journal, p.publisher,
                               p.file_path, p.full_text, p.is_duplicate, p.duplicate_of_id, p.similarity_score,
                               pm.department, pm.research_domain, pm.paper_type, pm.student, pm.review_status, 
                               pm.indexing_status, pm.issn, pm.published_month,
                               cd.citation_count, cd.scimago_quartile, cd.impact_factor, cd.h_index, 
                               cd.citation_source, cd.citation_updated_at
                        FROM papers_unified p
                        LEFT JOIN paper_metadata pm ON p.id = pm.paper_id
                        LEFT JOIN citation_data cd ON p.id = cd.paper_id
                        WHERE p.id = :paper_id
                    """), {'paper_id': paper_id})
                else:
                    result = session.execute(text("""
                        SELECT p.id, p.title, p.authors, p.year, p.abstract, p.doi, p.journal, p.publisher,
                               p.file_path, p.full_text, p.is_duplicate, p.duplicate_of_id, p.similarity_score,
                               pm.department, pm.research_domain, pm.paper_type, pm.student, pm.review_status, 
                               pm.indexing_status, pm.issn, pm.published_month,
                               cd.citation_count, cd.scimago_quartile, cd.impact_factor, cd.h_index, 
                               cd.citation_source, cd.citation_updated_at
                        FROM papers_unified p
                        LEFT JOIN paper_metadata pm ON p.id = pm.paper_id
                        LEFT JOIN citation_data cd ON p.id = cd.paper_id
                        WHERE p.id = :paper_id
                    """), {'paper_id': paper_id})
                
                row = result.fetchone()
                if row:
                    return dict(row._mapping)
                return None
                
            except Exception as e:
                logger.error(f"Error getting paper {paper_id}: {e}")
                return None
    
    def list_all(self) -> List[Dict[str, Any]]:
        """List all papers with normalized data."""
        return self.search_papers("", limit=10000)
    
    def update_verification_status(self, paper_id: int, status: str, method: str, 
                                 confidence: float, verified_metadata: Dict[str, Any]) -> bool:
        """Update verification status for a paper with retry on transient DB errors."""
        for attempt in range(3):
            session = self.db_manager.get_session()
            try:
                # Update verification fields in main table
                verification_updates = {
                    'verification_status': status,
                    'verification_method': method,
                    'verification_confidence': confidence,
                    'verification_date': datetime.utcnow(),
                    'last_verification_attempt': datetime.utcnow()
                }
                
                if DB_BACKEND == "postgres":
                    verification_updates['updated_at'] = 'NOW()'
                else:
                    verification_updates['updated_at'] = datetime.utcnow()
                
                verification_updates['paper_id'] = paper_id
                
                session.execute(text("""
                    UPDATE papers_unified 
                    SET verification_status = :verification_status,
                        verification_method = :verification_method,
                        verification_confidence = :verification_confidence,
                        verification_date = :verification_date,
                        last_verification_attempt = :last_verification_attempt,
                        updated_at = :updated_at
                    WHERE id = :paper_id
                """), verification_updates)
                
                # Update metadata if provided
                if verified_metadata:
                    # Use same session to keep transaction boundaries
                    self._update_within_session(session, paper_id, verified_metadata)
                
                session.commit()
                return True
            except OperationalError as e:
                session.rollback()
                logger.warning(f"Transient DB error on verification update (attempt {attempt+1}/3): {e}")
                if attempt == 2:
                    logger.error("Giving up after retries for verification update")
                    return False
            except Exception as e:
                session.rollback()
                logger.error(f"Error updating verification status: {e}")
                return False
            finally:
                session.close()
        return False

    def _update_within_session(self, session, paper_id: int, updates: Dict[str, Any]) -> None:
        """Update paper/metadata/citation using an existing session (no commit)."""
        paper_fields = ['title', 'authors', 'year', 'abstract', 'doi', 'journal', 'publisher', 'file_path', 'full_text', 
                       'is_duplicate', 'duplicate_of_id', 'similarity_score']
        metadata_fields = ['department', 'research_domain', 'paper_type', 'student', 'review_status', 'indexing_status', 'issn', 'published_month']
        citation_fields = ['citation_count', 'scimago_quartile', 'impact_factor', 'h_index', 'citation_source', 'citation_updated_at']
        
        # Update main paper table
        paper_updates = {k: v for k, v in updates.items() if k in paper_fields and v is not None}
        if paper_updates:
            set_clauses = [f"{k} = :{k}" for k in paper_updates.keys()]
            if DB_BACKEND == "postgres":
                set_clauses.append("updated_at = NOW()")
            else:
                set_clauses.append("updated_at = :updated_at")
                paper_updates['updated_at'] = 'NOW()'
            paper_updates['paper_id'] = paper_id
            session.execute(text(f"UPDATE papers_unified SET {', '.join(set_clauses)} WHERE id = :paper_id"), paper_updates)
        
        # Update metadata table
        metadata_updates = {k: v for k, v in updates.items() if k in metadata_fields and v is not None}
        if metadata_updates:
            result = session.execute(text("SELECT id FROM paper_metadata WHERE paper_id = :paper_id"), {'paper_id': paper_id})
            if result.fetchone():
                set_clauses = [f"{k} = :{k}" for k in metadata_updates.keys()]
                if DB_BACKEND == "postgres":
                    set_clauses.append("updated_at = NOW()")
                else:
                    set_clauses.append("updated_at = :updated_at")
                    metadata_updates['updated_at'] = 'NOW()'
                metadata_updates['paper_id'] = paper_id
                session.execute(text(f"UPDATE paper_metadata SET {', '.join(set_clauses)} WHERE paper_id = :paper_id"), metadata_updates)
            else:
                # Insert new metadata record
                session.execute(text("""
                    INSERT INTO paper_metadata (paper_id, department, research_domain, paper_type, 
                                              student, review_status, indexing_status, issn, published_month)
                    VALUES (:paper_id, :department, :research_domain, :paper_type, :student, 
                            :review_status, :indexing_status, :issn, :published_month)
                """), {'paper_id': paper_id, **metadata_updates})
        
        # Update/insert citation data
        citation_updates = {k: v for k, v in updates.items() if k in citation_fields and v is not None}
        if citation_updates:
            result = session.execute(text("SELECT id FROM citation_data WHERE paper_id = :paper_id"), {'paper_id': paper_id})
            if result.fetchone():
                set_clauses = [f"{k} = :{k}" for k in citation_updates.keys()]
                if DB_BACKEND == "postgres":
                    set_clauses.append("updated_at = NOW()")
                else:
                    set_clauses.append("updated_at = :updated_at")
                    citation_updates['updated_at'] = 'NOW()'
                citation_updates['paper_id'] = paper_id
                session.execute(text(f"UPDATE citation_data SET {', '.join(set_clauses)} WHERE paper_id = :paper_id"), citation_updates)
            else:
                session.execute(text("""
                    INSERT INTO citation_data (paper_id, citation_count, scimago_quartile, impact_factor, 
                                             h_index, citation_source, citation_updated_at)
                    VALUES (:paper_id, :citation_count, :scimago_quartile, :impact_factor, 
                            :h_index, :citation_source, :citation_updated_at)
                """), {'paper_id': paper_id, **citation_updates})

    def update_paper_metadata(self, paper_id: int, updates: Dict[str, Any]) -> bool:
        """Update paper metadata in normalized structure with retry on transient DB errors."""
        for attempt in range(3):
            session = self.db_manager.get_session()
            try:
                # Reuse helper to perform updates without committing
                self._update_within_session(session, paper_id, updates)
                session.commit()
                return True
            except OperationalError as e:
                session.rollback()
                logger.warning(f"Transient DB error on metadata update (attempt {attempt+1}/3): {e}")
                if attempt == 2:
                    logger.error("Giving up after retries for metadata update")
                    return False
            except Exception as e:
                session.rollback()
                logger.error(f"Error updating paper metadata: {e}")
                return False
            finally:
                session.close()
        return False
    
    def delete_paper(self, paper_id: int) -> bool:
        """Delete a paper and all related data from the unified database."""
        with self.db_manager.get_session() as session:
            try:
                # Delete from all related tables in correct order (due to foreign keys)
                # 1. Delete citation data
                session.execute(text("DELETE FROM citation_data WHERE paper_id = :paper_id"), {'paper_id': paper_id})
                
                # 2. Delete paper metadata
                session.execute(text("DELETE FROM paper_metadata WHERE paper_id = :paper_id"), {'paper_id': paper_id})
                
                # 3. Delete main paper record
                result = session.execute(text("DELETE FROM papers_unified WHERE id = :paper_id"), {'paper_id': paper_id})
                
                # Check if any rows were affected
                if result.rowcount > 0:
                    session.commit()
                    logger.info(f"Successfully deleted paper {paper_id}")
                    return True
                else:
                    logger.warning(f"Paper {paper_id} not found for deletion")
                    return False
                    
            except Exception as e:
                session.rollback()
                logger.error(f"Error deleting paper {paper_id}: {e}")
                return False


# Global unified database manager
unified_db_manager = UnifiedDatabaseManager()


def get_unified_database_manager() -> UnifiedDatabaseManager:
    """Get global unified database manager instance."""
    return unified_db_manager


def get_unified_paper_repository() -> UnifiedPaperRepository:
    """Get unified paper repository instance."""
    return UnifiedPaperRepository(unified_db_manager)
