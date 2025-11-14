"""
Enhanced Main Window for Research Paper Browser v2.0
Features automated PDF processing, metadata correction UI, and advanced search.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import json
from datetime import datetime

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QTableWidget, QTableWidgetItem, QMenu,
    QFileDialog, QMessageBox, QProgressDialog, QDialog, QLabel,
    QCheckBox, QComboBox, QSpinBox, QTextEdit, QGroupBox,
    QSplitter, QTabWidget, QFormLayout, QScrollArea, QFrame,
    QHeaderView, QButtonGroup, QRadioButton, QStyledItemDelegate, QStyleOptionViewItem
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QFont, QPixmap, QIcon, QColor, QTextOption

from ..config import APP_NAME, DB_BACKEND
from ..database_unified import get_unified_paper_repository
from ..integration_manager import get_integration_manager
from ..utils.enhanced_pdf_extractor import extract_paper_metadata, get_extraction_stats
from ..utils.metadata_enricher import enrich_paper_metadata
from ..utils.pdf_opener import open_pdf
from ..utils.department_manager import get_all_departments, get_departments_for_dropdown
from ..utils.hybrid_search_engine import HybridSearchEngine
from ..utils.semantic_search_engine import SemanticSearchEngine
from .smart_verification_dialog import SmartVerificationDialog
from .duplicate_management_dialog import DuplicateManagementDialog
from ..utils.pdf_table_generator import generate_paper_table_pdf, generate_summary_pdf

logger = logging.getLogger(__name__)


class WrappingItemDelegate(QStyledItemDelegate):
    """Delegate that enables word-wrap and computes proper row height."""
    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        option.textElideMode = Qt.ElideNone
        # Enable wrapping on the style option (Qt6 uses boolean wrapText)
        try:
            option.wrapText = True
        except Exception:
            # Fallback: rely on table's wordWrap and ResizeToContents
            pass

    def paint(self, painter, option, index):
        option.textElideMode = Qt.ElideNone
        super().paint(painter, option, index)

    def sizeHint(self, option, index):
        # Let Qt compute height for wrapped text based on column width
        option.textElideMode = Qt.ElideNone
        return super().sizeHint(option, index)

class PDFProcessingThread(QThread):
    """Thread for processing PDF files in the background."""
    
    progress_updated = Signal(str, int, int)  # status, current, total
    processing_completed = Signal(dict)  # results
    error_occurred = Signal(str)  # error message
    
    def __init__(self, file_paths: List[str]):
        super().__init__()
        self.file_paths = file_paths
        self.results = []
    
    def run(self):
        """Process PDF files."""
        try:
            total_files = len(self.file_paths)
            
            for i, file_path in enumerate(self.file_paths):
                self.progress_updated.emit(f"Processing {Path(file_path).name}...", i, total_files)
                
                try:
                    # Extract metadata
                    metadata = extract_paper_metadata(file_path)
                    
                    # Enrich metadata
                    enriched = enrich_paper_metadata(
                        metadata.title, metadata.authors, metadata.abstract,
                        metadata.doi, metadata.journal, metadata.year
                    )
                    
                    result = {
                        'file_path': file_path,
                        'extracted': metadata,
                        'enriched': enriched,
                        'success': True
                    }
                    
                except Exception as e:
                    result = {
                        'file_path': file_path,
                        'error': str(e),
                        'success': False
                    }
                
                self.results.append(result)
            
            self.processing_completed.emit({'results': self.results})
            
        except Exception as e:
            self.error_occurred.emit(str(e))


class MetadataCorrectionDialog(QDialog):
    """Dialog for reviewing and correcting extracted metadata."""
    
    def __init__(self, extracted_data: Dict, enriched_data: Dict, parent=None):
        super().__init__(parent)
        self.extracted_data = extracted_data
        self.enriched_data = enriched_data
        self.setWindowTitle("Review and Correct Metadata")
        self.setModal(True)
        self.resize(800, 600)
        
        self._setup_ui()
        self._populate_data()
    
    def _setup_ui(self):
        """Setup the correction UI."""
        layout = QVBoxLayout(self)
        
        # Create tab widget
        tab_widget = QTabWidget()
        
        # Basic Information tab
        basic_tab = QWidget()
        basic_layout = QFormLayout(basic_tab)
        
        self.title_edit = QLineEdit()
        self.authors_edit = QLineEdit()
        self.abstract_edit = QTextEdit()
        self.abstract_edit.setMaximumHeight(150)
        self.year_spin = QSpinBox()
        self.year_spin.setRange(1900, 2030)
        # Published month selector
        self.published_month_combo = QComboBox()
        self.published_month_combo.addItems([
            "", "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ])
        self.doi_edit = QLineEdit()
        self.journal_edit = QLineEdit()
        self.publisher_edit = QLineEdit()
        
        basic_layout.addRow("Title:", self.title_edit)
        basic_layout.addRow("Authors:", self.authors_edit)
        basic_layout.addRow("Abstract:", self.abstract_edit)
        basic_layout.addRow("Year:", self.year_spin)
        basic_layout.addRow("Published Month:", self.published_month_combo)
        basic_layout.addRow("DOI:", self.doi_edit)
        basic_layout.addRow("Journal:", self.journal_edit)
        basic_layout.addRow("Publisher:", self.publisher_edit)
        
        tab_widget.addTab(basic_tab, "Basic Information")
        
        # Classification tab
        class_tab = QWidget()
        class_layout = QFormLayout(class_tab)
        
        self.indexing_status_combo = QComboBox()
        self.indexing_status_combo.addItems(["SCI", "Scopus", "SCI + Scopus", "Non-SCI/Non-Scopus"])
        
        # Department dropdown
        self.department_combo = QComboBox()
        self.department_combo.setEditable(True)
        self.department_combo.setPlaceholderText("Select or enter department...")
        
        # Research domain dropdown
        self.research_domain_combo = QComboBox()
        self.research_domain_combo.setEditable(True)
        self.research_domain_combo.setPlaceholderText("Select or enter research domain...")
        
        # Published month filter combo (define before using in layout)
        self.published_month_combo = QComboBox()
        self.published_month_combo.addItems([
            "All", "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ])
        
        # Published month filter
        self.published_month_combo = QComboBox()
        self.published_month_combo.addItems([
            "All", "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ])
        # Published month filter
        self.published_month_combo = QComboBox()
        self.published_month_combo.addItems([
            "All", "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ])
        
        # Keywords
        self.keywords_edit = QTextEdit()
        self.keywords_edit.setMaximumHeight(100)
        self.keywords_edit.setPlaceholderText("Enter keywords separated by commas")
        
        class_layout.addRow("Indexing Status:", self.indexing_status_combo)
        class_layout.addRow("Department:", self.department_combo)
        class_layout.addRow("Research Domain:", self.research_domain_combo)
        class_layout.addRow("Keywords:", self.keywords_edit)
        
        # Load department options
        self._load_classification_options()
        
        tab_widget.addTab(class_tab, "Classification")
        
        # Confidence and Status
        status_group = QGroupBox("Extraction Status")
        status_layout = QVBoxLayout(status_group)
        
        self.confidence_label = QLabel()
        self.validation_label = QLabel()
        
        status_layout.addWidget(self.confidence_label)
        status_layout.addWidget(self.validation_label)
        
        layout.addWidget(tab_widget)
        layout.addWidget(status_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.save_button = QPushButton("Save & Import")
        self.save_button.clicked.connect(self.accept)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
    
    def _populate_data(self):
        """Populate the form with extracted data."""
        # Basic information
        self.title_edit.setText(self.extracted_data.get('title', ''))
        self.authors_edit.setText(self.extracted_data.get('authors', ''))
        self.abstract_edit.setPlainText(self.extracted_data.get('abstract', ''))
        self.year_spin.setValue(self.extracted_data.get('year', 2024))
        # Set published month if present
        pub_month = self.extracted_data.get('published_month', '')
        if pub_month and pub_month in [self.published_month_combo.itemText(i) for i in range(self.published_month_combo.count())]:
            self.published_month_combo.setCurrentText(pub_month)
        self.doi_edit.setText(self.extracted_data.get('doi', ''))
        self.journal_edit.setText(self.extracted_data.get('journal', ''))
        self.publisher_edit.setText(self.extracted_data.get('publisher', ''))
        
        # Classification
        self.indexing_status_combo.setCurrentText(self.extracted_data.get('indexing_status', 'Non-SCI/Non-Scopus'))
        self.department_combo.setCurrentText(self.enriched_data.get('department', ''))
        # Get research domain from extracted data (where it's actually assigned)
        research_domain = self.extracted_data.get('research_domain', '') or self.enriched_data.get('research_domain', '')
        self.research_domain_combo.setCurrentText(research_domain)
        
        keywords = self.extracted_data.get('keywords', [])
        if isinstance(keywords, list):
            self.keywords_edit.setPlainText(', '.join(keywords))
        
        # Status
        confidence = self.extracted_data.get('confidence', 0.0)
        self.confidence_label.setText(f"Extraction Confidence: {confidence:.1%}")
        
        validated = self.enriched_data.get('validated_doi', False)
        self.validation_label.setText(f"DOI Validated: {'Yes' if validated else 'No'}")
    
    def get_corrected_data(self) -> Dict[str, Any]:
        """Get the corrected data from the form."""
        return {
            'title': self.title_edit.text(),
            'authors': self.authors_edit.text(),
            'abstract': self.abstract_edit.toPlainText(),
            'year': self.year_spin.value(),
            'published_month': self.published_month_combo.currentText(),
            'doi': self.doi_edit.text(),
            'journal': self.journal_edit.text(),
            'publisher': self.publisher_edit.text(),
            'indexing_status': self.indexing_status_combo.currentText(),
            'department': self.department_combo.currentText(),
            'research_domain': self.research_domain_combo.currentText(),
            'keywords': [kw.strip() for kw in self.keywords_edit.toPlainText().split(',') if kw.strip()],
        }
    
    def _load_classification_options(self):
        """Load department and research domain options."""
        try:
            # Load departments from department manager
            departments = get_all_departments()
            self.department_combo.addItems(departments)
            logger.info(f"Loaded {len(departments)} departments for classification")
        except Exception as e:
            logger.error(f"Error loading departments for classification: {e}")
            # Fallback to basic departments
            self.department_combo.addItems([
                "Computer Science & Engineering",
                "Civil Engineering", 
                "Mechanical Engineering",
                "Electrical & Electronics Engineering",
                "Electronics & Communication Engineering"
            ])
        
        # Load research domains
        research_domains = [
            "Machine Learning", "Artificial Intelligence", "Data Science", "Quantum Computing",
            "Computer Vision", "Natural Language Processing", "Robotics", "Cybersecurity",
            "Software Engineering", "Database Systems", "Computer Networks", "Operating Systems",
            "Algorithms", "Data Structures", "Computer Graphics", "Human-Computer Interaction",
            "Information Systems", "Web Technologies", "Mobile Computing", "Cloud Computing",
            "Big Data", "Internet of Things", "Blockchain", "Digital Forensics",
            "Materials Science", "Renewable Energy", "Biomedical Engineering", "Environmental Engineering"
        ]
        self.research_domain_combo.addItems(research_domains)


class AdvancedSearchWidget(QWidget):
    """Advanced search widget with multiple filters."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        # Define month range combos early to avoid attribute access before creation
        from PySide6.QtWidgets import QComboBox
        self.month_from_combo = QComboBox()
        self.month_to_combo = QComboBox()
        month_items = [
            "All", "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]
        self.month_from_combo.addItems(month_items)
        self.month_to_combo.addItems(month_items)
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the advanced search UI."""
        layout = QVBoxLayout(self)
        
        # Search query
        query_group = QGroupBox("Search Query")
        query_layout = QVBoxLayout(query_group)
        
        # Search type selection
        search_type_layout = QHBoxLayout()
        search_type_layout.addWidget(QLabel("Search Type:"))
        self.search_type_combo = QComboBox()
        self.search_type_combo.addItems(["Hybrid", "Semantic", "Keyword"])
        self.search_type_combo.setToolTip("Hybrid: Combines semantic and keyword search\nSemantic: Meaning-based search\nKeyword: Traditional text search")
        search_type_layout.addWidget(self.search_type_combo)
        search_type_layout.addStretch()
        query_layout.addLayout(search_type_layout)
        
        self.query_edit = QLineEdit()
        self.query_edit.setPlaceholderText("Enter search terms... (e.g., 'machine learning in healthcare', 'deep learning computer vision')")
        query_layout.addWidget(self.query_edit)
        
        layout.addWidget(query_group)
        
        # Filters
        filters_group = QGroupBox("Filters")
        filters_layout = QFormLayout(filters_group)
        
        self.year_from_spin = QSpinBox()
        self.year_from_spin.setRange(1900, 2030)
        self.year_from_spin.setValue(2000)
        
        self.year_to_spin = QSpinBox()
        self.year_to_spin.setRange(1900, 2030)
        self.year_to_spin.setValue(2024)
        
        self.journal_combo = QComboBox()
        self.journal_combo.setEditable(True)
        self.journal_combo.setPlaceholderText("Select or enter journal...")
        
        self.indexing_status_combo = QComboBox()
        self.indexing_status_combo.addItems(["All", "SCI", "Scopus", "Non-Indexed"])
        
        self.department_combo = QComboBox()
        self.department_combo.setEditable(True)
        self.department_combo.setPlaceholderText("Select or enter department...")
        
        self.research_domain_combo = QComboBox()
        self.research_domain_combo.setEditable(True)
        self.research_domain_combo.setPlaceholderText("Select or enter research domain...")
        
        self.paper_type_combo = QComboBox()
        self.paper_type_combo.setEditable(True)
        self.paper_type_combo.setPlaceholderText("Select or enter paper type...")
        
        self.publisher_combo = QComboBox()
        self.publisher_combo.setEditable(True)
        self.publisher_combo.setPlaceholderText("Select or enter publisher...")
        
        self.student_status_combo = QComboBox()
        self.student_status_combo.addItems(["All", "Yes", "No", "Unknown"])
        
        self.review_status_combo = QComboBox()
        self.review_status_combo.setEditable(True)
        self.review_status_combo.addItems(["All", "Imported", "Under Review", "Accepted", "Rejected", "Published", "Draft"])
        
        filters_layout.addRow("Year From:", self.year_from_spin)
        filters_layout.addRow("Year To:", self.year_to_spin)
        filters_layout.addRow("Journal:", self.journal_combo)
        filters_layout.addRow("Publisher:", self.publisher_combo)
        filters_layout.addRow("Indexing Status:", self.indexing_status_combo)
        filters_layout.addRow("Department:", self.department_combo)
        filters_layout.addRow("Research Domain:", self.research_domain_combo)
        filters_layout.addRow("Month From:", self.month_from_combo)
        filters_layout.addRow("Month To:", self.month_to_combo)
        filters_layout.addRow("Paper Type:", self.paper_type_combo)
        filters_layout.addRow("Student Work:", self.student_status_combo)
        filters_layout.addRow("Review Status:", self.review_status_combo)
        
        layout.addWidget(filters_group)
        
        # Search buttons
        button_layout = QHBoxLayout()
        
        self.search_button = QPushButton("Search")
        self.clear_button = QPushButton("Clear Filters")
        
        button_layout.addWidget(self.search_button)
        button_layout.addWidget(self.clear_button)
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
    
    def get_search_filters(self) -> Dict[str, Any]:
        """Get current search filters."""
        filters = {}
        
        # Only add year filters if they're not at default values
        if self.year_from_spin.value() > 2000:  # Default is 2000
            filters['year_from'] = self.year_from_spin.value()
        if self.year_to_spin.value() < 2024:  # Default is 2024
            filters['year_to'] = self.year_to_spin.value()
        if self.journal_combo.currentText() and self.journal_combo.currentText().strip():
            filters['journal'] = self.journal_combo.currentText()
        if self.publisher_combo.currentText() and self.publisher_combo.currentText().strip():
            filters['publisher'] = self.publisher_combo.currentText()
        if self.indexing_status_combo.currentText() != "All":
            filters['indexing_status'] = self.indexing_status_combo.currentText()
        if self.department_combo.currentText() and self.department_combo.currentText().strip():
            filters['department'] = self.department_combo.currentText()
        if self.research_domain_combo.currentText() and self.research_domain_combo.currentText().strip():
            filters['research_domain'] = self.research_domain_combo.currentText()
        if self.paper_type_combo.currentText() and self.paper_type_combo.currentText().strip():
            filters['paper_type'] = self.paper_type_combo.currentText()
        if self.student_status_combo.currentText() != "All":
            filters['student'] = self.student_status_combo.currentText()
        if self.review_status_combo.currentText() != "All":
            filters['review_status'] = self.review_status_combo.currentText()
        # Month range filters
        if hasattr(self, 'month_from_combo') and self.month_from_combo.currentText() != "All":
            filters['month_from'] = self.month_from_combo.currentText()
        if hasattr(self, 'month_to_combo') and self.month_to_combo.currentText() != "All":
            filters['month_to'] = self.month_to_combo.currentText()
        
        return filters
    
    def clear_filters(self):
        """Clear all filters."""
        self.query_edit.clear()
        self.year_from_spin.setValue(2000)
        self.year_to_spin.setValue(2024)
        self.journal_combo.setCurrentText("")
        self.publisher_combo.setCurrentText("")
        self.indexing_status_combo.setCurrentText("All")
        self.department_combo.setCurrentText("")
        self.research_domain_combo.setCurrentText("")
        self.paper_type_combo.setCurrentText("")
        self.student_status_combo.setCurrentText("All")
        self.review_status_combo.setCurrentText("All")


class EnhancedMainWindow(QMainWindow):
    """Enhanced main window with automated PDF processing."""
    
    def __init__(self):
        QMainWindow.__init__(self)
        self.setWindowTitle(f"{APP_NAME} v2.0")
        self.resize(1400, 900)
        
        # Initialize database and repository
        self.paper_repo = get_unified_paper_repository()
        self.integration_manager = get_integration_manager()
        
        # Initialize search engines
        self.hybrid_search_engine = HybridSearchEngine(self.paper_repo)
        self.semantic_search_engine = SemanticSearchEngine(self.paper_repo)
        
        # Processing thread
        self.processing_thread = None
        
        # Flag to prevent auto-search during initialization
        self._initializing = True
        
        self._build_menu()
        self._build_body()
        self._load_initial_data()
        
        # Enable auto-search after initialization
        self._initializing = False
        
        # Development mode logging
        if self._is_development_mode():
            logger.info("Running in development mode")
    
    def _build_menu(self):
        """Build the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        file_menu.addAction("Import Single PDF", self._import_single_pdf)
        file_menu.addAction("Bulk Import PDFs", self._bulk_import_pdfs)
        file_menu.addSeparator()
        
        # Export submenu
        export_menu = file_menu.addMenu("Export")
        export_menu.addAction("Export Table to PDF", self._export_table_to_pdf)
        export_menu.addAction("Export Summary to PDF", self._export_summary_to_pdf)
        
        file_menu.addSeparator()
        
        # Delete submenu
        delete_menu = file_menu.addMenu("Delete Papers")
        delete_menu.addAction("Delete Selected Paper", self._delete_selected_paper)
        delete_menu.addAction("Delete Selected Papers", self._delete_selected_papers)
        delete_menu.addSeparator()
        delete_menu.addAction("Delete All Duplicates", self._delete_all_duplicates)
        delete_menu.addSeparator()
        delete_menu.addAction("Delete All Papers", self._delete_all_papers)
        
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)
        
        # Tools menu
        tools_menu = menubar.addMenu("Tools")
        tools_menu.addAction("Manage Duplicates", self._manage_duplicates)
        tools_menu.addSeparator()
        tools_menu.addAction("Verify Papers", self._verify_papers)
        tools_menu.addAction("Refresh Citations", self._refresh_citations)
        tools_menu.addAction("Update Indexing Status", self._update_indexing_status)
        tools_menu.addSeparator()
        
        # Semantic search submenu
        semantic_menu = tools_menu.addMenu("Semantic Search")
        semantic_menu.addAction("Generate All Embeddings", self._generate_all_embeddings)
        semantic_menu.addAction("Clear Embeddings Cache", self._clear_embeddings_cache)
        semantic_menu.addAction("Show Embedding Stats", self._show_embedding_stats)
        
        tools_menu.addSeparator()
        tools_menu.addAction("Database Statistics", self._show_database_stats)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        help_menu.addAction("About", self._show_about)
        help_menu.addAction("Keyboard Shortcuts", self._show_shortcuts)
    
    def _build_body(self):
        """Build the main UI."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Create splitter for search and results
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Search
        search_panel = QWidget()
        search_layout = QVBoxLayout(search_panel)
        
        self.search_widget = AdvancedSearchWidget()
        self.search_widget.search_button.clicked.connect(self._perform_search)
        self.search_widget.clear_button.clicked.connect(self._clear_search)
        
        # Connect filter changes to auto-search
        self.search_widget.query_edit.textChanged.connect(self._perform_search)
        self.search_widget.year_from_spin.valueChanged.connect(self._perform_search)
        self.search_widget.year_to_spin.valueChanged.connect(self._perform_search)
        self.search_widget.journal_combo.currentTextChanged.connect(self._perform_search)
        self.search_widget.indexing_status_combo.currentTextChanged.connect(self._perform_search)
        self.search_widget.department_combo.currentTextChanged.connect(self._perform_search)
        self.search_widget.research_domain_combo.currentTextChanged.connect(self._perform_search)
        
        search_layout.addWidget(self.search_widget)
        
        # Right panel - Results
        results_panel = QWidget()
        results_layout = QVBoxLayout(results_panel)
        
        # Results table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(11)
        self.results_table.setHorizontalHeaderLabels([
            "Title", "Authors", "Year", "Month", "Journal", "Indexing", "Citations", "Quartile", "Department", "Domain", "Verification"
        ])
        
        # Set column widths for better display
        header = self.results_table.horizontalHeader()
        header.setStretchLastSection(True)
        # Smarter fixed/stretched widths: make Title and Journal wider, others tighter
        header.setSectionResizeMode(0, QHeaderView.Stretch)          # Title
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # Authors
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # Year
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # Month
        header.setSectionResizeMode(4, QHeaderView.Stretch)           # Journal
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)  # Indexing
        header.setSectionResizeMode(6, QHeaderView.ResizeToContents)  # Citations
        header.setSectionResizeMode(7, QHeaderView.ResizeToContents)  # Quartile
        header.setSectionResizeMode(8, QHeaderView.ResizeToContents)  # Department
        header.setSectionResizeMode(9, QHeaderView.ResizeToContents)  # Domain
        header.setSectionResizeMode(10, QHeaderView.ResizeToContents) # Verification

        # Enable word wrap and a delegate that wraps and adjusts row heights
        self.results_table.setWordWrap(True)
        self.results_table.setItemDelegate(WrappingItemDelegate(self.results_table))
        self.results_table.verticalHeader().setDefaultSectionSize(24)
        self.results_table.verticalHeader().setMinimumSectionSize(24)
        
        # Set alternating row colors
        self.results_table.setAlternatingRowColors(True)
        
        # Enable context menu for similar papers
        self.results_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.results_table.customContextMenuRequested.connect(self._show_context_menu)
        self.results_table.cellDoubleClicked.connect(self._open_paper_details)
        
        results_layout.addWidget(self.results_table)
        
        # Results actions
        results_actions = QHBoxLayout()
        
        self.open_pdf_button = QPushButton("Open PDF")
        self.open_pdf_button.clicked.connect(self._open_selected_pdf)
        
        self.edit_metadata_button = QPushButton("Edit Metadata")
        self.edit_metadata_button.clicked.connect(self._edit_selected_metadata)
        
        self.delete_paper_button = QPushButton("Delete Paper")
        self.delete_paper_button.clicked.connect(self._delete_selected_paper)
        
        results_actions.addWidget(self.open_pdf_button)
        results_actions.addWidget(self.edit_metadata_button)
        results_actions.addWidget(self.delete_paper_button)
        results_actions.addStretch()
        
        results_layout.addLayout(results_actions)
        
        # Add panels to splitter
        splitter.addWidget(search_panel)
        splitter.addWidget(results_panel)
        splitter.setSizes([400, 1000])
        
        main_layout.addWidget(splitter)
    
    def _load_initial_data(self):
        """Load initial data and populate filters."""
        # Load distinct values for filter dropdowns
        self._populate_filter_dropdowns()
        
        # Perform initial search
        self._perform_search()
    
    def _populate_filter_dropdowns(self):
        """Populate filter dropdowns with distinct values."""
        # Load journals
        self.search_widget.journal_combo.addItems([
            "Nature", "Science", "IEEE Transactions", "ACM Computing", "Physical Review",
            "Journal of Machine Learning Research", "Neural Information Processing Systems",
            "International Conference on Machine Learning", "Computer Vision and Pattern Recognition"
        ])
        
        # Load publishers
        self.search_widget.publisher_combo.addItems([
            "Nature Publishing Group", "Springer Nature", "Wiley", "Elsevier", "IEEE",
            "ACM", "Taylor & Francis", "SAGE", "Oxford University Press", "Cambridge University Press"
        ])
        
        # Load departments from department manager
        try:
            departments = get_all_departments()
            self.search_widget.department_combo.addItems(departments)
            logger.info(f"Loaded {len(departments)} departments from department manager")
        except Exception as e:
            logger.error(f"Error loading departments: {e}")
            # Fallback to basic departments
            self.search_widget.department_combo.addItems([
                "Computer Science & Engineering", "Civil Engineering", "Mechanical Engineering",
                "Electrical & Electronics Engineering", "Chemical Engineering", "Aerospace Engineering",
                "Biomedical Engineering", "Environmental Engineering"
            ])
        
        # Load research domains
        research_domains = [
            "Machine Learning", "Artificial Intelligence", "Data Science", "Quantum Computing",
            "Computer Vision", "Natural Language Processing", "Robotics", "Cybersecurity",
            "Software Engineering", "Database Systems", "Computer Networks", "Operating Systems",
            "Algorithms", "Data Structures", "Computer Graphics", "Human-Computer Interaction",
            "Information Systems", "Web Technologies", "Mobile Computing", "Cloud Computing",
            "Big Data", "Internet of Things", "Blockchain", "Digital Forensics",
            "Materials Science", "Renewable Energy", "Biomedical Engineering", "Environmental Engineering",
            "Structural Engineering", "Transportation Engineering", "Geotechnical Engineering",
            "Water Resources Engineering", "Thermodynamics", "Fluid Mechanics", "Heat Transfer",
            "Control Systems", "Power Systems", "Signal Processing", "Communication Systems"
        ]
        self.search_widget.research_domain_combo.addItems(research_domains)
        
        # Load paper types
        paper_types = [
            "Journal Article", "Conference Paper", "Book Chapter", "Thesis/Dissertation",
            "Technical Report", "Preprint", "Review Article", "Case Study", "Short Paper",
            "Poster", "Workshop Paper", "White Paper", "Research Paper", "Other"
        ]
        self.search_widget.paper_type_combo.addItems(paper_types)
    
    def _import_single_pdf(self):
        """Import a single PDF file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select PDF File", "", "PDF Files (*.pdf)"
        )
        
        if file_path:
            self._process_pdf_files([file_path])
    
    def _bulk_import_pdfs(self):
        """Import multiple PDF files."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select PDF Files", "", "PDF Files (*.pdf)"
        )
        
        if file_paths:
            self._process_pdf_files(file_paths)
    
    def _process_pdf_files(self, file_paths: List[str]):
        """Process PDF files in background thread."""
        if self.processing_thread and self.processing_thread.isRunning():
            QMessageBox.warning(self, "Processing", "Another import is already in progress.")
            return
        
        # Create progress dialog
        self.progress_dialog = QProgressDialog("Processing PDF files...", "Cancel", 0, len(file_paths), self)
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.show()
        
        # Create and start processing thread
        self.processing_thread = PDFProcessingThread(file_paths)
        self.processing_thread.progress_updated.connect(self._update_progress)
        self.processing_thread.processing_completed.connect(self._handle_processing_completed)
        self.processing_thread.error_occurred.connect(self._handle_processing_error)
        self.processing_thread.start()
    
    def _update_progress(self, status: str, current: int, total: int):
        """Update progress dialog."""
        self.progress_dialog.setLabelText(status)
        self.progress_dialog.setValue(current)
        self.progress_dialog.setMaximum(total)
    
    def _handle_processing_completed(self, results: Dict):
        """Handle completed PDF processing."""
        self.progress_dialog.close()
        
        successful_imports = 0
        failed_imports = 0
        
        for result in results['results']:
            if result['success']:
                # Show correction dialog
                dialog = MetadataCorrectionDialog(
                    result['extracted'].__dict__,
                    result['enriched'].__dict__,
                    self
                )
                
                if dialog.exec() == QDialog.Accepted:
                    corrected_data = dialog.get_corrected_data()
                    self._import_paper_data(corrected_data, result['file_path'])
                    successful_imports += 1
                else:
                    failed_imports += 1
            else:
                failed_imports += 1
        
        # Show results
        QMessageBox.information(
            self, "Import Complete",
            f"Successfully imported: {successful_imports}\nFailed: {failed_imports}"
        )
        
        # Refresh search results
        self._perform_search()
    
    def _handle_processing_error(self, error: str):
        """Handle processing error."""
        self.progress_dialog.close()
        QMessageBox.critical(self, "Processing Error", f"Error processing PDFs: {error}")
    
    def _import_paper_data(self, data: Dict[str, Any], file_path: str):
        """Import paper data to database."""
        try:
            
            paper_data = {
                'title': data['title'],
                'authors': data['authors'],
                'year': data['year'],
                'abstract': data['abstract'],
                'doi': data.get('doi', ''),
                'journal': data.get('journal', ''),
                'publisher': data.get('publisher', ''),
                'file_path': file_path,
                'full_text': '',  # Would be extracted separately
                'department': data.get('department', ''),
                'research_domain': data.get('research_domain', ''),
                'paper_type': data.get('paper_type', 'Research Paper'),
                'student': data.get('student', 'No'),
                'review_status': data.get('review_status', 'Imported'),
                'indexing_status': data.get('indexing_status', ''),
                'issn': data.get('issn', ''),
                'published_month': data.get('published_month', '')
            }
            
            new_paper_id = self.paper_repo.add_paper(paper_data)
            logger.info(f"Imported paper {new_paper_id}: {data['title']}")
            
        except Exception as e:
            logger.error(f"Error importing paper: {e}")
            QMessageBox.critical(self, "Import Error", f"Failed to import paper: {e}")
    
    def _perform_search(self):
        """Perform search with current filters."""
        # Skip auto-search during initialization
        if hasattr(self, '_initializing') and self._initializing:
            return
            
        query = self.search_widget.query_edit.text()
        filters = self.search_widget.get_search_filters()
        search_type = self.search_widget.search_type_combo.currentText().lower()
        
        try:
            # Use appropriate search engine based on search type
            if query:
                if search_type == "semantic":
                    # Use semantic search engine
                    results = self.semantic_search_engine.search(
                        query=query,
                        top_k=50,
                        threshold=0.3
                    )
                    papers = [p for p, _ in results]
                elif search_type == "keyword":
                    # Use traditional TF-IDF search
                    from ..search_engine import TfidfSearchEngine
                    search_engine = TfidfSearchEngine(self.paper_repo)
                    results = search_engine.search(query)
                    papers = [p for p, _ in results]
                else:  # hybrid
                    # Use hybrid search engine
                    results = self.hybrid_search_engine.search(
                        query=query,
                        search_type=search_type,
                        top_k=50,  # Get more results for filtering
                        semantic_threshold=0.3
                    )
                    papers = [p for p, _ in results]
            else:
                # Get all papers if no query
                papers = self.paper_repo.list_all()
            
            # Convert to dict format for compatibility
            paper_dicts = []
            for paper in papers:
                paper_dict = {
                    'id': paper.get('id'),
                    'title': paper.get('title'),
                    'authors': paper.get('authors'),
                    'year': paper.get('year'),
                    'published_month': paper.get('published_month', ''),
                    'abstract': paper.get('abstract'),
                    'journal': paper.get('journal'),
                    'publisher': paper.get('publisher'),
                    'file_path': paper.get('file_path'),
                    'indexing_status': paper.get('indexing_status'),
                    # Include verification fields from repository/search results
                    'verification_status': paper.get('verification_status'),
                    'verification_method': paper.get('verification_method'),
                    'verification_confidence': paper.get('verification_confidence', 0.0),
                    'verification_date': paper.get('verification_date'),
                    'citation_count': paper.get('citation_count', 0),
                    'scimago_quartile': paper.get('scimago_quartile', ''),
                    'impact_factor': paper.get('impact_factor', 0.0),
                    'h_index': paper.get('h_index', 0),
                    'citation_source': paper.get('citation_source', ''),
                    'citation_updated_at': paper.get('citation_updated_at', ''),
                    'metadata': {
                        'department': paper.get('department'),
                        'research_domain': paper.get('research_domain'),
                        'indexing_status': paper.get('indexing_status')
                    }
                }
                paper_dicts.append(paper_dict)
            
            # Apply filters to the results
            filtered_papers = self._apply_filters(paper_dicts, filters)
            
            self._populate_results_table(filtered_papers)
        except Exception as e:
            logger.error(f"Search error: {e}")
            QMessageBox.critical(self, "Search Error", f"Search failed: {e}")
    
    def _apply_filters(self, papers: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply filters to the paper list."""
        logger.info(f"Applying filters: {filters}")
        logger.info(f"Total papers before filtering: {len(papers)}")
        
        filtered_papers = papers.copy()
        
        # Year filter
        if 'year_from' in filters:
            filtered_papers = [p for p in filtered_papers if p.get('year', 0) >= filters['year_from']]
            logger.info(f"After year_from filter: {len(filtered_papers)} papers")
        
        if 'year_to' in filters:
            filtered_papers = [p for p in filtered_papers if p.get('year', 0) <= filters['year_to']]
            logger.info(f"After year_to filter: {len(filtered_papers)} papers")
        # Month range filter within the selected year range (if provided)
        month_name_to_num = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
            'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12
        }
        month_from = filters.get('month_from')
        month_to = filters.get('month_to')
        if month_from or month_to:
            start_num = month_name_to_num.get((month_from or '').strip().lower(), 1)
            end_num = month_name_to_num.get((month_to or '').strip().lower(), 12)
            # Normalize if reversed
            if start_num > end_num:
                start_num, end_num = end_num, start_num
            def in_month_range(p):
                m = (p.get('published_month') or '').strip().lower()
                if not m or m not in month_name_to_num:
                    return False
                mn = month_name_to_num[m]
                return start_num <= mn <= end_num
            before = len(filtered_papers)
            filtered_papers = [p for p in filtered_papers if in_month_range(p)]
            logger.info(f"After month range filter ({month_from or 'Jan'}-{month_to or 'Dec'}): {len(filtered_papers)} of {before}")
        
        # Journal filter
        if 'journal' in filters and filters['journal']:
            journal_filter = filters['journal'].lower()
            filtered_papers = [p for p in filtered_papers 
                             if journal_filter in p.get('journal', '').lower()]
            logger.info(f"After journal filter: {len(filtered_papers)} papers")
        
        # Indexing status filter
        if 'indexing_status' in filters and filters['indexing_status'] != 'All':
            indexing_filter = filters['indexing_status']
            filtered_papers = [p for p in filtered_papers 
                             if p.get('metadata', {}).get('indexing_status', '') == indexing_filter]
            logger.info(f"After indexing status filter: {len(filtered_papers)} papers")
        
        # Department filter
        if 'department' in filters and filters['department']:
            dept_filter = filters['department'].lower()
            filtered_papers = [p for p in filtered_papers 
                             if dept_filter in p.get('metadata', {}).get('department', '').lower()]
            logger.info(f"After department filter: {len(filtered_papers)} papers")
        
        # Research domain filter
        if 'research_domain' in filters and filters['research_domain']:
            domain_filter = filters['research_domain'].lower()
            filtered_papers = [p for p in filtered_papers 
                             if domain_filter in p.get('metadata', {}).get('research_domain', '').lower()]
            logger.info(f"After research domain filter: {len(filtered_papers)} papers")
        
        logger.info(f"Total papers after all filters: {len(filtered_papers)}")
        return filtered_papers

    def _clear_search(self):
        """Clear search and show all papers."""
        self.search_widget.clear_filters()
        self._perform_search()
    
    def _show_context_menu(self, position):
        """Show context menu for table rows."""
        item = self.results_table.itemAt(position)
        if item is None:
            return
        
        row = item.row()
        paper_id = self.results_table.item(row, 0).data(Qt.UserRole)
        
        if paper_id is None:
            return
        
        # Create context menu
        context_menu = QMenu(self)
        
        # Find similar papers action
        similar_action = context_menu.addAction("Find Similar Papers")
        similar_action.triggered.connect(lambda: self._find_similar_papers(paper_id))
        
        # Open paper details action
        details_action = context_menu.addAction("View Details")
        details_action.triggered.connect(lambda: self._open_paper_details(row, 0))
        
        # Edit metadata action
        edit_action = context_menu.addAction("Edit Metadata")
        edit_action.triggered.connect(lambda: self._edit_selected_metadata())
        
        # Open PDF action
        open_pdf_action = context_menu.addAction("Open PDF")
        open_pdf_action.triggered.connect(lambda: self._open_selected_pdf())
        
        # Show context menu
        context_menu.exec(self.results_table.mapToGlobal(position))
    
    def _find_similar_papers(self, paper_id: int):
        """Find papers similar to the selected paper."""
        try:
            # Find similar papers using semantic search
            similar_papers = self.hybrid_search_engine.find_similar_papers(
                paper_id, top_k=10, threshold=0.3
            )
            
            if not similar_papers:
                QMessageBox.information(
                    self, "Similar Papers",
                    "No similar papers found for the selected paper."
                )
                return
            
            # Show similar papers in a dialog
            self._show_similar_papers_dialog(similar_papers)
            
        except Exception as e:
            logger.error(f"Error finding similar papers: {e}")
            QMessageBox.critical(self, "Error", f"Failed to find similar papers: {e}")
    
    def _show_similar_papers_dialog(self, similar_papers: List[Tuple[Any, float]]):
        """Show similar papers in a dialog."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Similar Papers")
        dialog.setModal(True)
        dialog.resize(800, 600)
        
        layout = QVBoxLayout(dialog)
        
        # Title
        title_label = QLabel(f"Found {len(similar_papers)} similar papers:")
        title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title_label)
        
        # Similar papers table
        table = QTableWidget()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["Title", "Authors", "Year", "Similarity"])
        
        table.setRowCount(len(similar_papers))
        for row, (paper, similarity) in enumerate(similar_papers):
            table.setItem(row, 0, QTableWidgetItem(paper.title))
            table.setItem(row, 1, QTableWidgetItem(paper.authors))
            table.setItem(row, 2, QTableWidgetItem(str(paper.year)))
            table.setItem(row, 3, QTableWidgetItem(f"{similarity:.3f}"))
        
        # Resize columns
        table.resizeColumnsToContents()
        layout.addWidget(table)
        
        # Buttons
        button_layout = QHBoxLayout()
        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        button_layout.addStretch()
        button_layout.addWidget(close_button)
        layout.addLayout(button_layout)
        
        dialog.exec()
    
    def _open_paper_details(self, row: int, column: int):
        """Open paper details dialog."""
        try:
            paper_id = self.results_table.item(row, 0).data(Qt.UserRole)
            if paper_id is None:
                return
            
            paper = self.paper_repo.get_by_id(paper_id)
            if paper is None:
                return
            
            # Create details dialog
            dialog = QDialog(self)
            dialog.setWindowTitle(f"Paper Details - {paper.title[:50]}...")
            dialog.setModal(True)
            dialog.resize(600, 500)
            
            layout = QVBoxLayout(dialog)
            
            # Paper details
            details_text = QTextEdit()
            details_text.setReadOnly(True)
            
            details = f"""
Title: {paper.title}
Authors: {paper.authors}
Year: {paper.year}
Abstract: {paper.abstract}
Department: {paper.department}
Research Domain: {paper.research_domain}
Publisher: {paper.publisher}
File Path: {paper.file_path}
            """
            
            details_text.setPlainText(details.strip())
            layout.addWidget(details_text)
            
            # Buttons
            button_layout = QHBoxLayout()
            close_button = QPushButton("Close")
            close_button.clicked.connect(dialog.accept)
            button_layout.addStretch()
            button_layout.addWidget(close_button)
            layout.addLayout(button_layout)
            
            dialog.exec()
            
        except Exception as e:
            logger.error(f"Error opening paper details: {e}")
            QMessageBox.critical(self, "Error", f"Failed to open paper details: {e}")
    
    def _populate_results_table(self, papers: List[Dict[str, Any]]):
        """Populate results table with papers."""
        self.results_table.setRowCount(len(papers))
        
        for row, paper in enumerate(papers):
            metadata = paper.get('metadata', {})
            
            # Store paper ID in the first column for later retrieval
            is_duplicate = paper.get('is_duplicate', False)
            title = paper.get('title', '')
            title_display = f"🔗 {title}" if is_duplicate else title  # Add duplicate indicator
            title_item = QTableWidgetItem(title_display)
            title_item.setData(Qt.UserRole, paper.get('id'))
            title_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            
            # Highlight duplicates with background color
            if is_duplicate:
                title_item.setBackground(QColor(255, 235, 235))  # Light red background
                dup_id = paper.get('duplicate_of_id')
                similarity = paper.get('similarity_score', 0)
                title_item.setToolTip(f"Duplicate of paper ID: {dup_id}\nSimilarity: {similarity*100:.1f}%")
            
            self.results_table.setItem(row, 0, title_item)
            
            self.results_table.setItem(row, 1, QTableWidgetItem(paper.get('authors', '')))
            self.results_table.setItem(row, 2, QTableWidgetItem(str(paper.get('year', ''))))
            # Published month
            published_month = paper.get('published_month', '')
            self.results_table.setItem(row, 3, QTableWidgetItem(published_month if published_month else 'N/A'))
            journal_item = QTableWidgetItem(paper.get('journal', ''))
            journal_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            self.results_table.setItem(row, 4, journal_item)
            # Get indexing_status from paper object or fallback to metadata
            indexing_status = paper.get('indexing_status', metadata.get('indexing_status', 'Unknown'))
            indexing_item = QTableWidgetItem(indexing_status)
            
            # Color code indexing status
            if indexing_status == "SCI":
                indexing_item.setBackground(QColor(0, 255, 0, 50))  # Light green
            elif indexing_status == "Scopus":
                indexing_item.setBackground(QColor(0, 0, 255, 50))  # Light blue
            elif indexing_status == "SCI + Scopus":
                indexing_item.setBackground(QColor(255, 165, 0, 50))  # Light orange
            elif indexing_status == "Unknown":
                indexing_item.setBackground(QColor(255, 255, 0, 50))  # Light yellow
            
            self.results_table.setItem(row, 5, indexing_item)
            
            # Citation count
            citation_count = paper.get('citation_count', 0)
            citation_item = QTableWidgetItem(str(citation_count))
            if citation_count > 0:
                citation_item.setToolTip(f"Source: {paper.get('citation_source', 'Unknown')}\nUpdated: {paper.get('citation_updated_at', 'Unknown')}")
            self.results_table.setItem(row, 6, citation_item)
            
            # SCImago Quartile
            quartile = paper.get('scimago_quartile', '')
            quartile_item = QTableWidgetItem(quartile if quartile else 'N/A')
            if quartile and (paper.get('impact_factor') or 0) > 0:
                quartile_item.setToolTip(f"Impact Factor: {paper.get('impact_factor', 0):.2f}")
            self.results_table.setItem(row, 7, quartile_item)
            
            department_value = paper.get('department') or metadata.get('department', '')
            domain_value = paper.get('research_domain') or metadata.get('research_domain', '')
            self.results_table.setItem(row, 8, QTableWidgetItem(department_value))
            self.results_table.setItem(row, 9, QTableWidgetItem(domain_value))
            
            # Verification status
            verification_status = paper.get('verification_status') or 'pending'
            verification_item = QTableWidgetItem(verification_status.title())
            verification_item.setTextAlignment(Qt.AlignCenter)
            
            # Color code verification status
            if verification_status == 'verified':
                verification_item.setBackground(QColor(200, 255, 200))  # Light green
            elif verification_status == 'partial':
                verification_item.setBackground(QColor(255, 255, 200))  # Light yellow
            elif verification_status == 'failed':
                verification_item.setBackground(QColor(255, 200, 200))  # Light red
            else:
                verification_item.setBackground(QColor(240, 240, 240))  # Light gray
            
            self.results_table.setItem(row, 10, verification_item)
    
    def _open_paper_details(self, row: int, column: int):
        """Open paper details when double-clicking."""
        paper_id = self.results_table.item(row, 0).data(Qt.UserRole)
        if paper_id:
            self._view_paper_details(paper_id)
    
    def _view_paper_details(self, paper_id: int):
        """View detailed paper information."""
        paper = self.paper_repo.get_paper_by_id(paper_id)
        if not paper:
            QMessageBox.warning(self, "Error", "Paper not found.")
            return
        
        # Create details dialog
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Paper Details - {paper.get('title', 'Unknown')}")
        dialog.resize(600, 500)
        
        layout = QVBoxLayout(dialog)
        
        # Create scrollable content
        scroll = QScrollArea()
        content = QWidget()
        content_layout = QFormLayout(content)
        
        # Add paper details
        content_layout.addRow("Title:", QLabel(paper.get('title', '')))
        content_layout.addRow("Authors:", QLabel(paper.get('authors', '')))
        content_layout.addRow("Year:", QLabel(str(paper.get('year', ''))))
        content_layout.addRow("Publisher:", QLabel(paper.get('publisher', '')))
        content_layout.addRow("Department:", QLabel(paper.get('department', '')))
        content_layout.addRow("Research Domain:", QLabel(paper.get('research_domain', '')))
        content_layout.addRow("Paper Type:", QLabel(paper.get('paper_type', '')))
        content_layout.addRow("Student:", QLabel(paper.get('student', '')))
        content_layout.addRow("Review Status:", QLabel(paper.get('review_status', '')))
        
        if paper.get('abstract'):
            abstract_label = QLabel(paper.get('abstract', ''))
            abstract_label.setWordWrap(True)
            content_layout.addRow("Abstract:", abstract_label)
        
        scroll.setWidget(content)
        layout.addWidget(scroll)
        
        # Buttons
        button_layout = QHBoxLayout()
        open_pdf_btn = QPushButton("Open PDF")
        open_pdf_btn.clicked.connect(lambda: self._open_pdf_file(paper.get('file_path', '')))
        button_layout.addWidget(open_pdf_btn)
        button_layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        dialog.exec()
    
    def _open_selected_pdf(self):
        """Open selected paper PDF."""
        current_row = self.results_table.currentRow()
        if current_row >= 0:
            paper_id = self.results_table.item(current_row, 0).data(Qt.UserRole)
            if paper_id:
                paper = self.paper_repo.get_paper_by_id(paper_id)
                if paper:
                    self._open_pdf_file(paper.get('file_path', ''))
    
    def _open_pdf_file(self, file_path: str):
        """Open PDF file with system default application."""
        try:
            open_pdf(file_path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not open PDF: {e}")
    
    def _edit_selected_metadata(self):
        """Edit metadata of selected paper."""
        current_row = self.results_table.currentRow()
        if current_row >= 0:
            paper_id = self.results_table.item(current_row, 0).data(Qt.UserRole)
            if paper_id:
                paper = self.paper_repo.get_paper_by_id(paper_id)
                if paper:
                    # Convert paper to dict format for edit dialog
                    paper_data = {
                        'id': paper.get('id'),
                        'title': paper.get('title'),
                        'authors': paper.get('authors'),
                        'year': paper.get('year'),
                        'published_month': paper.get('published_month', ''),
                        'abstract': paper.get('abstract'),
                        'doi': paper.get('doi'),
                        'journal': paper.get('journal'),
                        'publisher': paper.get('publisher'),
                        'issn': paper.get('issn'),
                        'file_path': paper.get('file_path'),
                        'indexing_status': paper.get('indexing_status'),
                        'citation_count': paper.get('citation_count', 0),
                        'scimago_quartile': paper.get('scimago_quartile', ''),
                        'impact_factor': paper.get('impact_factor', 0.0),
                        'h_index': paper.get('h_index', 0),
                        'citation_source': paper.get('citation_source', ''),
                        'citation_updated_at': paper.get('citation_updated_at', ''),
                        'metadata': {
                            'department': paper.get('department'),
                            'research_domain': paper.get('research_domain'),
                            'paper_type': paper.get('paper_type'),
                            'student': paper.get('student'),
                            'review_status': paper.get('review_status'),
                            'indexing_status': paper.get('indexing_status')
                        }
                    }
                    
                    # Create and show edit dialog
                    from .paper_edit_dialog import PaperEditDialog
                    dialog = PaperEditDialog(paper_data, {}, self)
                    
                    if dialog.exec() == QDialog.Accepted:
                        # Get edited data
                        edited_data = dialog.get_edited_data()
                        
                        # Update paper in database
                        try:
                            # Update basic fields
                            updates = {
                                'title': edited_data['title'],
                                'authors': edited_data['authors'],
                                'year': edited_data['year'],
                                'published_month': edited_data['published_month'],
                                'abstract': edited_data['abstract'],
                                'doi': edited_data['doi'],
                                'journal': edited_data['journal'],
                                'publisher': edited_data['publisher'],
                                'issn': edited_data['issn']
                            }
                            
                            # Update metadata fields
                            metadata_updates = {
                                'department': edited_data['metadata']['department'],
                                'research_domain': edited_data['metadata']['research_domain'],
                                'paper_type': edited_data['metadata']['paper_type'],
                                'student': edited_data['metadata']['student'],
                                'review_status': edited_data['metadata']['review_status']
                            }
                            
                            # Apply updates
                            success = self.paper_repo.update_paper_metadata(paper_id, updates)
                            
                            if success:
                                # Update metadata
                                if DB_BACKEND == "postgres":
                                    # For PostgreSQL, update metadata JSONB
                                    import json
                                    current_metadata = paper.metadata if hasattr(paper, 'metadata') else {}
                                    current_metadata.update(metadata_updates)
                                    self.paper_repo.update_paper_metadata(paper_id, {'metadata': json.dumps(current_metadata)})
                                else:
                                    # Update individual columns for PostgreSQL
                                    self.paper_repo.update_paper_metadata(paper_id, metadata_updates)
                                
                                QMessageBox.information(self, "Success", "Paper metadata updated successfully!")
                                
                                # Refresh the search results
                                self._perform_search()
                            else:
                                QMessageBox.warning(self, "Error", "Failed to update paper metadata.")
                                
                        except Exception as e:
                            logger.error(f"Error updating paper metadata: {e}")
                            QMessageBox.critical(self, "Error", f"Failed to update paper metadata: {str(e)}")
                else:
                    QMessageBox.warning(self, "Edit Metadata", "Paper not found.")
            else:
                QMessageBox.warning(self, "Edit Metadata", "No paper selected.")
        else:
            QMessageBox.warning(self, "Edit Metadata", "Please select a paper to edit.")
    
    def _delete_selected_paper(self):
        """Delete selected paper."""
        current_row = self.results_table.currentRow()
        if current_row >= 0:
            paper_id = self.results_table.item(current_row, 0).data(Qt.UserRole)
            if paper_id:
                self._delete_paper_by_id(paper_id)
    
    def _delete_paper_by_id(self, paper_id: int):
        """Delete a paper by its ID."""
        try:
            # Get paper information for confirmation
            paper = self.paper_repo.get_paper_by_id(paper_id)
            if not paper:
                QMessageBox.warning(self, "Delete Paper", "Paper not found.")
                return
            
            paper_info = {
                'title': paper.get('title', 'Unknown'),
                'authors': paper.get('authors', 'Unknown'),
                'year': paper.get('year', 'Unknown'),
                'department': paper.get('department', 'Unknown'),
                'research_domain': paper.get('research_domain', 'Unknown'),
                'file_path': paper.get('file_path', ''),
                'file_exists': Path(paper.get('file_path', '')).exists() if paper.get('file_path') else False,
                'file_size_mb': round(Path(paper.get('file_path', '')).stat().st_size / (1024*1024), 2) if paper.get('file_path') and Path(paper.get('file_path', '')).exists() else 0
            }
            
            # Create confirmation dialog
            from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QPushButton
            
            dialog = QDialog(self)
            dialog.setWindowTitle("Delete Paper")
            dialog.setModal(True)
            dialog.resize(500, 400)
            
            layout = QVBoxLayout(dialog)
            
            # Paper information
            info_layout = QVBoxLayout()
            info_layout.addWidget(QLabel(f"<b>Title:</b> {paper_info['title']}"))
            info_layout.addWidget(QLabel(f"<b>Authors:</b> {paper_info['authors']}"))
            info_layout.addWidget(QLabel(f"<b>Year:</b> {paper_info['year']}"))
            info_layout.addWidget(QLabel(f"<b>Department:</b> {paper_info['department']}"))
            info_layout.addWidget(QLabel(f"<b>Research Domain:</b> {paper_info['research_domain']}"))
            
            if paper_info['file_path']:
                info_layout.addWidget(QLabel(f"<b>PDF File:</b> {paper_info['file_path']}"))
                if paper_info['file_exists']:
                    info_layout.addWidget(QLabel(f"<b>File Size:</b> {paper_info['file_size_mb']} MB"))
                else:
                    info_layout.addWidget(QLabel("<b>File Status:</b> <span style='color: red;'>File not found</span>"))
            
            layout.addLayout(info_layout)
            
            # Delete options
            options_layout = QVBoxLayout()
            delete_file_checkbox = QCheckBox("Also delete PDF file from disk")
            delete_file_checkbox.setChecked(False)
            options_layout.addWidget(delete_file_checkbox)
            
            layout.addLayout(options_layout)
            
            # Warning message
            warning_label = QLabel("<b style='color: red;'>Warning:</b> This action cannot be undone!")
            layout.addWidget(warning_label)
            
            # Buttons
            button_layout = QHBoxLayout()
            
            delete_btn = QPushButton("Delete Paper")
            delete_btn.setStyleSheet("QPushButton { background-color: #dc3545; color: white; }")
            delete_btn.clicked.connect(lambda: self._confirm_delete_paper(dialog, paper_id, delete_file_checkbox.isChecked()))
            
            cancel_btn = QPushButton("Cancel")
            cancel_btn.clicked.connect(dialog.reject)
            
            button_layout.addWidget(delete_btn)
            button_layout.addWidget(cancel_btn)
            
            layout.addLayout(button_layout)
            
            dialog.exec()
            
        except Exception as e:
            QMessageBox.critical(self, "Delete Error", f"Error preparing deletion: {e}")
    
    def _confirm_delete_paper(self, dialog, paper_id: int, delete_file: bool):
        """Confirm and execute paper deletion."""
        try:
            # Delete the paper from database
            success = self.paper_repo.delete_paper(paper_id)
            
            if success:
                # Delete file if requested
                if delete_file:
                    paper = self.paper_repo.get_paper_by_id(paper_id)
                    if paper and paper.file_path and Path(paper.file_path).exists():
                        Path(paper.file_path).unlink()
                
                dialog.accept()
                QMessageBox.information(self, "Delete Paper", "Paper deleted successfully.")
                # Refresh search results
                self._perform_search()
            else:
                QMessageBox.critical(self, "Delete Error", "Failed to delete paper.")
                
        except Exception as e:
            QMessageBox.critical(self, "Delete Error", f"Error deleting paper: {e}")
    
    def _delete_selected_papers(self):
        """Delete multiple selected papers."""
        selected_rows = set()
        for item in self.results_table.selectedItems():
            selected_rows.add(item.row())
        
        if not selected_rows:
            QMessageBox.information(self, "Delete Papers", "Please select papers to delete.")
            return
        
        # Get paper IDs
        paper_ids = []
        for row in selected_rows:
            paper_id = self.results_table.item(row, 0).data(Qt.UserRole)
            if paper_id:
                paper_ids.append(paper_id)
        
        if not paper_ids:
            QMessageBox.warning(self, "Delete Papers", "No valid papers selected.")
            return
        
        # Confirmation dialog
        reply = QMessageBox.question(
            self, "Delete Papers",
            f"Are you sure you want to delete {len(paper_ids)} selected papers?\n\nThis action cannot be undone!",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                # Delete papers one by one
                success_count = 0
                failed_count = 0
                errors = []
                
                for paper_id in paper_ids:
                    try:
                        if self.paper_repo.delete_paper(paper_id):
                            success_count += 1
                        else:
                            failed_count += 1
                            errors.append(f"Failed to delete paper {paper_id}")
                    except Exception as e:
                        failed_count += 1
                        errors.append(f"Error deleting paper {paper_id}: {str(e)}")
                
                # Show results
                message = f"Deletion completed:\n"
                message += f"Successfully deleted: {success_count}\n"
                message += f"Failed: {failed_count}"
                
                if errors:
                    message += f"\n\nErrors:\n" + "\n".join(errors[:5])
                    if len(errors) > 5:
                        message += f"\n... and {len(errors) - 5} more errors"
                
                QMessageBox.information(self, "Delete Papers", message)
                
                # Refresh search results
                self._perform_search()
                
            except Exception as e:
                QMessageBox.critical(self, "Delete Error", f"Error deleting papers: {e}")
    
    def _delete_all_duplicates(self):
        """Delete all duplicate papers (with confirmation)."""
        try:
            # Count duplicates first
            all_papers = self.paper_repo.search_papers("", limit=10000)
            duplicates_count = sum(1 for paper in all_papers if paper.get('is_duplicate'))
            
            if duplicates_count == 0:
                QMessageBox.information(
                    self, "Delete Duplicates", 
                    "No duplicate papers found in database.\n\nYou may need to run 'Detect Duplicates' first."
                )
                return
            
            # Confirmation dialog
            reply = QMessageBox.question(
                self, "Delete All Duplicates",
                f"Are you sure you want to delete ALL {duplicates_count} duplicate papers?\n\n"
                "This will permanently delete papers marked as duplicates.\n"
                "This action cannot be undone!",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # Show progress
                progress = QProgressDialog("Deleting duplicates...", "Cancel", 0, 0, self)
                progress.setWindowModality(Qt.WindowModal)
                progress.setWindowTitle("Delete Duplicates")
                progress.show()
                
                # Delete duplicates
                result = self.integration_manager.delete_duplicates()
                
                progress.close()
                
                # Show results
                if 'error' in result:
                    QMessageBox.critical(
                        self, "Delete Duplicates Error", 
                        f"Error deleting duplicates:\n{result['error']}"
                    )
                else:
                    message = f"Deletion completed:\n\n"
                    message += f"Found: {result['duplicates_found']} duplicates\n"
                    message += f"Deleted: {result['deleted']} papers\n"
                    message += f"Failed: {result['failed']} papers"
                    
                    if result['failed'] > 0:
                        QMessageBox.warning(self, "Delete Duplicates", message)
                    else:
                        QMessageBox.information(self, "Delete Duplicates", message)
                    
                    # Refresh search results
                    self._perform_search()
            
        except Exception as e:
            QMessageBox.critical(self, "Delete Duplicates Error", f"Error deleting duplicates: {e}")
            logger.error(f"Error deleting duplicates: {e}")
    
    def _manage_duplicates(self):
        """Open duplicate management dialog."""
        try:
            dialog = DuplicateManagementDialog(self)
            dialog.set_integration_manager(self.integration_manager)
            dialog.exec()
            
            # Refresh search results after dialog closes
            self._perform_search()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error opening duplicate management dialog: {e}")
            logger.error(f"Error opening duplicate management dialog: {e}")
    
    def _delete_all_papers(self):
        """Delete all papers (with confirmation)."""
        # Get total count
        try:
            all_papers = self.paper_repo.list_all()
            total_count = len(all_papers)
            
            if total_count == 0:
                QMessageBox.information(self, "Delete All Papers", "No papers found in database.")
                return
            
            # Confirmation dialog
            reply = QMessageBox.question(
                self, "Delete All Papers",
                f"Are you sure you want to delete ALL {total_count} papers?\n\nThis action cannot be undone!",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # Final confirmation
                final_reply = QMessageBox.question(
                    self, "Final Confirmation",
                    f"This will permanently delete ALL {total_count} papers from the database.\n\nType 'DELETE ALL' to confirm:",
                    QMessageBox.Yes | QMessageBox.No
                )
                
                if final_reply == QMessageBox.Yes:
                    # Delete all papers
                    success_count = 0
                    failed_count = 0
                    errors = []
                    
                    for paper in all_papers:
                        try:
                            if self.paper_repo.delete_paper(paper.get('id')):
                                success_count += 1
                            else:
                                failed_count += 1
                                errors.append(f"Failed to delete paper {paper.get('id')}")
                        except Exception as e:
                            failed_count += 1
                            errors.append(f"Error deleting paper {paper.get('id')}: {str(e)}")
                    
                    # Show results
                    message = f"Deletion completed:\n"
                    message += f"Successfully deleted: {success_count}\n"
                    message += f"Failed: {failed_count}"
                    
                    if errors:
                        message += f"\n\nErrors:\n" + "\n".join(errors[:5])
                    
                    QMessageBox.information(self, "Delete All Papers", message)
                    
                    # Refresh search results
                    self._perform_search()
                    
        except Exception as e:
            QMessageBox.critical(self, "Delete Error", f"Error deleting all papers: {e}")
    
    def _show_context_menu(self, position):
        """Show context menu for results table."""
        if self.results_table.itemAt(position):
            menu = QMenu(self)
            
            # Open PDF action
            open_action = menu.addAction("Open PDF")
            open_action.triggered.connect(self._open_selected_pdf)
            
            # Edit metadata action
            edit_action = menu.addAction("Edit Metadata")
            edit_action.triggered.connect(self._edit_selected_metadata)
            
            menu.addSeparator()
            
            # Delete actions
            delete_single_action = menu.addAction("Delete This Paper")
            delete_single_action.triggered.connect(self._delete_selected_paper)
            
            delete_multiple_action = menu.addAction("Delete Selected Papers")
            delete_multiple_action.triggered.connect(self._delete_selected_papers)
            
            menu.exec(self.results_table.mapToGlobal(position))
    
    def _verify_papers(self):
        """Verify papers using DOI, ISSN, and author+title validation."""
        try:
            # Get all papers
            papers = self.paper_repo.list_all()
            
            if not papers:
                QMessageBox.information(
                    self, "Verification",
                    "No papers found to verify."
                )
                return
            
            # Papers are already dictionaries from the new repository
            # Show smart verification dialog
            dialog = SmartVerificationDialog(papers, self)
            # Pass repository reference for applying updates
            dialog.paper_repo = self.paper_repo
            # Connect signal to refresh data after updates
            dialog.data_updated.connect(self._refresh_after_verification)
            dialog.exec()
            
        except Exception as e:
            QMessageBox.critical(self, "Verification Error", f"Error starting verification: {str(e)}")
    
    def _refresh_after_verification(self):
        """Refresh the main window data after verification updates."""
        try:
            logger.info("Refreshing main window data after verification updates")
            # Refresh the search results to show updated data
            self._perform_search()
        except Exception as e:
            logger.error(f"Error refreshing data after verification: {e}")

    def _show_database_stats(self):
        """Show database statistics."""
        try:
            # Get basic statistics
            papers = self.paper_repo.list_all()
            
            # Count by status
            verified_count = sum(1 for p in papers if p.get('indexing_status') == 'SCI')
            scopus_count = sum(1 for p in papers if p.get('indexing_status') == 'Scopus')
            non_indexed_count = sum(1 for p in papers if p.get('indexing_status') == 'Non-Indexed')
            
            # Count by year
            years = {}
            for paper in papers:
                year = paper.get('year', 0)
                if year > 0:
                    years[year] = years.get(year, 0) + 1
            
            # Count by department
            departments = {}
            for paper in papers:
                dept = paper.get('department', 'Unknown')
                if dept:
                    departments[dept] = departments.get(dept, 0) + 1
            
            # Build statistics message
            stats_text = f"Database Statistics\n\n"
            stats_text += f"Total Papers: {len(papers)}\n\n"
            stats_text += f"Indexing Status:\n"
            stats_text += f"  SCI: {verified_count}\n"
            stats_text += f"  Scopus: {scopus_count}\n"
            stats_text += f"  Non-Indexed: {non_indexed_count}\n\n"
            
            if years:
                stats_text += f"Papers by Year:\n"
                for year in sorted(years.keys(), reverse=True)[:10]:  # Top 10 years
                    stats_text += f"  {year}: {years[year]}\n"
                stats_text += "\n"
            
            if departments:
                stats_text += f"Papers by Department:\n"
                for dept, count in sorted(departments.items(), key=lambda x: x[1], reverse=True)[:10]:
                    stats_text += f"  {dept}: {count}\n"
            
            QMessageBox.information(self, "Database Statistics", stats_text)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to get database statistics: {e}")
    
    def _export_table_to_pdf(self):
        """Export current table view to PDF."""
        try:
            # Get current papers data
            papers = self._get_current_papers()
            
            if not papers:
                QMessageBox.information(self, "Export PDF", "No papers to export.")
                return
            
            # Ask user for columns and author mode
            default_columns = [
                ["Title", "title"],
                ["Authors", "authors"],
                ["Year", "year"],
                ["Journal", "journal"],
                ["Department", "department"],
                ["Research Domain", "research_domain"],
                ["Indexing", "indexing_status"],
                ["Citations", "citation_count"],
                ["Quartile", "scimago_quartile"],
                ["Verification", "verification_status"]
            ]
            selected_columns = self._prompt_pdf_columns(default_columns)
            if selected_columns is None:
                return
            author_mode = self._prompt_author_mode()
            if author_mode is None:
                return

            # Get output file path
            file_path, _ = QFileDialog.getSaveFileName(
                self, 
                "Export Table to PDF", 
                f"research_papers_table_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                "PDF Files (*.pdf)"
            )
            
            if not file_path:
                return
            
            # Get current filters for title
            filters = self.search_widget.get_search_filters()
            title = "Research Papers Table"
            if any(filters.values()):
                filter_text = " | Filters: " + ", ".join([f"{k}={v}" for k, v in filters.items() if v])
                title += filter_text
            
            # Generate PDF
            success = generate_paper_table_pdf(
                papers, file_path, title, filters, selected_columns, author_mode
            )
            
            if success:
                QMessageBox.information(
                    self, 
                    "Export Complete", 
                    f"Table exported successfully to:\n{file_path}\n\nPapers exported: {len(papers)}"
                )
            else:
                QMessageBox.critical(self, "Export Error", "Failed to generate PDF. Check logs for details.")
                
        except Exception as e:
            logger.error(f"Error exporting table to PDF: {e}")
            QMessageBox.critical(self, "Export Error", f"Error exporting table: {str(e)}")
    
    def _export_summary_to_pdf(self):
        """Export summary report to PDF."""
        try:
            # Get current papers data
            papers = self._get_current_papers()
            
            if not papers:
                QMessageBox.information(self, "Export PDF", "No papers to export.")
                return
            
            # Get output file path
            file_path, _ = QFileDialog.getSaveFileName(
                self, 
                "Export Summary to PDF", 
                f"research_papers_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                "PDF Files (*.pdf)"
            )
            
            if not file_path:
                return
            
            # Get current filters for title
            filters = self.search_widget.get_search_filters()
            title = "Research Papers Summary"
            if any(filters.values()):
                filter_text = " | Filters: " + ", ".join([f"{k}={v}" for k, v in filters.items() if v])
                title += filter_text
            
            # Generate PDF
            success = generate_summary_pdf(papers, file_path, title)
            
            if success:
                QMessageBox.information(
                    self, 
                    "Export Complete", 
                    f"Summary exported successfully to:\n{file_path}\n\nPapers included: {len(papers)}"
                )
            else:
                QMessageBox.critical(self, "Export Error", "Failed to generate PDF. Check logs for details.")
                
        except Exception as e:
            logger.error(f"Error exporting summary to PDF: {e}")
            QMessageBox.critical(self, "Export Error", f"Error exporting summary: {str(e)}")
    
    def _prompt_pdf_columns(self, available: List[List[str]]):
        """Prompt the user to select which columns to include in the PDF.
        Returns a list of [header, key] or None if cancelled."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Columns for PDF")
        layout = QVBoxLayout(dialog)

        info = QLabel("Choose columns to include:")
        layout.addWidget(info)

        checks = []
        for header, key in available:
            cb = QCheckBox(header)
            cb.setChecked(True)
            cb.setProperty("col_key", key)
            layout.addWidget(cb)
            checks.append(cb)

        btns = QHBoxLayout()
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")
        btns.addStretch()
        btns.addWidget(ok_btn)
        btns.addWidget(cancel_btn)
        layout.addLayout(btns)

        selected_cols = []

        def on_ok():
            nonlocal selected_cols
            selected_cols = [[cb.text(), cb.property("col_key")] for cb in checks if cb.isChecked()]
            dialog.accept()

        def on_cancel():
            dialog.reject()

        ok_btn.clicked.connect(on_ok)
        cancel_btn.clicked.connect(on_cancel)

        if dialog.exec() == QDialog.Accepted:
            if not selected_cols:
                QMessageBox.information(self, "Export PDF", "No columns selected.")
                return None
            return selected_cols
        return None

    def _prompt_author_mode(self) -> Optional[str]:
        """Ask whether to include all authors or only the first/main author."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Author Display Option")
        layout = QVBoxLayout(dialog)

        label = QLabel("Authors to include in PDF:")
        layout.addWidget(label)

        group = QButtonGroup(dialog)
        all_radio = QRadioButton("All authors")
        first_radio = QRadioButton("Only first author")
        all_radio.setChecked(True)
        group.addButton(all_radio)
        group.addButton(first_radio)
        layout.addWidget(all_radio)
        layout.addWidget(first_radio)

        btns = QHBoxLayout()
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")
        btns.addStretch()
        btns.addWidget(ok_btn)
        btns.addWidget(cancel_btn)
        layout.addLayout(btns)

        mode: Optional[str] = None

        def on_ok():
            nonlocal mode
            mode = "first" if first_radio.isChecked() else "all"
            dialog.accept()

        def on_cancel():
            dialog.reject()

        ok_btn.clicked.connect(on_ok)
        cancel_btn.clicked.connect(on_cancel)

        if dialog.exec() == QDialog.Accepted:
            return mode
        return None
    
    def _get_current_papers(self) -> List[Dict[str, Any]]:
        """Get current papers displayed in the table."""
        try:
            # Get papers from the current search results
            papers = []
            for row in range(self.results_table.rowCount()):
                title_item = self.results_table.item(row, 0)
                if title_item:
                    # Get paper ID from UserRole data
                    paper_id = title_item.data(Qt.UserRole)
                    if paper_id:
                        paper = self.paper_repo.get_paper_by_id(paper_id)
                        if paper:
                            papers.append(paper)
            return papers
        except Exception as e:
            logger.error(f"Error getting current papers: {e}")
            return []
    
    def _generate_all_embeddings(self):
        """Generate embeddings for all papers."""
        try:
            # Show progress dialog
            progress = QProgressDialog("Generating embeddings for all papers...", "Cancel", 0, 0, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            
            # Generate embeddings
            embeddings = self.semantic_search_engine.generate_all_embeddings()
            
            progress.close()
            
            QMessageBox.information(
                self, "Embeddings Generated",
                f"Successfully generated embeddings for {len(embeddings)} papers!\n\n"
                f"Semantic search is now ready to use."
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate embeddings: {e}")
    
    def _clear_embeddings_cache(self):
        """Clear the embeddings cache."""
        try:
            self.semantic_search_engine.clear_cache()
            QMessageBox.information(
                self, "Cache Cleared",
                "Embeddings cache has been cleared.\n\n"
                "You may need to regenerate embeddings for semantic search to work."
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to clear cache: {e}")
    
    def _show_embedding_stats(self):
        """Show embedding statistics."""
        try:
            stats = self.semantic_search_engine.get_embedding_stats()
            
            stats_text = f"Embedding Statistics:\n\n"
            stats_text += f"Total Embeddings: {stats['total_embeddings']}\n"
            stats_text += f"Model: {stats['model_name']}\n"
            stats_text += f"Dimension: {stats['dimension']}\n"
            
            if stats['cached_paper_ids']:
                stats_text += f"\nCached Paper IDs: {len(stats['cached_paper_ids'])} papers\n"
                stats_text += f"First 10 IDs: {stats['cached_paper_ids'][:10]}\n"
            else:
                stats_text += "\nNo embeddings cached. Generate embeddings to enable semantic search."
            
            QMessageBox.information(self, "Embedding Statistics", stats_text)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to get embedding statistics: {e}")
    
    def _refresh_citations(self):
        """Refresh citation data for all papers."""
        try:
            # Get all papers
            papers = self.paper_repo.list_all()
            
            if not papers:
                QMessageBox.information(
                    self, "Refresh Citations",
                    "No papers found to refresh citations for."
                )
                return
            
            # Show confirmation dialog
            reply = QMessageBox.question(
                self, "Refresh Citations",
                f"Refresh citation data for {len(papers)} papers?\n\n"
                f"This will fetch citation counts and SCImago quartile information from APIs.\n"
                f"This may take several minutes. Continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply != QMessageBox.Yes:
                return
            
            # Show progress dialog
            progress = QProgressDialog("Refreshing citations...", "Cancel", 0, len(papers), self)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            
            # Import citation fetcher
            from ..utils.citation_fetcher import fetch_citation_data
            
            success_count = 0
            error_count = 0
            errors = []
            
            for i, paper in enumerate(papers):
                try:
                    progress.setValue(i)
                    progress.setLabelText(f"Refreshing citations for paper {i+1}/{len(papers)}: {paper.title[:50]}...")
                    
                    # Fetch citation data
                    citation_data = fetch_citation_data(
                        paper.doi, paper.title, paper.journal, paper.year
                    )
                    
                    if citation_data.success:
                        # Update paper with citation data
                        updates = {
                            'citation_count': citation_data.citation_count,
                            'scimago_quartile': citation_data.scimago_quartile,
                            'impact_factor': citation_data.impact_factor,
                            'h_index': citation_data.h_index,
                            'citation_source': citation_data.source,
                            'citation_updated_at': citation_data.last_updated
                        }
                        
                        # Update in database
                        if hasattr(self.paper_repo, 'update_paper_metadata'):
                            success = self.paper_repo.update_paper_metadata(paper.get('id'), updates)
                            if success:
                                success_count += 1
                            else:
                                error_count += 1
                                errors.append(f"Paper {paper.get('id')}: Database update failed")
                        else:
                            error_count += 1
                            errors.append(f"Paper {paper.get('id')}: No update method available")
                    else:
                        error_count += 1
                        errors.append(f"Paper {paper.get('id')}: {citation_data.error}")
                    
                    # Check if cancelled
                    if progress.wasCanceled():
                        break
                        
                except Exception as e:
                    error_count += 1
                    errors.append(f"Paper {paper.get('id')}: {str(e)}")
            
            progress.close()
            
            # Show results
            if error_count == 0:
                QMessageBox.information(
                    self, "Citations Refreshed",
                    f"Successfully refreshed citations for {success_count} papers!\n\n"
                    f"Citation data includes:\n"
                    f"- Citation counts from Crossref/Google Scholar\n"
                    f"- SCImago quartile rankings\n"
                    f"- Journal impact factors"
                )
            else:
                QMessageBox.warning(
                    self, "Citations Refresh Complete with Errors",
                    f"Refreshed {success_count} papers successfully.\n"
                    f"Failed to refresh {error_count} papers.\n\n"
                    f"Errors:\n" + "\n".join(errors[:5]) + 
                    (f"\n... and {len(errors)-5} more errors" if len(errors) > 5 else "")
                )
            
            # Refresh search results to show updated data
            self._perform_search()
            
        except Exception as e:
            QMessageBox.critical(self, "Refresh Citations Error", f"Error refreshing citations: {str(e)}")
    
    def _update_indexing_status(self):
        """Update indexing status for all papers based on journal information."""
        try:
            # Get all papers
            papers = self.paper_repo.list_all()
            
            if not papers:
                QMessageBox.information(
                    self, "Update Indexing Status",
                    "No papers found to update indexing status for."
                )
                return
            
            # Show confirmation dialog
            reply = QMessageBox.question(
                self, "Update Indexing Status",
                f"Update indexing status for {len(papers)} papers?\n\n"
                f"This will analyze journal names and publishers to determine\n"
                f"if papers are indexed in SCI, Scopus, or other databases.\n\n"
                f"Continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply != QMessageBox.Yes:
                return
            
            # Show progress dialog
            progress = QProgressDialog("Updating indexing status...", "Cancel", 0, len(papers), self)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            
            # Import the indexing status determination function
            from ..utils.post_import_verifier import PostImportVerifier
            verifier = PostImportVerifier()
            
            success_count = 0
            error_count = 0
            errors = []
            
            for i, paper in enumerate(papers):
                try:
                    progress.setValue(i)
                    progress.setLabelText(f"Updating indexing status for paper {i+1}/{len(papers)}: {paper.title[:50]}...")
                    
                    # Create metadata dict for indexing status determination
                    metadata = {
                        'journal': paper.journal or '',
                        'publisher': paper.publisher or '',
                        'issn': getattr(paper, 'issn', '') or ''
                    }
                    
                    # Determine indexing status
                    indexing_status = verifier._determine_indexing_status(metadata)
                    
                    # Update paper with indexing status
                    updates = {'indexing_status': indexing_status}
                    success = self.paper_repo.update_paper_metadata(paper.get('id'), updates)
                    
                    if success:
                        success_count += 1
                    else:
                        error_count += 1
                        errors.append(f"Paper {paper.get('id')}: Database update failed")
                    
                    # Check if cancelled
                    if progress.wasCanceled():
                        break
                        
                except Exception as e:
                    error_count += 1
                    errors.append(f"Paper {paper.get('id')}: {str(e)}")
            
            progress.close()
            
            # Show results
            if error_count == 0:
                QMessageBox.information(
                    self, "Indexing Status Updated",
                    f"Successfully updated indexing status for {success_count} papers!\n\n"
                    f"Indexing statuses determined:\n"
                    f"- SCI: High-impact journals\n"
                    f"- Scopus: Broader database coverage\n"
                    f"- SCI + Scopus: Both databases\n"
                    f"- Conference Proceedings: Conference papers\n"
                    f"- Open Access: Open access journals\n"
                    f"- Preprint: Preprint servers\n"
                    f"- Unknown: No clear pattern match"
                )
            else:
                QMessageBox.warning(
                    self, "Indexing Status Update Complete with Errors",
                    f"Updated {success_count} papers successfully.\n"
                    f"Failed to update {error_count} papers.\n\n"
                    f"Errors:\n" + "\n".join(errors[:5]) + 
                    (f"\n... and {len(errors)-5} more errors" if len(errors) > 5 else "")
                )
            
            # Refresh search results to show updated data
            self._perform_search()
            
        except Exception as e:
            QMessageBox.critical(self, "Update Indexing Status Error", f"Error updating indexing status: {str(e)}")
    
    def _is_development_mode(self) -> bool:
        """Check if running in development mode."""
        # Check for development environment variables or files
        dev_indicators = [
            os.environ.get('RESEARCH_PAPER_BROWSER_DEV', '').lower() == 'true',
            os.path.exists('.dev'),
            os.path.exists('dev_mode'),
            '--dev' in sys.argv,
            '--development' in sys.argv
        ]
        return any(dev_indicators)
    
    def _refresh_ui(self):
        """Refresh the UI after code changes."""
        try:
            logger.info("Refreshing UI due to code changes...")
            
            # Show a notification to the user
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(
                self, "Code Reloaded",
                "Code changes detected and reloaded!\n\n"
                "Some changes may require a full restart to take effect.\n"
                "If you experience issues, please restart the application."
            )
            
            # Refresh the search results to show any data changes
            self._perform_search()
            
            logger.info("UI refresh completed")
            
        except Exception as e:
            logger.error(f"Failed to refresh UI: {e}")
    
    def _show_about(self):
        """Show about dialog."""
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.about(
            self, 
            "About Research Paper Browser",
            f"""
            <h3>Research Paper Browser v2.0</h3>
            <p>A comprehensive tool for managing and analyzing research papers.</p>
            <p><b>Features:</b></p>
            <ul>
            <li>PDF import and metadata extraction</li>
            <li>Advanced search (semantic, keyword, hybrid)</li>
            <li>Paper verification and validation</li>
            <li>Citation tracking and indexing</li>
            <li>Department and domain classification</li>
            </ul>
            <p><b>Database:</b> {DB_BACKEND}</p>
            <p><b>Papers:</b> {len(self.paper_repo.list_all())} papers in database</p>
            """
        )
    
    def _show_shortcuts(self):
        """Show keyboard shortcuts dialog."""
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information(
            self,
            "Keyboard Shortcuts",
            """
            <h3>Keyboard Shortcuts</h3>
            <p><b>Ctrl+E</b> - Edit selected paper metadata</p>
            <p><b>Delete</b> - Delete selected paper</p>
            <p><b>Enter</b> - View paper details</p>
            <p><b>Ctrl+O</b> - Import single PDF</p>
            <p><b>Ctrl+Shift+O</b> - Bulk import PDFs</p>
            <p><b>Ctrl+F</b> - Focus search box</p>
            <p><b>Ctrl+R</b> - Refresh search results</p>
            <p><b>F5</b> - Refresh all data</p>
            <p><b>Ctrl+P</b> - Export table to PDF</p>
            <p><b>Ctrl+Shift+P</b> - Export summary to PDF</p>
            <p><b>Ctrl+Q</b> - Quit application</p>
            """
        )
    
    def closeEvent(self, event):
        """Handle application close event."""
        try:
            # Clean up resources
            logger.info("Application closing")
        except Exception as e:
            logger.error(f"Error during application close: {e}")
        
        # Call parent close event
        super().closeEvent(event)
    
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts."""
        if event.key() == Qt.Key_E and event.modifiers() == Qt.ControlModifier:
            # Ctrl+E for edit
            self._edit_selected_metadata()
        elif event.key() == Qt.Key_Delete:
            # Delete key for delete
            self._delete_selected_paper()
        elif event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            # Enter key for view details
            current_row = self.results_table.currentRow()
            if current_row >= 0:
                self._open_paper_details(current_row, 0)
        elif event.key() == Qt.Key_O and event.modifiers() == Qt.ControlModifier:
            # Ctrl+O for import single PDF
            self._import_single_pdf()
        elif event.key() == Qt.Key_O and event.modifiers() == (Qt.ControlModifier | Qt.ShiftModifier):
            # Ctrl+Shift+O for bulk import
            self._bulk_import_pdfs()
        elif event.key() == Qt.Key_F and event.modifiers() == Qt.ControlModifier:
            # Ctrl+F for focus search
            self.search_widget.query_edit.setFocus()
        elif event.key() == Qt.Key_R and event.modifiers() == Qt.ControlModifier:
            # Ctrl+R for refresh search
            self._perform_search()
        elif event.key() == Qt.Key_F5:
            # F5 for refresh all data
            self._load_initial_data()
        elif event.key() == Qt.Key_P and event.modifiers() == Qt.ControlModifier:
            # Ctrl+P for export table to PDF
            self._export_table_to_pdf()
        elif event.key() == Qt.Key_P and event.modifiers() == (Qt.ControlModifier | Qt.ShiftModifier):
            # Ctrl+Shift+P for export summary to PDF
            self._export_summary_to_pdf()
        elif event.key() == Qt.Key_Q and event.modifiers() == Qt.ControlModifier:
            # Ctrl+Q for quit
            self.close()
        else:
            super().keyPressEvent(event)


def launch_enhanced_app():
    """Launch the enhanced application."""
    app = QApplication([])
    window = EnhancedMainWindow()
    window.show()
    app.exec()
