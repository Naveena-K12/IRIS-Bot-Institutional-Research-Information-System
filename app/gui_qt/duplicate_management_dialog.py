"""
Duplicate Management Dialog
Provides an interactive interface for reviewing and managing duplicate papers.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox,
    QSplitter, QTextEdit, QProgressDialog, QMessageBox,
    QCheckBox, QTabWidget, QWidget, QFormLayout, QSpinBox,
    QComboBox, QListWidget, QListWidgetItem, QApplication
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor, QFont

logger = logging.getLogger(__name__)


class DuplicateDetectionThread(QThread):
    """Thread for detecting duplicates in the background."""
    
    progress_updated = Signal(str, int, int)  # status, current, total
    duplicate_found = Signal(dict)  # duplicate group
    detection_complete = Signal(dict)  # results
    error_occurred = Signal(str)  # error message
    
    def __init__(self, integration_manager, limit: int = 10000):
        super().__init__()
        self.integration_manager = integration_manager
        self.limit = limit
        self.should_stop = False
    
    def run(self):
        """Detect duplicates."""
        try:
            self.progress_updated.emit("Scanning database...", 0, 100)
            
            # Get all papers
            all_papers = self.integration_manager.paper_repo.search_papers("", limit=self.limit)
            
            if len(all_papers) < 2:
                self.detection_complete.emit({
                    'total_papers': len(all_papers),
                    'duplicate_groups': [],
                    'papers_checked': 0
                })
                return
            
            self.progress_updated.emit(f"Checking {len(all_papers)} papers for duplicates...", 0, len(all_papers))
            
            # Group duplicates
            duplicate_groups = []
            processed_ids = set()
            
            for i, paper in enumerate(all_papers):
                if self.should_stop:
                    break
                
                paper_id = paper.get('id')
                if paper_id in processed_ids or paper.get('is_duplicate'):
                    continue
                
                self.progress_updated.emit(
                    f"Checking paper {i+1}/{len(all_papers)}: {paper.get('title', 'Unknown')[:50]}...",
                    i + 1,
                    len(all_papers)
                )
                
                # Find duplicates for this paper
                duplicates = self.integration_manager._check_for_duplicates(paper)
                
                if duplicates:
                    # Create duplicate group
                    group = {
                        'original': paper,
                        'duplicates': []
                    }
                    
                    for dup_id, similarity, reason in duplicates:
                        # Find duplicate paper
                        dup_paper = next((p for p in all_papers if p.get('id') == dup_id), None)
                        if dup_paper and not dup_paper.get('is_duplicate'):
                            group['duplicates'].append({
                                'paper': dup_paper,
                                'similarity': similarity,
                                'reason': reason
                            })
                            processed_ids.add(dup_id)
                    
                    if group['duplicates']:
                        duplicate_groups.append(group)
                        self.duplicate_found.emit(group)
                
                processed_ids.add(paper_id)
            
            self.detection_complete.emit({
                'total_papers': len(all_papers),
                'duplicate_groups': duplicate_groups,
                'papers_checked': len(all_papers)
            })
            
        except Exception as e:
            logger.error(f"Error in duplicate detection thread: {e}")
            self.error_occurred.emit(str(e))
    
    def stop(self):
        """Stop the detection process."""
        self.should_stop = True


class DuplicateManagementDialog(QDialog):
    """Dialog for managing duplicate papers."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.integration_manager = None
        self.duplicate_groups = []
        self.selected_for_deletion = set()  # Paper IDs to delete
        
        self.setWindowTitle("Duplicate Management")
        self.setMinimumSize(1200, 700)
        
        self.setup_ui()
        self.setup_connections()
    
    def set_integration_manager(self, integration_manager):
        """Set the integration manager."""
        self.integration_manager = integration_manager
    
    def setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Header with better styling
        header = QLabel("<h2 style='color: #2c3e50; margin: 10px;'>Duplicate Paper Management</h2>")
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("background-color: #ecf0f1; padding: 15px; border-radius: 8px;")
        layout.addWidget(header)
        
        # Clear instructions with better contrast
        instructions = QLabel(
            "<b>Instructions:</b><br>"
            "1. Click 'Detect Duplicates' to scan your database<br>"
            "2. Select a duplicate group from the left panel<br>"
            "3. Review papers in each group - the <b style='color: #27ae60;'>KEEP</b> paper (original) is marked in green<br>"
            "4. Check boxes next to duplicate papers you want to <b style='color: #e74c3c;'>DELETE</b><br>"
            "5. Click 'Delete Selected' to remove checked duplicates"
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet(
            "background-color: #ffffff; "
            "color: #2c3e50; "
            "padding: 15px; "
            "border-radius: 5px; "
            "border: 2px solid #3498db; "
            "font-size: 12pt;"
        )
        layout.addWidget(instructions)
        
        # Main splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel: Duplicate groups list
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(10)
        
        # Group selection with better styling
        group_label = QLabel("<b style='color: #2c3e50; font-size: 11pt;'>Duplicate Groups:</b>")
        group_label.setStyleSheet("padding: 5px;")
        left_layout.addWidget(group_label)
        
        self.groups_list = QListWidget()
        self.groups_list.setMaximumWidth(320)
        self.groups_list.setStyleSheet(
            "QListWidget {"
            "  background-color: #ffffff;"
            "  border: 2px solid #bdc3c7;"
            "  border-radius: 5px;"
            "  padding: 5px;"
            "  font-size: 10pt;"
            "}"
            "QListWidget::item {"
            "  padding: 8px;"
            "  border-bottom: 1px solid #ecf0f1;"
            "  color: #2c3e50;"
            "}"
            "QListWidget::item:hover {"
            "  background-color: #ecf0f1;"
            "}"
            "QListWidget::item:selected {"
            "  background-color: #3498db;"
            "  color: #ffffff;"
            "}"
        )
        left_layout.addWidget(self.groups_list)
        
        # Statistics with better styling
        stats_group = QGroupBox("Statistics")
        stats_group.setStyleSheet(
            "QGroupBox {"
            "  font-weight: bold;"
            "  color: #2c3e50;"
            "  border: 2px solid #bdc3c7;"
            "  border-radius: 5px;"
            "  margin-top: 10px;"
            "  padding-top: 10px;"
            "}"
            "QGroupBox::title {"
            "  subcontrol-origin: margin;"
            "  left: 10px;"
            "  padding: 0 5px;"
            "}"
        )
        stats_layout = QVBoxLayout(stats_group)
        self.stats_label = QLabel("No duplicates found.<br>Click 'Detect Duplicates' to scan.")
        self.stats_label.setWordWrap(True)
        self.stats_label.setStyleSheet(
            "color: #7f8c8d;"
            "padding: 10px;"
            "background-color: #ecf0f1;"
            "border-radius: 3px;"
        )
        stats_layout.addWidget(self.stats_label)
        left_layout.addWidget(stats_group)
        
        splitter.addWidget(left_panel)
        
        # Right panel: Duplicate comparison
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Comparison view with better organization
        self.comparison_widget = QWidget()
        comparison_layout = QVBoxLayout(self.comparison_widget)
        comparison_layout.setSpacing(10)
        
        # Title with clear status
        title_container = QWidget()
        title_layout = QHBoxLayout(title_container)
        title_layout.setContentsMargins(10, 10, 10, 10)
        
        self.comparison_title = QLabel("<b style='color: #2c3e50; font-size: 12pt;'>Select a duplicate group to review</b>")
        self.comparison_title.setAlignment(Qt.AlignLeft)
        title_layout.addWidget(self.comparison_title)
        title_layout.addStretch()
        
        # Status indicator
        self.status_indicator = QLabel("")
        self.status_indicator.setStyleSheet(
            "padding: 5px 15px;"
            "border-radius: 15px;"
            "font-weight: bold;"
            "font-size: 10pt;"
        )
        title_layout.addWidget(self.status_indicator)
        
        title_container.setStyleSheet("background-color: #ecf0f1; border-radius: 5px;")
        comparison_layout.addWidget(title_container)
        
        # Duplicate pairs table with better styling
        self.comparison_table = QTableWidget()
        self.comparison_table.setColumnCount(6)
        self.comparison_table.setHorizontalHeaderLabels([
            "Action", "Status", "Paper ID", "Title", "Authors", "Similarity"
        ])
        
        # Better column sizing
        header = self.comparison_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # Action
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # Status
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # Paper ID
        header.setSectionResizeMode(3, QHeaderView.Stretch)  # Title
        header.setSectionResizeMode(4, QHeaderView.Stretch)  # Authors
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)  # Similarity
        
        # Style the table
        self.comparison_table.setStyleSheet(
            "QTableWidget {"
            "  background-color: #ffffff;"
            "  border: 2px solid #bdc3c7;"
            "  border-radius: 5px;"
            "  gridline-color: #ecf0f1;"
            "  font-size: 10pt;"
            "}"
            "QTableWidget::item {"
            "  padding: 8px;"
            "  color: #2c3e50;"
            "}"
            "QHeaderView::section {"
            "  background-color: #34495e;"
            "  color: #ffffff;"
            "  padding: 8px;"
            "  border: none;"
            "  font-weight: bold;"
            "  font-size: 10pt;"
            "}"
            "QTableWidget::item:selected {"
            "  background-color: #3498db;"
            "  color: #ffffff;"
            "}"
        )
        
        self.comparison_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.comparison_table.setSelectionMode(QTableWidget.SingleSelection)
        self.comparison_table.setAlternatingRowColors(True)
        comparison_layout.addWidget(self.comparison_table)
        
        # Details view with better styling
        details_group = QGroupBox("Paper Details & Duplicate Reason")
        details_group.setStyleSheet(
            "QGroupBox {"
            "  font-weight: bold;"
            "  color: #2c3e50;"
            "  border: 2px solid #bdc3c7;"
            "  border-radius: 5px;"
            "  margin-top: 10px;"
            "  padding-top: 10px;"
            "}"
            "QGroupBox::title {"
            "  subcontrol-origin: margin;"
            "  left: 10px;"
            "  padding: 0 5px;"
            "}"
        )
        details_layout = QVBoxLayout(details_group)
        
        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setMaximumHeight(120)
        self.details_text.setStyleSheet(
            "QTextEdit {"
            "  background-color: #f8f9fa;"
            "  color: #2c3e50;"
            "  border: 1px solid #bdc3c7;"
            "  border-radius: 3px;"
            "  padding: 8px;"
            "  font-size: 10pt;"
            "}"
        )
        details_layout.addWidget(self.details_text)
        
        comparison_layout.addWidget(details_group)
        
        right_layout.addWidget(self.comparison_widget)
        
        splitter.addWidget(right_panel)
        splitter.setSizes([300, 900])
        
        layout.addWidget(splitter)
        
        # Action buttons with better styling and clarity
        button_container = QWidget()
        button_container.setStyleSheet("background-color: #ecf0f1; padding: 10px; border-radius: 5px;")
        button_layout = QHBoxLayout(button_container)
        button_layout.setSpacing(10)
        
        # Detection button
        self.detect_button = QPushButton("🔍 Detect Duplicates")
        self.detect_button.setStyleSheet(
            "QPushButton {"
            "  background-color: #3498db;"
            "  color: #ffffff;"
            "  padding: 10px 20px;"
            "  border: none;"
            "  border-radius: 5px;"
            "  font-weight: bold;"
            "  font-size: 11pt;"
            "}"
            "QPushButton:hover {"
            "  background-color: #2980b9;"
            "}"
            "QPushButton:pressed {"
            "  background-color: #21618c;"
            "}"
        )
        button_layout.addWidget(self.detect_button)
        
        button_layout.addWidget(QLabel("│"))  # Separator
        
        # Selection buttons
        self.select_all_button = QPushButton("✓ Select All Duplicates")
        self.select_all_button.setStyleSheet(
            "QPushButton {"
            "  background-color: #95a5a6;"
            "  color: #ffffff;"
            "  padding: 10px 15px;"
            "  border: none;"
            "  border-radius: 5px;"
            "  font-size: 10pt;"
            "}"
            "QPushButton:hover {"
            "  background-color: #7f8c8d;"
            "}"
        )
        button_layout.addWidget(self.select_all_button)
        
        self.deselect_all_button = QPushButton("✗ Clear Selection")
        self.deselect_all_button.setStyleSheet(
            "QPushButton {"
            "  background-color: #95a5a6;"
            "  color: #ffffff;"
            "  padding: 10px 15px;"
            "  border: none;"
            "  border-radius: 5px;"
            "  font-size: 10pt;"
            "}"
            "QPushButton:hover {"
            "  background-color: #7f8c8d;"
            "}"
        )
        button_layout.addWidget(self.deselect_all_button)
        
        button_layout.addStretch()
        
        # Delete button with counter
        self.delete_button = QPushButton("🗑 Delete Selected (0)")
        self.delete_button.setEnabled(False)
        self.delete_button.setStyleSheet(
            "QPushButton {"
            "  background-color: #e74c3c;"
            "  color: #ffffff;"
            "  padding: 10px 20px;"
            "  border: none;"
            "  border-radius: 5px;"
            "  font-weight: bold;"
            "  font-size: 11pt;"
            "}"
            "QPushButton:hover:enabled {"
            "  background-color: #c0392b;"
            "}"
            "QPushButton:disabled {"
            "  background-color: #bdc3c7;"
            "  color: #7f8c8d;"
            "}"
        )
        button_layout.addWidget(self.delete_button)
        
        # Close button
        self.close_button = QPushButton("Close")
        self.close_button.setStyleSheet(
            "QPushButton {"
            "  background-color: #95a5a6;"
            "  color: #ffffff;"
            "  padding: 10px 20px;"
            "  border: none;"
            "  border-radius: 5px;"
            "  font-size: 10pt;"
            "}"
            "QPushButton:hover {"
            "  background-color: #7f8c8d;"
            "}"
        )
        button_layout.addWidget(self.close_button)
        
        layout.addWidget(button_container)
    
    def setup_connections(self):
        """Setup signal connections."""
        self.groups_list.itemSelectionChanged.connect(self.on_group_selected)
        self.comparison_table.cellChanged.connect(self.on_comparison_selection_changed)
        self.comparison_table.itemSelectionChanged.connect(self.on_comparison_row_selected)
        
        self.detect_button.clicked.connect(self.start_detection)
        self.select_all_button.clicked.connect(self.select_all_duplicates)
        self.deselect_all_button.clicked.connect(self.deselect_all_duplicates)
        self.delete_button.clicked.connect(self.delete_selected)
        self.close_button.clicked.connect(self.accept)
    
    def start_detection(self):
        """Start duplicate detection."""
        if not self.integration_manager:
            QMessageBox.warning(self, "Error", "Integration manager not set.")
            return
        
        # Clear previous results
        self.duplicate_groups = []
        self.selected_for_deletion.clear()
        self.groups_list.clear()
        self.comparison_table.setRowCount(0)
        self.details_text.clear()
        self.delete_button.setEnabled(False)
        
        # Show progress
        progress = QProgressDialog("Detecting duplicates...", "Cancel", 0, 0, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setWindowTitle("Detect Duplicates")
        progress.show()
        
        # Run detection in thread
        self.detection_thread = DuplicateDetectionThread(self.integration_manager)
        self.detection_thread.duplicate_found.connect(lambda group: self.add_duplicate_group(group))
        self.detection_thread.detection_complete.connect(
            lambda result: self.on_detection_complete(progress, result)
        )
        self.detection_thread.error_occurred.connect(
            lambda error: self.on_detection_error(progress, error)
        )
        self.detection_thread.start()
        
        # Keep progress dialog open
        while self.detection_thread.isRunning():
            progress.setValue(0)
            QApplication.processEvents()
        
        progress.close()
    
    def add_duplicate_group(self, group: dict):
        """Add a duplicate group to the list."""
        self.duplicate_groups.append(group)
        
        original = group['original']
        duplicates_count = len(group['duplicates'])
        
        item_text = f"Group {len(self.duplicate_groups)}: {duplicates_count + 1} papers"
        item = QListWidgetItem(item_text)
        item.setData(Qt.UserRole, len(self.duplicate_groups) - 1)  # Store group index
        self.groups_list.addItem(item)
    
    def on_detection_complete(self, progress, result):
        """Handle detection completion."""
        progress.close()
        
        total_groups = len(result['duplicate_groups'])
        total_duplicates = sum(len(g['duplicates']) for g in result['duplicate_groups'])
        
        self.stats_label.setText(
            f"<b>Statistics:</b><br>"
            f"Total Papers: {result['total_papers']}<br>"
            f"Papers Checked: {result['papers_checked']}<br>"
            f"Duplicate Groups: {total_groups}<br>"
            f"Total Duplicates: {total_duplicates}"
        )
        
        if total_groups == 0:
            QMessageBox.information(
                self, "Detection Complete",
                "No duplicates found in the database!"
            )
        else:
            QMessageBox.information(
                self, "Detection Complete",
                f"Found {total_groups} duplicate groups with {total_duplicates} duplicate papers.\n\n"
                "Review each group and select which papers to delete."
            )
    
    def on_detection_error(self, progress, error):
        """Handle detection error."""
        progress.close()
        QMessageBox.critical(self, "Detection Error", f"Error detecting duplicates:\n{error}")
    
    def on_group_selected(self):
        """Handle group selection."""
        current_item = self.groups_list.currentItem()
        if not current_item:
            return
        
        group_index = current_item.data(Qt.UserRole)
        if group_index is None or group_index >= len(self.duplicate_groups):
            return
        
        self.show_group_comparison(group_index)
    
    def show_group_comparison(self, group_index: int):
        """Show comparison for a duplicate group."""
        if group_index >= len(self.duplicate_groups):
            return
        
        group = self.duplicate_groups[group_index]
        original = group['original']
        duplicates = group['duplicates']
        
        # Update title and status
        self.comparison_title.setText(
            f"<b style='color: #2c3e50;'>Group {group_index + 1}: {len(duplicates) + 1} Papers</b>"
        )
        
        selected_count = sum(1 for dup in duplicates if dup['paper'].get('id') in self.selected_for_deletion)
        if selected_count > 0:
            self.status_indicator.setText(f"⚠ {selected_count} selected for deletion")
            self.status_indicator.setStyleSheet(
                "background-color: #f39c12;"
                "color: #ffffff;"
                "padding: 5px 15px;"
                "border-radius: 15px;"
                "font-weight: bold;"
                "font-size: 10pt;"
            )
        else:
            self.status_indicator.setText("✓ No selections")
            self.status_indicator.setStyleSheet(
                "background-color: #27ae60;"
                "color: #ffffff;"
                "padding: 5px 15px;"
                "border-radius: 15px;"
                "font-weight: bold;"
                "font-size: 10pt;"
            )
        
        # Populate comparison table
        self.comparison_table.setRowCount(len(duplicates) + 1)
        
        # Add original paper (always keep) - CLEARLY MARKED
        original_id = original.get('id')
        original_year = original.get('year', 'N/A')
        
        # Action column - KEEP label (no checkbox)
        keep_label = QLabel("KEEP")
        keep_label.setStyleSheet(
            "background-color: #27ae60;"
            "color: #ffffff;"
            "padding: 5px 10px;"
            "border-radius: 3px;"
            "font-weight: bold;"
            "font-size: 9pt;"
        )
        keep_label.setAlignment(Qt.AlignCenter)
        self.comparison_table.setCellWidget(0, 0, keep_label)
        
        # Status column - Original badge
        status_label = QLabel("ORIGINAL")
        status_label.setStyleSheet(
            "background-color: #27ae60;"
            "color: #ffffff;"
            "padding: 5px 10px;"
            "border-radius: 3px;"
            "font-weight: bold;"
            "font-size: 9pt;"
        )
        status_label.setAlignment(Qt.AlignCenter)
        self.comparison_table.setCellWidget(0, 1, status_label)
        
        # Paper info with dark text on light background
        id_item = QTableWidgetItem(str(original_id))
        id_item.setForeground(QColor(0, 0, 0))  # Black text
        self.comparison_table.setItem(0, 2, id_item)
        
        title_item = QTableWidgetItem(original.get('title', ''))
        title_item.setForeground(QColor(0, 0, 0))  # Black text
        self.comparison_table.setItem(0, 3, title_item)
        
        authors_item = QTableWidgetItem(original.get('authors', ''))
        authors_item.setForeground(QColor(0, 0, 0))  # Black text
        self.comparison_table.setItem(0, 4, authors_item)
        
        similarity_item = QTableWidgetItem("—")
        similarity_item.setForeground(QColor(0, 0, 0))  # Black text
        similarity_item.setToolTip("This is the original paper")
        self.comparison_table.setItem(0, 5, similarity_item)
        
        # Mark original row with light green background
        for col in range(6):
            item = self.comparison_table.item(0, col)
            if item:
                item.setBackground(QColor(212, 237, 218))  # Light green - good contrast
                item.setForeground(QColor(0, 0, 0))  # Black text
        
        # Store original info for details
        id_item.setData(Qt.UserRole, {'type': 'original', 'paper': original})
        
        # Add duplicates with clear DELETE option
        for i, dup_info in enumerate(duplicates):
            row = i + 1
            dup_paper = dup_info['paper']
            dup_id = dup_paper.get('id')
            similarity = dup_info['similarity']
            reason = dup_info['reason']
            dup_year = dup_paper.get('year', 'N/A')
            
            # Action column - DELETE checkbox with clear label
            checkbox_container = QWidget()
            checkbox_layout = QHBoxLayout(checkbox_container)
            checkbox_layout.setContentsMargins(5, 2, 5, 2)
            checkbox_layout.setAlignment(Qt.AlignCenter)
            
            checkbox = QCheckBox("DELETE")
            checkbox.setChecked(dup_id in self.selected_for_deletion)
            checkbox.stateChanged.connect(
                lambda state, pid=dup_id: self.on_duplicate_checkbox_changed(pid, state)
            )
            checkbox.setStyleSheet(
                "QCheckBox {"
                "  color: #2c3e50;"
                "  font-weight: bold;"
                "  font-size: 9pt;"
                "}"
                "QCheckBox::indicator {"
                "  width: 18px;"
                "  height: 18px;"
                "}"
                "QCheckBox::indicator:checked {"
                "  background-color: #e74c3c;"
                "  border: 2px solid #c0392b;"
                "}"
            )
            checkbox_layout.addWidget(checkbox)
            checkbox_container.setStyleSheet("background-color: #ffffff;")
            self.comparison_table.setCellWidget(row, 0, checkbox_container)
            
            # Status column - DUPLICATE badge
            dup_status = QLabel("DUPLICATE")
            if similarity >= 0.95:
                dup_status.setStyleSheet(
                    "background-color: #e74c3c;"
                    "color: #ffffff;"
                    "padding: 5px 10px;"
                    "border-radius: 3px;"
                    "font-weight: bold;"
                    "font-size: 9pt;"
                )
            else:
                dup_status.setStyleSheet(
                    "background-color: #f39c12;"
                    "color: #ffffff;"
                    "padding: 5px 10px;"
                    "border-radius: 3px;"
                    "font-weight: bold;"
                    "font-size: 9pt;"
                )
            dup_status.setAlignment(Qt.AlignCenter)
            self.comparison_table.setCellWidget(row, 1, dup_status)
            
            # Paper info - always dark text
            id_item = QTableWidgetItem(str(dup_id))
            id_item.setForeground(QColor(0, 0, 0))  # Black text
            self.comparison_table.setItem(row, 2, id_item)
            
            title_item = QTableWidgetItem(dup_paper.get('title', ''))
            title_item.setForeground(QColor(0, 0, 0))  # Black text
            self.comparison_table.setItem(row, 3, title_item)
            
            authors_item = QTableWidgetItem(dup_paper.get('authors', ''))
            authors_item.setForeground(QColor(0, 0, 0))  # Black text
            self.comparison_table.setItem(row, 4, authors_item)
            
            # Similarity with color coding
            similarity_text = f"{similarity * 100:.1f}%"
            similarity_item = QTableWidgetItem(similarity_text)
            similarity_item.setForeground(QColor(0, 0, 0))  # Black text
            similarity_item.setToolTip(f"Match reason: {reason}")
            self.comparison_table.setItem(row, 5, similarity_item)
            
            # Store paper and reason in user role
            id_item.setData(Qt.UserRole, {
                'type': 'duplicate',
                'paper': dup_paper,
                'similarity': similarity,
                'reason': reason
            })
            
            # Background color based on similarity - but ensure text is readable
            if similarity >= 0.95:
                # High similarity - light red background, black text
                bg_color = QColor(255, 235, 238)  # Very light red
                text_color = QColor(0, 0, 0)  # Black text
            else:
                # Medium similarity - light orange background, black text
                bg_color = QColor(255, 248, 220)  # Very light yellow
                text_color = QColor(0, 0, 0)  # Black text
            
            for col in range(2, 6):  # Skip action and status columns
                item = self.comparison_table.item(row, col)
                if item:
                    item.setBackground(bg_color)
                    item.setForeground(text_color)
    
    def on_comparison_row_selected(self):
        """Handle row selection in comparison table."""
        current_row = self.comparison_table.currentRow()
        if current_row < 0:
            return
        
        # Get paper data from ID column (column 2)
        id_item = self.comparison_table.item(current_row, 2)
        if not id_item:
            return
        
        paper_data = id_item.data(Qt.UserRole)
        
        if paper_data:
            paper = paper_data.get('paper', {})
            paper_type = paper_data.get('type', 'duplicate')
            
            # Build details text with good formatting
            details = []
            
            if paper_type == 'original':
                details.append("📌 ORIGINAL PAPER (Will be kept)")
                details.append("")
            else:
                details.append("🔗 DUPLICATE PAPER")
                similarity = paper_data.get('similarity', 0)
                reason = paper_data.get('reason', 'Unknown')
                details.append(f"Similarity: {similarity * 100:.1f}%")
                details.append(f"Match Reason: {reason}")
                details.append("")
            
            details.append(f"Paper ID: {paper.get('id', 'N/A')}")
            details.append(f"Title: {paper.get('title', 'N/A')}")
            details.append(f"Authors: {paper.get('authors', 'N/A')}")
            details.append(f"Year: {paper.get('year', 'N/A')}")
            
            if paper.get('journal'):
                details.append(f"Journal: {paper.get('journal')}")
            
            if paper.get('doi'):
                details.append(f"DOI: {paper.get('doi')}")
            
            if paper_type == 'duplicate':
                details.append("")
                details.append("⚠ Check the DELETE box to remove this duplicate")
            
            self.details_text.setPlainText("\n".join(details))
        else:
            # Fallback to basic info
            id_item = self.comparison_table.item(current_row, 2)
            title_item = self.comparison_table.item(current_row, 3)
            authors_item = self.comparison_table.item(current_row, 4)
            
            if id_item and title_item:
                details = f"Paper ID: {id_item.text()}\n"
                details += f"Title: {title_item.text()}\n"
                if authors_item:
                    details += f"Authors: {authors_item.text()}\n"
                self.details_text.setPlainText(details)
    
    def on_duplicate_checkbox_changed(self, paper_id: int, state: int):
        """Handle checkbox change for duplicate selection."""
        if state == Qt.Checked:
            self.selected_for_deletion.add(paper_id)
        else:
            self.selected_for_deletion.discard(paper_id)
        
        # Update status indicator
        current_item = self.groups_list.currentItem()
        if current_item:
            group_index = current_item.data(Qt.UserRole)
            if group_index is not None:
                self.show_group_comparison(group_index)
        
        self.update_delete_button()
    
    def on_comparison_selection_changed(self, row, column):
        """Handle cell change in comparison table."""
        if column == 0:  # Checkbox column
            checkbox = self.comparison_table.cellWidget(row, 0)
            if checkbox and row > 0:  # Not the original
                paper_id_item = self.comparison_table.item(row, 1)
                if paper_id_item:
                    paper_id = int(paper_id_item.text())
                    if checkbox.isChecked():
                        self.selected_for_deletion.add(paper_id)
                    else:
                        self.selected_for_deletion.discard(paper_id)
                    self.update_delete_button()
    
    def select_all_duplicates(self):
        """Select all duplicate papers for deletion."""
        # Select all duplicates in current group
        current_item = self.groups_list.currentItem()
        if current_item:
            group_index = current_item.data(Qt.UserRole)
            if group_index is not None and group_index < len(self.duplicate_groups):
                group = self.duplicate_groups[group_index]
                for dup_info in group['duplicates']:
                    dup_id = dup_info['paper'].get('id')
                    self.selected_for_deletion.add(dup_id)
                
                # Refresh display
                self.show_group_comparison(group_index)
        
        self.update_delete_button()
    
    def deselect_all_duplicates(self):
        """Deselect all duplicate papers."""
        # Deselect all duplicates in current group
        current_item = self.groups_list.currentItem()
        if current_item:
            group_index = current_item.data(Qt.UserRole)
            if group_index is not None and group_index < len(self.duplicate_groups):
                group = self.duplicate_groups[group_index]
                for dup_info in group['duplicates']:
                    dup_id = dup_info['paper'].get('id')
                    self.selected_for_deletion.discard(dup_id)
                
                # Refresh display
                self.show_group_comparison(group_index)
        
        self.update_delete_button()
    
    def update_delete_button(self):
        """Update delete button state."""
        count = len(self.selected_for_deletion)
        self.delete_button.setEnabled(count > 0)
        if count > 0:
            self.delete_button.setText(f"🗑 Delete Selected ({count})")
            self.delete_button.setToolTip(f"{count} duplicate paper(s) will be permanently deleted")
        else:
            self.delete_button.setText("🗑 Delete Selected (0)")
            self.delete_button.setToolTip("No papers selected for deletion")
    
    def delete_selected(self):
        """Delete selected duplicate papers."""
        if not self.selected_for_deletion:
            QMessageBox.information(self, "No Selection", "Please select papers to delete.")
            return
        
        # Confirmation
        reply = QMessageBox.question(
            self, "Confirm Deletion",
            f"Are you sure you want to delete {len(self.selected_for_deletion)} selected duplicate papers?\n\n"
            "This action cannot be undone!",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        # Delete papers
        progress = QProgressDialog("Deleting papers...", "Cancel", 0, len(self.selected_for_deletion), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        
        deleted_count = 0
        failed_count = 0
        
        for i, paper_id in enumerate(self.selected_for_deletion):
            progress.setValue(i)
            QApplication.processEvents()
            
            try:
                if self.integration_manager.paper_repo.delete_paper(paper_id):
                    deleted_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                logger.error(f"Error deleting paper {paper_id}: {e}")
                failed_count += 1
        
        progress.close()
        
        # Show results
        message = f"Deletion completed:\n\n"
        message += f"Deleted: {deleted_count} papers\n"
        message += f"Failed: {failed_count} papers"
        
        if failed_count > 0:
            QMessageBox.warning(self, "Deletion Complete", message)
        else:
            QMessageBox.information(self, "Deletion Complete", message)
        
        # Refresh
        self.start_detection()



