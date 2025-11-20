"""
Smart Verification Dialog
Provides optimized verification with selective processing and caching.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QTableWidget, QTableWidgetItem, QProgressBar, QTextEdit,
    QSplitter, QGroupBox, QCheckBox, QMessageBox, QHeaderView,
    QAbstractItemView, QFrame, QComboBox, QSpinBox, QGroupBox,
    QRadioButton, QButtonGroup
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QFont, QColor, QPalette

from ..utils.post_import_verifier import PostImportVerifier, VerificationResult, VerificationStatus

logger = logging.getLogger(__name__)


class SmartVerificationWorker(QThread):
    """Optimized worker thread for verification process."""
    
    progress_updated = Signal(int, int)  # current, total
    paper_verified = Signal(VerificationResult)
    verification_complete = Signal(list)  # List[VerificationResult]
    error_occurred = Signal(str)
    
    def __init__(self, papers: List[Dict[str, Any]], verifier: PostImportVerifier, 
                 verification_mode: str = "unverified_only", max_papers: int = 50):
        super().__init__()
        self.papers = papers
        self.verifier = verifier
        self.verification_mode = verification_mode
        self.max_papers = max_papers
        self.should_stop = False
    
    def run(self):
        """Run optimized verification process."""
        try:
            # Filter papers based on verification mode
            papers_to_verify = self._filter_papers_for_verification()

            # Heuristic: disable Google Scholar for larger batches to avoid
            # repeated errors and rate limiting. It remains enabled for small
            # batches (e.g. quick manual checks).
            if hasattr(self.verifier, "enable_google_scholar"):
                if len(papers_to_verify) > 5:
                    logger.info(
                        "Disabling Google Scholar for batch verification of %d papers",
                        len(papers_to_verify),
                    )
                    self.verifier.enable_google_scholar = False
                else:
                    self.verifier.enable_google_scholar = True
            
            if not papers_to_verify:
                self.verification_complete.emit([])
                return
            
            # Limit number of papers to prevent long delays
            if len(papers_to_verify) > self.max_papers:
                papers_to_verify = papers_to_verify[:self.max_papers]
            
            total_papers = len(papers_to_verify)
            results = []
            
            for i, paper in enumerate(papers_to_verify):
                if self.should_stop:
                    break
                
                try:
                    result = self.verifier.verify_paper(paper)
                    logger.info(f"Verification result for paper {paper.get('id', 'unknown')}: status={result.status}, confidence={result.confidence_score}, has_metadata={bool(result.verified_metadata)}")
                    results.append(result)
                    self.paper_verified.emit(result)
                    self.progress_updated.emit(i + 1, total_papers)
                except Exception as e:
                    logger.error(f"Error verifying paper {paper.get('id', 'unknown')}: {e}")
                    # Create a failed result
                    failed_result = VerificationResult(
                        paper_id=paper.get('id', 0),
                        status=VerificationStatus.FAILED,
                        method_used="error",
                        confidence_score=0.0,
                        verified_metadata={},
                        errors=[str(e)],
                        suggestions=[]
                    )
                    results.append(failed_result)
                    self.paper_verified.emit(failed_result)
                    self.progress_updated.emit(i + 1, total_papers)
            
            self.verification_complete.emit(results)
            
        except Exception as e:
            logger.error(f"Smart verification worker error: {e}")
            self.error_occurred.emit(str(e))
    
    def _filter_papers_for_verification(self) -> List[Dict[str, Any]]:
        """Filter papers based on verification mode."""
        if self.verification_mode == "all":
            return self.papers
        elif self.verification_mode == "unverified_only":
            return [p for p in self.papers if p.get('verification_status') in [None, 'pending', 'failed']]
        elif self.verification_mode == "outdated":
            # Verify papers that haven't been verified in the last 30 days
            cutoff_date = datetime.now() - timedelta(days=30)
            filtered_papers = []
            for p in self.papers:
                verification_date = p.get('verification_date')
                if verification_date is None:
                    filtered_papers.append(p)
                else:
                    # Handle string dates from database
                    if isinstance(verification_date, str):
                        try:
                            verification_date = datetime.fromisoformat(verification_date.replace('Z', '+00:00'))
                        except:
                            filtered_papers.append(p)
                            continue
                    if verification_date < cutoff_date:
                        filtered_papers.append(p)
            return filtered_papers
        elif self.verification_mode == "low_confidence":
            return [p for p in self.papers if 
                   (p.get('verification_confidence') or 0) < 0.7]
        else:
            return self.papers
    
    def stop(self):
        """Stop the verification process."""
        self.should_stop = True


class SmartVerificationDialog(QDialog):
    """Optimized dialog for paper verification with smart filtering."""
    
    data_updated = Signal()  # Signal emitted when data is updated
    
    def __init__(self, papers: List[Dict[str, Any]], parent=None):
        super().__init__(parent)
        self.papers = papers
        self.verifier = PostImportVerifier()
        self.verification_results = []
        self.paper_repo = None  # Will be set by parent
        
        self.setup_ui()
        self.setup_connections()
        
        # Analyze papers and show statistics
        self.analyze_papers()
    
    def refresh_paper_data(self):
        """Refresh paper data from database."""
        if self.paper_repo:
            try:
                self.papers = self.paper_repo.list_all()
                self.analyze_papers()
            except Exception as e:
                logger.error(f"Error refreshing paper data: {e}")
    
    def setup_ui(self):
        """Setup the user interface."""
        self.setWindowTitle("Smart Paper Verification")
        self.setModal(True)
        self.resize(1200, 800)
        
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Header with verification options
        header_group = QGroupBox("Verification Options")
        header_layout = QVBoxLayout(header_group)
        
        # Verification mode selection
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Verification Mode:"))
        
        self.mode_group = QButtonGroup()
        self.mode_all = QRadioButton("All Papers")
        self.mode_unverified = QRadioButton("Unverified Only")
        self.mode_outdated = QRadioButton("Outdated (30+ days)")
        self.mode_low_confidence = QRadioButton("Low Confidence (<70%)")
        
        self.mode_unverified.setChecked(True)  # Default to unverified only
        
        self.mode_group.addButton(self.mode_all, 0)
        self.mode_group.addButton(self.mode_unverified, 1)
        self.mode_group.addButton(self.mode_outdated, 2)
        self.mode_group.addButton(self.mode_low_confidence, 3)
        
        mode_layout.addWidget(self.mode_all)
        mode_layout.addWidget(self.mode_unverified)
        mode_layout.addWidget(self.mode_outdated)
        mode_layout.addWidget(self.mode_low_confidence)
        mode_layout.addStretch()
        
        # Max papers limit
        limit_layout = QHBoxLayout()
        limit_layout.addWidget(QLabel("Max Papers to Verify:"))
        self.max_papers_spin = QSpinBox()
        self.max_papers_spin.setRange(1, 1000)
        self.max_papers_spin.setValue(50)
        self.max_papers_spin.setSuffix(" papers")
        limit_layout.addWidget(self.max_papers_spin)
        limit_layout.addStretch()
        
        # Statistics
        self.stats_label = QLabel("Analyzing papers...")
        self.stats_label.setStyleSheet("font-weight: bold; color: #666;")
        
        header_layout.addLayout(mode_layout)
        header_layout.addLayout(limit_layout)
        header_layout.addWidget(self.stats_label)
        
        main_layout.addWidget(header_group)
        
        # Progress section
        progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_label = QLabel("Ready to verify papers")
        progress_layout.addWidget(self.status_label)
        progress_layout.addWidget(self.progress_bar)
        main_layout.addLayout(progress_layout)
        
        # Splitter for main content
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Results table
        left_panel = self.create_results_panel()
        splitter.addWidget(left_panel)
        
        # Right panel - Details
        right_panel = self.create_details_panel()
        splitter.addWidget(right_panel)
        
        # Set splitter proportions
        splitter.setSizes([800, 400])
        main_layout.addWidget(splitter)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start Verification")
        self.start_button.clicked.connect(self.start_verification)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_verification)
        
        self.apply_button = QPushButton("Apply Selected Updates")
        self.apply_button.setEnabled(False)
        self.apply_button.clicked.connect(self.apply_updates)
        
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)
        
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addStretch()
        button_layout.addWidget(self.apply_button)
        button_layout.addWidget(self.close_button)
        main_layout.addLayout(button_layout)
    
    def create_results_panel(self):
        """Create the results table panel."""
        group = QGroupBox("Verification Results")
        layout = QVBoxLayout(group)
        
        # Results table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(8)
        self.results_table.setHorizontalHeaderLabels([
            "Select", "Title", "Status", "Method", "Confidence", "Errors", "Suggestions", "Last Verified"
        ])
        
        # Configure table
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self.results_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.results_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.results_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.results_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.results_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.results_table.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeToContents)
        self.results_table.horizontalHeader().setSectionResizeMode(7, QHeaderView.ResizeToContents)
        
        self.results_table.setColumnWidth(0, 60)
        
        # Connect selection change
        self.results_table.selectionModel().selectionChanged.connect(self.on_selection_changed)
        
        layout.addWidget(self.results_table)
        
        # Summary
        self.summary_label = QLabel("No results yet...")
        self.summary_label.setStyleSheet("font-weight: bold; color: #666;")
        layout.addWidget(self.summary_label)
        
        return group
    
    def create_details_panel(self):
        """Create the details panel."""
        group = QGroupBox("Details")
        layout = QVBoxLayout(group)
        
        # Details text area
        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setFont(QFont("Consolas", 9))
        layout.addWidget(self.details_text)
        
        # Action buttons
        action_layout = QHBoxLayout()
        
        self.view_original_button = QPushButton("View Original")
        self.view_original_button.setEnabled(False)
        self.view_original_button.clicked.connect(self.view_original_paper)
        
        self.view_verified_button = QPushButton("View Verified")
        self.view_verified_button.setEnabled(False)
        self.view_verified_button.clicked.connect(self.view_verified_metadata)
        
        action_layout.addWidget(self.view_original_button)
        action_layout.addWidget(self.view_verified_button)
        action_layout.addStretch()
        
        layout.addLayout(action_layout)
        
        return group
    
    def setup_connections(self):
        """Setup signal connections."""
        pass  # Connections will be set up when worker is created
    
    def analyze_papers(self):
        """Analyze papers and show statistics."""
        total_papers = len(self.papers)
        
        # Count verification statuses
        unverified = 0
        verified = 0
        partial = 0
        failed = 0
        low_confidence = 0
        
        for p in self.papers:
            status = p.get('verification_status', 'pending')
            confidence = p.get('verification_confidence', 0) or 0
            
            if status in [None, 'pending', 'failed']:
                unverified += 1
            elif status == 'verified':
                verified += 1
            elif status == 'partial':
                partial += 1
            elif status == 'failed':
                failed += 1
            
            if confidence < 0.7:
                low_confidence += 1
        
        # Count outdated papers (not verified in last 30 days)
        cutoff_date = datetime.now() - timedelta(days=30)
        outdated = 0
        for p in self.papers:
            verification_date = p.get('verification_date')
            if verification_date is None:
                outdated += 1
            else:
                # Handle string dates from database
                if isinstance(verification_date, str):
                    try:
                        verification_date = datetime.fromisoformat(verification_date.replace('Z', '+00:00'))
                    except:
                        outdated += 1
                        continue
                if verification_date < cutoff_date:
                    outdated += 1
        
        stats_text = f"Total: {total_papers} | Unverified: {unverified} | Verified: {verified} | Partial: {partial} | Failed: {failed} | Low Confidence: {low_confidence} | Outdated: {outdated}"
        self.stats_label.setText(stats_text)
    
    def start_verification(self):
        """Start the verification process."""
        if not self.papers:
            self.status_label.setText("No papers to verify.")
            return
        
        # Get verification mode
        mode_map = {0: "all", 1: "unverified_only", 2: "outdated", 3: "low_confidence"}
        verification_mode = mode_map[self.mode_group.checkedId()]
        max_papers = self.max_papers_spin.value()
        
        # Create and start worker
        self.worker = SmartVerificationWorker(self.papers, self.verifier, verification_mode, max_papers)
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.paper_verified.connect(self.add_verification_result)
        self.worker.verification_complete.connect(self.verification_finished)
        self.worker.error_occurred.connect(self.handle_error)
        
        self.worker.start()
        
        # Update UI
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.status_label.setText(f"Starting verification in {verification_mode} mode...")
    
    def stop_verification(self):
        """Stop the verification process."""
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(3000)  # Wait up to 3 seconds
            self.status_label.setText("Verification stopped by user")
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
    
    def update_progress(self, current: int, total: int):
        """Update progress bar and status."""
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.status_label.setText(f"Verifying paper {current} of {total}...")
    
    def add_verification_result(self, result: VerificationResult):
        """Add a verification result to the table."""
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)
        
        # Checkbox for selection
        checkbox = QCheckBox()
        checkbox.setChecked(False)
        self.results_table.setCellWidget(row, 0, checkbox)
        
        # Title (truncated)
        title = result.verified_metadata.get('title', 'Unknown')
        if len(title) > 50:
            title = title[:47] + "..."
        self.results_table.setItem(row, 1, QTableWidgetItem(title))
        
        # Status
        status_item = QTableWidgetItem(result.status.value.title())
        status_item.setTextAlignment(Qt.AlignCenter)
        
        # Color code status
        if result.status == VerificationStatus.VERIFIED:
            status_item.setBackground(QColor(200, 255, 200))  # Light green
        elif result.status == VerificationStatus.PARTIAL:
            status_item.setBackground(QColor(255, 255, 200))  # Light yellow
        else:
            status_item.setBackground(QColor(255, 200, 200))  # Light red
        
        self.results_table.setItem(row, 2, status_item)
        
        # Method
        self.results_table.setItem(row, 3, QTableWidgetItem(result.method_used.title()))
        
        # Confidence
        confidence_text = f"{result.confidence_score:.2f}"
        confidence_item = QTableWidgetItem(confidence_text)
        confidence_item.setTextAlignment(Qt.AlignCenter)
        self.results_table.setItem(row, 4, confidence_item)
        
        # Errors
        error_text = "; ".join(result.errors) if result.errors else "None"
        if len(error_text) > 30:
            error_text = error_text[:27] + "..."
        self.results_table.setItem(row, 5, QTableWidgetItem(error_text))
        
        # Suggestions
        suggestion_text = "; ".join(result.suggestions) if result.suggestions else "None"
        if len(suggestion_text) > 30:
            suggestion_text = suggestion_text[:27] + "..."
        self.results_table.setItem(row, 6, QTableWidgetItem(suggestion_text))
        
        # Last verified
        last_verified = "Now" if result.status != VerificationStatus.FAILED else "Never"
        self.results_table.setItem(row, 7, QTableWidgetItem(last_verified))
        
        # Store result for later use
        self.results_table.item(row, 1).setData(Qt.UserRole, result)
    
    def verification_finished(self, results: List[VerificationResult]):
        """Handle verification completion."""
        logger.info(f"Verification finished with {len(results)} results")
        self.verification_results = results
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Verification complete: {len(results)} papers processed")
        
        # Update summary
        verified_count = sum(1 for r in results if r.status == VerificationStatus.VERIFIED)
        partial_count = sum(1 for r in results if r.status == VerificationStatus.PARTIAL)
        failed_count = sum(1 for r in results if r.status == VerificationStatus.FAILED)
        
        summary_text = f"Verified: {verified_count}, Partial: {partial_count}, Failed: {failed_count}"
        self.summary_label.setText(summary_text)
        
        # Auto-apply verification results if we have a repository
        if self.paper_repo and results:
            logger.info(f"Auto-applying {len(results)} verification results")
            self._auto_apply_verification_results(results)
        else:
            logger.warning(f"No repository or no results: repo={bool(self.paper_repo)}, results={len(results)}")
        
        # Enable apply button if there are results to apply
        if any(r.status in [VerificationStatus.VERIFIED, VerificationStatus.PARTIAL] for r in results):
            self.apply_button.setEnabled(True)
        
        # Reset buttons
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
    
    def _auto_apply_verification_results(self, results: List[VerificationResult]):
        """Automatically apply verification results to the database."""
        try:
            logger.info(f"Auto-applying {len(results)} verification results")
            updated_count = 0
            for result in results:
                logger.info(f"Processing result for paper {result.paper_id}: status={result.status}, has_metadata={bool(result.verified_metadata)}")
                logger.info(f"Verified metadata content: {result.verified_metadata}")
                # Apply all verification results, even if they don't have verified_metadata
                if result.status in [VerificationStatus.VERIFIED, VerificationStatus.PARTIAL]:
                    logger.info(f"Updating paper {result.paper_id} with status {result.status.value}")
                    # Use empty dict if no verified_metadata
                    metadata_to_apply = result.verified_metadata if result.verified_metadata else {}
                    logger.info(f"Metadata to apply: {metadata_to_apply}")
                    if self.paper_repo.update_verification_status(
                        result.paper_id, 
                        result.status.value, 
                        result.method_used, 
                        result.confidence_score, 
                        metadata_to_apply
                    ):
                        updated_count += 1
                        logger.info(f"Successfully updated paper {result.paper_id}")
                    else:
                        logger.error(f"Failed to update paper {result.paper_id}")
                else:
                    logger.warning(f"Skipping paper {result.paper_id}: status={result.status}")
            
            if updated_count > 0:
                self.status_label.setText(f"Auto-applied {updated_count} verification results to database")
                # Emit signal to refresh parent
                self.data_updated.emit()
                # Refresh paper data from database
                self.refresh_paper_data()
            else:
                logger.warning("No verification results were applied")
                self.status_label.setText("No verification results to apply")
            
        except Exception as e:
            logger.error(f"Error auto-applying verification results: {e}")
            self.status_label.setText(f"Auto-apply failed: {str(e)}")
    
    def handle_error(self, error_message: str):
        """Handle verification errors."""
        self.status_label.setText(f"Error: {error_message}")
        QMessageBox.critical(self, "Verification Error", f"An error occurred during verification:\n{error_message}")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
    
    def on_selection_changed(self):
        """Handle table selection change."""
        current_row = self.results_table.currentRow()
        if current_row >= 0:
            item = self.results_table.item(current_row, 1)
            if item:
                result = item.data(Qt.UserRole)
                if result:
                    self.show_result_details(result)
                    self.view_original_button.setEnabled(True)
                    self.view_verified_button.setEnabled(bool(result.verified_metadata))
    
    def show_result_details(self, result: VerificationResult):
        """Show detailed information about a verification result."""
        details = []
        details.append(f"Paper ID: {result.paper_id}")
        details.append(f"Status: {result.status.value.title()}")
        details.append(f"Method: {result.method_used.title()}")
        details.append(f"Confidence: {result.confidence_score:.2f}")
        details.append("")
        
        if result.errors:
            details.append("Errors:")
            for error in result.errors:
                details.append(f"  • {error}")
            details.append("")
        
        if result.suggestions:
            details.append("Suggestions:")
            for suggestion in result.suggestions:
                details.append(f"  • {suggestion}")
            details.append("")
        
        if result.verified_metadata:
            details.append("Verified Metadata:")
            for key, value in result.verified_metadata.items():
                if value:
                    details.append(f"  {key}: {value}")
        
        self.details_text.setPlainText("\n".join(details))
    
    def view_original_paper(self):
        """View original paper data."""
        current_row = self.results_table.currentRow()
        if current_row >= 0:
            item = self.results_table.item(current_row, 1)
            if item:
                result = item.data(Qt.UserRole)
                if result:
                    # Find original paper
                    original_paper = next((p for p in self.papers if p.get('id') == result.paper_id), None)
                    if original_paper:
                        self.show_paper_data("Original Paper", original_paper)
    
    def view_verified_metadata(self):
        """View verified metadata."""
        current_row = self.results_table.currentRow()
        if current_row >= 0:
            item = self.results_table.item(current_row, 1)
            if item:
                result = item.data(Qt.UserRole)
                if result and result.verified_metadata:
                    self.show_paper_data("Verified Metadata", result.verified_metadata)
    
    def show_paper_data(self, title: str, data: Dict[str, Any]):
        """Show paper data in a dialog."""
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.setModal(True)
        dialog.resize(600, 400)
        
        layout = QVBoxLayout(dialog)
        
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setFont(QFont("Consolas", 9))
        
        # Format data nicely
        formatted_data = []
        for key, value in data.items():
            if value:
                formatted_data.append(f"{key}: {value}")
        
        text_edit.setPlainText("\n".join(formatted_data))
        layout.addWidget(text_edit)
        
        # Close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button)
        
        dialog.exec()
    
    def apply_updates(self):
        """Apply selected updates to the database."""
        if not self.paper_repo:
            QMessageBox.warning(self, "No Repository", "No repository available for updates.")
            return
        
        # Get selected results
        selected_results = []
        for row in range(self.results_table.rowCount()):
            checkbox = self.results_table.cellWidget(row, 0)
            if checkbox and checkbox.isChecked():
                item = self.results_table.item(row, 1)
                if item:
                    result = item.data(Qt.UserRole)
                    if result and result.verified_metadata:
                        selected_results.append(result)
        
        if not selected_results:
            QMessageBox.information(self, "No Selection", "Please select papers to update.")
            return
        
        # Apply updates
        try:
            updated_count = 0
            for result in selected_results:
                if self.paper_repo.update_verification_status(
                    result.paper_id, 
                    result.status.value, 
                    result.method_used, 
                    result.confidence_score, 
                    result.verified_metadata
                ):
                    updated_count += 1
            
            QMessageBox.information(
                self, "Updates Applied", 
                f"Successfully updated {updated_count} out of {len(selected_results)} selected papers."
            )
            
            # Emit signal to refresh parent
            self.data_updated.emit()
            
            # Refresh paper data from database
            self.refresh_paper_data()
        except Exception as e:
            QMessageBox.critical(self, "Update Error", f"Error applying updates: {str(e)}")
            logger.error(f"Error applying updates: {e}")
    
    def closeEvent(self, event):
        """Handle dialog close event."""
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(3000)  # Wait up to 3 seconds
        event.accept()
