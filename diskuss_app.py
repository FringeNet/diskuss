# Main Diskuss application file

# import tkinter as tk
# from tkinter import filedialog, scrolledtext, ttk
import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QComboBox, QLineEdit, QTextEdit,
                             QFileDialog, QSpinBox, QFrame, QProgressBar) # Import PyQt6 widgets
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QObject, QTimer # Import PyQt6 core components and QTimer
from PyQt6.QtGui import QTextCursor, QColor # Added QColor for status label

import os
import threading # Keep threading for now, might refactor later to QThread if needed for all background tasks
import queue
import numpy as np # Import numpy

from ollama_integration import is_ollama_running, list_ollama_models, generate_embedding, chat_with_ollama
from file_processor import process_file
from database import initialize_database, add_document, search_documents, DATABASE_FILE, count_total_documents # Import count_total_documents

class DiskussApp(QMainWindow):
    # Signal for updating chat history from other threads
    update_chat_signal = pyqtSignal(str)
    # Signal for updating progress label from other threads
    update_progress_signal = pyqtSignal(int, int) # count, total
    update_ollama_status_signal = pyqtSignal(bool) # Signal for Ollama status
    set_index_button_enabled_signal = pyqtSignal(bool) # New signal for index button
    set_send_button_enabled_signal = pyqtSignal(bool) # New signal for send button
    update_indexed_docs_count_signal = pyqtSignal(int) # New signal for indexed docs count

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Diskuss")
        self.setGeometry(100, 100, 800, 700) # Adjusted height for better layout

        self.embedding_model = None
        self.chat_model = None
        self.indexing_queue = queue.Queue()
        self.processing_thread = None
        self.discovery_thread = None # Keep track of discovery thread
        self.stop_indexing = threading.Event()
        self.top_k_documents_value = 5 # Default k value

        # --- Main Widget and Layout ---
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        self._init_ui()

        initialize_database()
        # self.check_ollama_status() # Initial check, will emit signal
        QTimer.singleShot(0, self.check_ollama_status) # Use QTimer for initial check
        QTimer.singleShot(0, self.refresh_indexed_docs_count) # Initial refresh of doc count

        # Connect signals for thread-safe UI updates
        self.update_chat_signal.connect(self._insert_chat_message)
        self.update_progress_signal.connect(self._update_progress_label_slot)
        self.update_ollama_status_signal.connect(self._update_ollama_ui_status)
        self.set_index_button_enabled_signal.connect(self._set_index_button_enabled) # Connect new signal
        self.set_send_button_enabled_signal.connect(self._set_send_button_enabled) # Connect new signal
        self.update_indexed_docs_count_signal.connect(self._update_indexed_docs_count_label) # Connect new signal

    def _init_ui(self):
        # --- Ollama Status ---
        status_frame = QFrame()
        status_frame.setFrameShape(QFrame.Shape.StyledPanel)
        status_layout = QHBoxLayout(status_frame)
        self.ollama_status_label = QLabel("Ollama Status: Checking...")
        status_layout.addWidget(self.ollama_status_label)
        self.check_ollama_button = QPushButton("Check Ollama")
        self.check_ollama_button.clicked.connect(self.check_ollama_status)
        status_layout.addWidget(self.check_ollama_button)
        status_layout.addStretch(1) # Push elements to the left
        self.main_layout.addWidget(status_frame)

        # --- Model Selection ---
        model_group_box = QFrame()
        model_group_box.setFrameShape(QFrame.Shape.StyledPanel)
        model_group_box_layout = QVBoxLayout(model_group_box) # Main layout for this group
        model_group_box_layout.addWidget(QLabel("Model Selection"))

        model_selection_layout = QHBoxLayout() # Layout for the comboboxes and labels
        model_selection_layout.addWidget(QLabel("Embedding Model:"))
        self.embedding_model_combobox = QComboBox()
        self.embedding_model_combobox.setDisabled(True)
        self.embedding_model_combobox.currentTextChanged.connect(self.set_embedding_model)
        model_selection_layout.addWidget(self.embedding_model_combobox, 1) # Add stretch factor

        model_selection_layout.addWidget(QLabel("Chat Model:"))
        self.chat_model_combobox = QComboBox()
        self.chat_model_combobox.setDisabled(True)
        self.chat_model_combobox.currentTextChanged.connect(self.set_chat_model)
        model_selection_layout.addWidget(self.chat_model_combobox, 1) # Add stretch factor
        
        model_group_box_layout.addLayout(model_selection_layout)
        self.main_layout.addWidget(model_group_box)
        self.load_models()


        # --- Indexing ---
        indexing_group_box = QFrame()
        indexing_group_box.setFrameShape(QFrame.Shape.StyledPanel)
        indexing_group_box_layout = QVBoxLayout(indexing_group_box)
        indexing_group_box_layout.addWidget(QLabel("Document Indexing"))

        path_layout = QHBoxLayout()
        self.path_entry = QLineEdit()
        self.path_entry.setPlaceholderText("Enter file or directory path")
        path_layout.addWidget(self.path_entry, 1)
        self.browse_file_button = QPushButton("Browse File")
        self.browse_file_button.clicked.connect(self.browse_file)
        path_layout.addWidget(self.browse_file_button)
        self.browse_dir_button = QPushButton("Browse Directory")
        self.browse_dir_button.clicked.connect(self.browse_directory)
        path_layout.addWidget(self.browse_dir_button)
        indexing_group_box_layout.addLayout(path_layout)

        indexing_controls_layout = QHBoxLayout()
        self.index_button = QPushButton("Start Indexing")
        self.index_button.clicked.connect(self.start_indexing)
        self.index_button.setDisabled(True)
        indexing_controls_layout.addWidget(self.index_button)
        self.progress_label = QLabel("Progress: Idle")
        indexing_controls_layout.addWidget(self.progress_label, 1) # Allow progress label to expand
        self.indexed_docs_count_label = QLabel("Total Documents Indexed: 0") # New label
        indexing_controls_layout.addWidget(self.indexed_docs_count_label) # Add to layout
        indexing_group_box_layout.addLayout(indexing_controls_layout)
        self.main_layout.addWidget(indexing_group_box)

        # --- Chat ---
        chat_group_box = QFrame()
        chat_group_box.setFrameShape(QFrame.Shape.StyledPanel)
        chat_group_box_layout = QVBoxLayout(chat_group_box)
        chat_group_box_layout.addWidget(QLabel("Chat"))

        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        chat_group_box_layout.addWidget(self.chat_history, 1) # Make chat history expand

        query_controls_layout = QHBoxLayout()
        self.query_entry = QLineEdit()
        self.query_entry.setPlaceholderText("Type your query here and press Enter")
        self.query_entry.returnPressed.connect(self.send_query)
        query_controls_layout.addWidget(self.query_entry, 1)

        query_controls_layout.addWidget(QLabel("Top K:"))
        self.top_k_spinbox = QSpinBox()
        self.top_k_spinbox.setMinimum(1)
        self.top_k_spinbox.setMaximum(20)
        self.top_k_spinbox.setValue(self.top_k_documents_value)
        self.top_k_spinbox.valueChanged.connect(self.set_top_k_value)
        query_controls_layout.addWidget(self.top_k_spinbox)

        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_query)
        self.send_button.setDisabled(True)
        query_controls_layout.addWidget(self.send_button)

        chat_group_box_layout.addLayout(query_controls_layout)
        self.main_layout.addWidget(chat_group_box, 1) # Make chat group box expand

        # Adjust overall layout stretching
        self.main_layout.setStretch(self.main_layout.indexOf(status_frame), 0)
        self.main_layout.setStretch(self.main_layout.indexOf(model_group_box), 0)
        self.main_layout.setStretch(self.main_layout.indexOf(indexing_group_box), 0)
        self.main_layout.setStretch(self.main_layout.indexOf(chat_group_box), 1)
    
    def check_ollama_status(self):
        """Checks Ollama status and emits a signal to update UI (thread-safe)."""
        # This can be run in a separate thread if it becomes blocking
        running = is_ollama_running()
        self.update_ollama_status_signal.emit(running)
        self.refresh_indexed_docs_count() # Refresh count after indexing

    def _update_ollama_ui_status(self, is_running):
        """Updates UI elements based on Ollama status. Connected to a signal."""
        if is_running:
            self.ollama_status_label.setText("Ollama Status: <font color='green'>Running</font>")
            self.embedding_model_combobox.setDisabled(False)
            self.chat_model_combobox.setDisabled(False)
            self.index_button.setDisabled(False)
            self.send_button.setDisabled(False)
        else:
            self.ollama_status_label.setText("Ollama Status: <font color='red'>Not Running. Please start Ollama.</font>")
            self.embedding_model_combobox.setDisabled(True)
            self.chat_model_combobox.setDisabled(True)
            self.index_button.setDisabled(True)
            self.send_button.setDisabled(True)
        self.load_models() # Reload models based on status

    def load_models(self):
        """Loads available Ollama models into the comboboxes."""
        # Clear previous items, important if called multiple times
        self.embedding_model_combobox.clear()
        self.chat_model_combobox.clear()

        if self.embedding_model_combobox.isEnabled(): # Only load if comboboxes are enabled
            models = list_ollama_models()
            if models:
                self.embedding_model_combobox.addItems(models)
                self.chat_model_combobox.addItems(models)
                
                # Attempt to select default models if available
                default_embed_model_set = False
                for model in models:
                    if 'embed' in model.lower() or 'mxbai' in model.lower(): # More robust check
                        self.embedding_model_combobox.setCurrentText(model)
                        self.embedding_model = model
                        default_embed_model_set = True
                        break
                if not default_embed_model_set and models: # Fallback to first if no embed found
                    self.embedding_model_combobox.setCurrentIndex(0)
                    self.embedding_model = self.embedding_model_combobox.currentText()

                default_chat_model_set = False
                for model in models:
                    if any(keyword in model.lower() for keyword in ['llama', 'mistral', 'phi', 'qwen', 'gemma']): # Common chat keywords
                        self.chat_model_combobox.setCurrentText(model)
                        self.chat_model = model
                        default_chat_model_set = True
                        break
                if not default_chat_model_set and models: # Fallback to first if no chat model found
                    self.chat_model_combobox.setCurrentIndex(0)
                    self.chat_model = self.chat_model_combobox.currentText()
            else:
                self.embedding_model_combobox.addItem("No models found")
                self.chat_model_combobox.addItem("No models found")
        else:
            self.embedding_model_combobox.addItem("Ollama not running")
            self.chat_model_combobox.addItem("Ollama not running")

    def set_embedding_model(self, text):
        if text and text != "No models found" and text != "Ollama not running":
            self.embedding_model = text
            self.update_chat_signal.emit(f"Embedding model set to: {self.embedding_model}")

    def set_chat_model(self, text):
        if text and text != "No models found" and text != "Ollama not running":
            self.chat_model = text
            self.update_chat_signal.emit(f"Chat model set to: {self.chat_model}")

    def browse_file(self):
        """Opens a file dialog to select a file for indexing."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", "Document Files (*.docx *.pdf *.txt *.csv);;All Files (*.*)")
        if file_path:
            self.path_entry.setText(file_path)

    def browse_directory(self):
        """Opens a directory dialog to select a folder for indexing."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if dir_path:
            self.path_entry.setText(dir_path)

    def start_indexing(self):
        """Starts the file/directory indexing process in a separate thread."""
        path = self.path_entry.text()
        if not path:
            self.update_chat_signal.emit("Please select a file or directory to index.")
            return

        if not self.embedding_model or self.embedding_model == "No models found" or self.embedding_model == "Ollama not running":
            self.update_chat_signal.emit("Please select a valid embedding model.")
            return

        self.index_button.setDisabled(True)
        self.progress_label.setText("Progress: Preparing for indexing...")
        # self.progress_bar.setVisible(True)
        # self.progress_bar.setRange(0,0) # Indeterminate progress

        self.stop_indexing.clear()
        self.indexing_queue = queue.Queue()

        if self.discovery_thread and self.discovery_thread.is_alive():
             self.update_chat_signal.emit("A discovery process is already running.")
             self.index_button.setDisabled(False) # Re-enable button
             return

        self.discovery_thread = threading.Thread(target=self._discover_files, args=(path,))
        self.discovery_thread.daemon = True # Allow main program to exit even if thread is running
        self.discovery_thread.start()

        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.processing_thread = threading.Thread(target=self._process_indexing_queue)
            self.processing_thread.daemon = True # Allow main program to exit
            self.processing_thread.start()
        else:
            self.update_chat_signal.emit("Indexing process is already active or starting.")
            # No need to start another processing thread if one is already set up to listen to the queue

    def _discover_files(self, path):
        """Recursively finds supported files in a directory and puts them in the queue."""
        threading.current_thread().name = '_discover_files_thread' # Name the thread
        supported_extensions = ['.docx', '.pdf', '.txt', '.csv'] # Add other extensions here
        files_found_count = 0
        initial_discovery_done = False

        try:
            if os.path.isfile(path):
                if os.path.splitext(path)[1].lower() in supported_extensions:
                    self.indexing_queue.put(path)
                    files_found_count = 1
                else:
                    self.update_chat_signal.emit(f"Unsupported file type for indexing: {os.path.basename(path)}")
                    # No need to emit progress signal here as it's handled in finally
            elif os.path.isdir(path):
                for root_dir, _, files in os.walk(path):
                    if self.stop_indexing.is_set():
                        self.update_chat_signal.emit("File discovery stopped by user.")
                        break
                    for file in files:
                        if self.stop_indexing.is_set(): break
                        file_path = os.path.join(root_dir, file)
                        if os.path.splitext(file_path)[1].lower() in supported_extensions:
                            self.indexing_queue.put(file_path)
                            files_found_count += 1
                            # Emit progress for responsive UI during long discovery
                            if files_found_count % 10 == 0: # Update every 10 files
                                self.update_progress_signal.emit(0, files_found_count) 
                    if self.stop_indexing.is_set(): break
            else:
                self.update_chat_signal.emit(f"Invalid path: {path}")
        except Exception as e:
            self.update_chat_signal.emit(f"Error during file discovery: {e}")
        finally:
            initial_discovery_done = True
            self.indexing_queue.put(None) # Sentinel value
            if files_found_count > 0:
                self.update_chat_signal.emit(f"Found {files_found_count} supported files. Indexing started...")
                self.update_progress_signal.emit(0, files_found_count) # Final count for total
            else:
                self.update_chat_signal.emit("No supported files found in the selected path.")
                self.update_progress_signal.emit(0, 0) # Reset progress
            
            # Re-enable indexing button only if not stopped by user
            if not self.stop_indexing.is_set():
                 # Ensure UI update is on main thread if this was a quick sync operation for some reason
                # QApplication.instance().postEvent(self, threading.Event()) # Dummy event to trigger main thread processing if needed
                # self.index_button.setDisabled(False) # This should be safe if called from main thread, but consider signal if issues
                self.set_index_button_enabled_signal.emit(True) # Use signal
                # If self.progress_label is directly manipulated here, ensure it's via signal or main thread execution
                self.update_progress_signal.emit(0,0) # Reset to idle after discovery (if no files or error)

    def _process_indexing_queue(self):
        """Processes files from the queue (run in a separate thread)."""
        threading.current_thread().name = '_process_indexing_thread'
        indexed_count = 0
        total_files_to_process = 0 
        first_item = True

        while True:
            if self.stop_indexing.is_set():
                self.update_chat_signal.emit("Indexing stopped by user.")
                break
            try:
                file_path = self.indexing_queue.get(timeout=1)
                if file_path is None: # Sentinel
                    self.indexing_queue.task_done()
                    if first_item: # Edge case: Sentinel was the very first item (empty directory)
                        total_files_to_process = 0
                    break
                
                first_item = False
                # Estimate total files only once, after the first real item is dequeued
                # This is an approximation as discovery might still be adding files
                if indexed_count == 0:
                    # qsize can be unreliable, so we rely on the count from _discover_files for the total
                    # The total is passed via update_progress_signal from _discover_files
                    pass # Total is now set by _discover_files

                self.update_chat_signal.emit(f"Processing: {os.path.basename(file_path)}")
                text = process_file(file_path)

                if text:
                    embedding_list = generate_embedding(self.embedding_model, text)
                    if embedding_list is not None:
                        embedding_np = np.array(embedding_list, dtype=np.float32)
                        add_document(DATABASE_FILE, file_path, embedding_np)
                        indexed_count += 1
                        self.update_chat_signal.emit(f"Indexed: {os.path.basename(file_path)}")
                    else:
                        self.update_chat_signal.emit(f"Failed to generate embedding for {os.path.basename(file_path)}")
                else:
                    self.update_chat_signal.emit(f"Failed to extract text from {os.path.basename(file_path)}")
                
                self.indexing_queue.task_done()
                # The total in update_progress_signal is now the definitive total from discovery
                self.update_progress_signal.emit(indexed_count, -1) # Use -1 or another marker for total if already set

            except queue.Empty:
                if not (self.discovery_thread and self.discovery_thread.is_alive()) and self.indexing_queue.empty():
                    break # Discovery done and queue empty
                continue # Continue waiting for items or sentinel
            except Exception as e:
                self.update_chat_signal.emit(f"Error during indexing file {file_path if 'file_path' in locals() else 'unknown'}: {e}")
                self.indexing_queue.task_done() # Ensure task_done is called even on error
                continue

        self.update_chat_signal.emit("Indexing process finished.")
        # Re-enable button and reset progress label via signals or direct call if sure on main thread
        # For safety, let's assume this thread might not be the main one for UI directly
        # QApplication.instance().postEvent(self, threading.Event()) # Trigger main thread processing
        # self.index_button.setDisabled(False)
        self.set_index_button_enabled_signal.emit(True) # Use signal
        self.update_progress_signal.emit(indexed_count, total_files_to_process if total_files_to_process > 0 else indexed_count) # Final update
        # self.progress_bar.setVisible(False)

    def send_query(self):
        """Processes the user query and displays the response."""
        query = self.query_entry.text()
        if not query:
            return

        self.update_chat_signal.emit(f"You: {query}")
        self.query_entry.clear()

        if not self.embedding_model or self.embedding_model == "No models found" or self.embedding_model == "Ollama not running":
            self.update_chat_signal.emit("Please select a valid embedding model.")
            return
        if not self.chat_model or self.chat_model == "No models found" or self.chat_model == "Ollama not running":
            self.update_chat_signal.emit("Please select a valid chat model.")
            return

        self.send_button.setDisabled(True) # Disable send button during processing
        self.set_send_button_enabled_signal.emit(False) # Use signal to disable
        query_thread = threading.Thread(target=self._process_query, args=(query,))
        query_thread.daemon = True
        query_thread.start()

    def _process_query(self, query):
        """Processes the query, searches documents, and gets chat response (in separate thread)."""
        try:
            query_embedding = generate_embedding(self.embedding_model, query)

            if query_embedding is not None:
                # Perform document search
                k_value = self.top_k_spinbox.value()
                # Ensure query_embedding is a NumPy array before passing to search_documents
                query_embedding_np = np.array(query_embedding, dtype=np.float32)
                search_results = search_documents(DATABASE_FILE, query_embedding_np, k=k_value)

                if search_results:
                    # Prepare context for the chat model
                    context_parts = []
                    for path, score in search_results:
                        file_content = process_file(path) 
                        if file_content:
                            context_parts.append(f"File: {path}\nScore: {score:.4f}\nContent:\n{file_content[:1000]}\n---") # Use full path
                        else:
                            context_parts.append(f"File: {path}\nScore: {score:.4f}\nContent: [Could not extract content]") # Use full path

                    context_docs = "\n\n".join(context_parts)
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant. Use the provided document information (including file content) to answer the user's query."},
                        {"role": "user", "content": f"Documents:\n{context_docs}\n\nQuery: {query}"}
                    ]

                    response = chat_with_ollama(self.chat_model, messages)

                    if response:
                        self.update_chat_signal.emit(f"Diskuss: {response}")
                    else:
                        self.update_chat_signal.emit("Diskuss: Sorry, I could not get a response from the chat model.")
                else:
                    self.update_chat_signal.emit("Diskuss: No relevant documents found.")
            else:
                self.update_chat_signal.emit("Diskuss: Failed to generate embedding for the query.")
        except Exception as e:
            self.update_chat_signal.emit(f"Diskuss: Error processing query: {e}")
        finally:
             # Re-enable send button on main thread via event posting or direct signal if available
            # QApplication.instance().postEvent(self, threading.Event()) # Dummy event
            # self.send_button.setDisabled(False)
            self.set_send_button_enabled_signal.emit(True) # Use signal

    def update_chat_signal_emit(self, message):
        self.update_chat_signal.emit(message)

    def _insert_chat_message(self, message):
        """Inserts a message into the chat history text widget."""
        self.chat_history.append(message) # QTextEdit's append handles newlines
        self.chat_history.moveCursor(QTextCursor.MoveOperation.End) # Auto-scroll

    def _update_progress_label_slot(self, count, total):
        """Updates the progress label in the main thread."""
        current_total_str = self.progress_label.text().split('/')[-1].replace(" files...","").replace(" Complete.","") # Adjusted to handle "Complete."
        if total == -1: # Marker to use existing total from label
            if current_total_str.isdigit():
                 total = int(current_total_str)
            else: # If current total is '?' or invalid, try to get it from queue or mark as discovering
                 total = self.indexing_queue.qsize() # Approximate, might include sentinel
                 if self.discovery_thread and self.discovery_thread.is_alive():
                     total_str = "? (discovering)"
                 elif total > 0:
                     total_str = str(total)
                 else:
                     total_str = "?"
        elif total == 0 and count == 0: # Reset to idle
            self.progress_label.setText("Progress: Idle")
            # self.progress_bar.setVisible(False)
            return
        elif total > 0:
            total_str = str(total)
            # self.progress_bar.setVisible(True)
            # self.progress_bar.setRange(0, total)
            # self.progress_bar.setValue(count)
        else: # total is 0 or invalid from initial call, but count might be > 0
            total_str = "?"
            # self.progress_bar.setVisible(True)
            # self.progress_bar.setRange(0,0) # Indeterminate

        self.progress_label.setText(f"Progress: Indexed {count}/{total_str} files...")
        if count == total and total > 0:
             self.progress_label.setText(f"Progress: Indexed {count}/{total_str} files. Complete.")
             # self.progress_bar.setValue(total)

    def set_top_k_value(self, value):
        self.top_k_documents_value = value

    def closeEvent(self, event):
        """Handle window close event to stop background threads."""
        self.update_chat_signal.emit("Stopping indexing threads...")
        self.stop_indexing.set() # Signal threads to stop
        # Wait for threads to finish (optional, with timeout)
        if self.discovery_thread and self.discovery_thread.is_alive():
            self.discovery_thread.join(timeout=2)
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2)
        self.update_chat_signal.emit("Exiting Diskuss.")
        event.accept() # Proceed with closing

    def _set_index_button_enabled(self, enabled):
        self.index_button.setEnabled(enabled)

    def _set_send_button_enabled(self, enabled):
        self.send_button.setEnabled(enabled)

    def _update_indexed_docs_count_label(self, count):
        self.indexed_docs_count_label.setText(f"Total Documents Indexed: {count}")

    def refresh_indexed_docs_count(self):
        """Fetches and updates the total indexed documents count."""
        count = count_total_documents(DATABASE_FILE)
        self.update_indexed_docs_count_signal.emit(count)

# Moved main() outside the class
def main():
    app = QApplication(sys.argv)
    # app.setStyle("Fusion") 
    diskuss_app = DiskussApp()
    diskuss_app.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    # DiskussApp.main() # Call as a static method or move main() outside the class 
    main() # Call the main function directly 