# Main Diskuss application file

import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk
import os
import threading
import queue
import numpy as np # Import numpy

from ollama_integration import is_ollama_running, list_ollama_models, generate_embedding, chat_with_ollama
from file_processor import process_file
from database import initialize_database, add_document, search_documents, DATABASE_FILE

class DiskussApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Diskuss")
        self.root.geometry("800x600")

        self.embedding_model = None
        self.chat_model = None
        self.indexing_queue = queue.Queue()
        self.processing_thread = None
        self.stop_indexing = threading.Event()
        self.top_k_documents = tk.IntVar(value=5) # Default k value

        # --- Ollama Status ---
        self.status_frame = tk.Frame(root)
        self.status_frame.pack(pady=5)

        self.ollama_status_label = tk.Label(self.status_frame, text="Ollama Status: Checking...")
        self.ollama_status_label.pack(side=tk.LEFT)

        self.check_ollama_button = tk.Button(self.status_frame, text="Check Ollama", command=self.check_ollama_status)
        self.check_ollama_button.pack(side=tk.LEFT, padx=5)

        # --- Model Selection ---
        self.model_frame = tk.LabelFrame(root, text="Model Selection")
        self.model_frame.pack(pady=5, padx=10, fill=tk.X)

        tk.Label(self.model_frame, text="Embedding Model:").pack(side=tk.LEFT, padx=5)
        self.embedding_model_combobox = ttk.Combobox(self.model_frame, state="readonly")
        self.embedding_model_combobox.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        tk.Label(self.model_frame, text="Chat Model:").pack(side=tk.LEFT, padx=5)
        self.chat_model_combobox = ttk.Combobox(self.model_frame, state="readonly")
        self.chat_model_combobox.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        self.load_models()

        # --- Indexing ---
        self.index_frame = tk.LabelFrame(root, text="Document Indexing")
        self.index_frame.pack(pady=5, padx=10, fill=tk.X)

        self.path_entry = tk.Entry(self.index_frame)
        self.path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        self.browse_file_button = tk.Button(self.index_frame, text="Browse File", command=self.browse_file)
        self.browse_file_button.pack(side=tk.LEFT, padx=5)

        self.browse_dir_button = tk.Button(self.index_frame, text="Browse Directory", command=self.browse_directory)
        self.browse_dir_button.pack(side=tk.LEFT, padx=5)

        self.index_button = tk.Button(self.index_frame, text="Start Indexing", command=self.start_indexing)
        self.index_button.pack(side=tk.LEFT, padx=5)

        self.progress_label = tk.Label(self.index_frame, text="Progress: Idle")
        self.progress_label.pack(side=tk.LEFT, padx=5)

        # --- Chat ---
        self.chat_frame = tk.LabelFrame(root, text="Chat")
        self.chat_frame.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)

        self.chat_history = scrolledtext.ScrolledText(self.chat_frame, state='disabled', wrap=tk.WORD)
        self.chat_history.pack(pady=5, padx=5, fill=tk.BOTH, expand=True)

        self.query_controls_frame = tk.Frame(self.chat_frame) # New frame for query input and k selection
        self.query_controls_frame.pack(fill=tk.X, padx=5, pady=5)

        self.query_entry = tk.Entry(self.query_controls_frame)
        self.query_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.query_entry.bind("<Return>", self.send_query_event)

        tk.Label(self.query_controls_frame, text="Top K:").pack(side=tk.LEFT, padx=(10, 2))
        self.top_k_spinbox = tk.Spinbox(self.query_controls_frame, from_=1, to=20, textvariable=self.top_k_documents, width=3)
        self.top_k_spinbox.pack(side=tk.LEFT)

        self.send_button = tk.Button(self.query_controls_frame, text="Send", command=self.send_query)
        self.send_button.pack(side=tk.LEFT, padx=5)

        # Initialize database
        initialize_database()
        self.check_ollama_status()

    def check_ollama_status(self):
        """Checks and updates the Ollama service status."""
        if is_ollama_running():
            self.ollama_status_label.config(text="Ollama Status: Running", fg="green")
            # Enable features that require Ollama
            self.embedding_model_combobox.config(state="readonly")
            self.chat_model_combobox.config(state="readonly")
            self.index_button.config(state="normal")
            self.send_button.config(state="normal")
        else:
            self.ollama_status_label.config(text="Ollama Status: Not Running. Please start Ollama.", fg="red")
            # Disable features that require Ollama
            self.embedding_model_combobox.config(state="disabled")
            self.chat_model_combobox.config(state="disabled")
            self.index_button.config(state="disabled")
            self.send_button.config(state="disabled")

    def load_models(self):
        """Loads available Ollama models into the comboboxes."""
        models = list_ollama_models()
        if models:
            self.embedding_model_combobox['values'] = models
            self.chat_model_combobox['values'] = models
            # Attempt to select default models if available
            for model in models:
                if 'embed' in model:
                    self.embedding_model_combobox.set(model)
                    self.embedding_model = model
                    break
            for model in models:
                if 'llama' in model or 'mistral' in model: # Simple check for common chat models
                    self.chat_model_combobox.set(model)
                    self.chat_model = model
                    break
        else:
            self.embedding_model_combobox['values'] = ["No models found"]
            self.chat_model_combobox['values'] = ["No models found"]
            self.embedding_model_combobox.set("No models found")
            self.chat_model_combobox.set("No models found")
            self.embedding_model_combobox.config(state="disabled")
            self.chat_model_combobox.config(state="disabled")

        self.embedding_model_combobox.bind("<<ComboboxSelected>>", self.set_embedding_model)
        self.chat_model_combobox.bind("<<ComboboxSelected>>", self.set_chat_model)

    def set_embedding_model(self, event):
        self.embedding_model = self.embedding_model_combobox.get()
        self.update_chat_history(f"Embedding model set to: {self.embedding_model}")

    def set_chat_model(self, event):
        self.chat_model = self.chat_model_combobox.get()
        self.update_chat_history(f"Chat model set to: {self.chat_model}")

    def browse_file(self):
        """Opens a file dialog to select a file for indexing."""
        file_path = filedialog.askopenfilename(
            filetypes=(
                ("Document Files", "*.docx *.pdf *.txt *.csv"),
                ("All files", "*.*")
            )
        )
        if file_path:
            self.path_entry.delete(0, tk.END)
            self.path_entry.insert(0, file_path)

    def browse_directory(self):
        """Opens a directory dialog to select a folder for indexing."""
        dir_path = filedialog.askdirectory()
        if dir_path:
            self.path_entry.delete(0, tk.END)
            self.path_entry.insert(0, dir_path)

    def start_indexing(self):
        """Starts the file/directory indexing process in a separate thread."""
        path = self.path_entry.get()
        if not path:
            self.update_chat_history("Please select a file or directory to index.")
            return

        if not self.embedding_model:
            self.update_chat_history("Please select an embedding model.")
            return

        self.index_button.config(state="disabled")
        self.progress_label.config(text="Progress: Preparing for indexing...")

        self.stop_indexing.clear() # Clear the stop signal
        self.indexing_queue = queue.Queue() # Clear previous queue

        # Start a thread to find files and populate the queue
        discovery_thread = threading.Thread(target=self._discover_files, args=(path,))
        discovery_thread.start()

        # Start a separate thread pool or a single thread to process the queue
        # For simplicity, using a single processing thread here.
        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.processing_thread = threading.Thread(target=self._process_indexing_queue)
            self.processing_thread.start()

    def _discover_files(self, path):
        """Recursively finds supported files in a directory and puts them in the queue."""
        threading.current_thread().name = '_discover_files_thread' # Name the thread
        supported_extensions = ['.docx', '.pdf', '.txt', '.csv'] # Add other extensions here
        files_found_count = 0

        if os.path.isfile(path):
            if os.path.splitext(path)[1].lower() in supported_extensions:
                self.indexing_queue.put(path)
                files_found_count = 1
            else:
                self.update_chat_history(f"Unsupported file type for indexing: {os.path.basename(path)}")
                self.root.after(0, lambda: self.index_button.config(state="normal"))
                self.root.after(0, lambda: self.progress_label.config(text="Progress: Idle"))
                return
        elif os.path.isdir(path):
            for root_dir, _, files in os.walk(path):
                for file in files:
                    if self.stop_indexing.is_set():
                        self.update_chat_history("Indexing stopped by user.")
                        break
                    file_path = os.path.join(root_dir, file)
                    if os.path.splitext(file_path)[1].lower() in supported_extensions:
                        self.indexing_queue.put(file_path)
                        files_found_count += 1
                if self.stop_indexing.is_set():
                    break
        else:
            self.update_chat_history(f"Invalid path: {path}")
            self.root.after(0, lambda: self.index_button.config(state="normal"))
            self.root.after(0, lambda: self.progress_label.config(text="Progress: Idle"))
            return

        if files_found_count > 0:
            self.update_chat_history(f"Found {files_found_count} supported files. Starting indexing...")
            self.root.after(0, self._update_progress_label, 0, files_found_count) # Initial progress update
        else:
            self.update_chat_history("No supported files found in the selected path.")
            self.root.after(0, lambda: self.index_button.config(state="normal"))
            self.root.after(0, lambda: self.progress_label.config(text="Progress: Idle"))
        
        # Add a 'None' sentinel to the queue to signal the end of discovery
        self.indexing_queue.put(None) # Sentinel value

    def _process_indexing_queue(self):
        """Processes files from the queue (run in a separate thread)."""
        indexed_count = 0
        total_files_to_process = 0 # To keep track of initial queue size for progress
        # Get initial size if possible (might be slightly off if discovery is very fast)
        if self.indexing_queue.qsize() > 0:
            # The sentinel None is also in queue, so -1 if it's already there
            total_files_to_process = self.indexing_queue.qsize() -1 if self.indexing_queue.queue[-1] is None else self.indexing_queue.qsize()
            if total_files_to_process < 0: total_files_to_process = 0

        while True:
            if self.stop_indexing.is_set():
                break
            try:
                file_path = self.indexing_queue.get(timeout=1) # Use a timeout
                if file_path is None:
                    self.indexing_queue.task_done()
                    break # Sentinel value received, exit loop

                self.update_chat_history(f"Processing {file_path}")
                text = process_file(file_path)

                if text:
                    embedding_list = generate_embedding(self.embedding_model, text)
                    if embedding_list is not None:
                        # Convert list to numpy array for database storage
                        embedding_np = np.array(embedding_list, dtype=np.float32)
                        add_document(DATABASE_FILE, file_path, embedding_np)
                        indexed_count += 1
                        self.update_chat_history(f"Indexed {os.path.basename(file_path)}")
                    else:
                        self.update_chat_history(f"Failed to generate embedding for {os.path.basename(file_path)}")
                else:
                    self.update_chat_history(f"Failed to extract text from {os.path.basename(file_path)}")

                self.indexing_queue.task_done()
                # Update progress label (requires sending back to main thread)
                self.root.after(0, self._update_progress_label, indexed_count, total_files_to_process)
            except queue.Empty:
                # Queue is empty, but discovery might still be running or finished.
                # If discovery thread is done and queue is empty, we might be done.
                if not any(t.name == '_discover_files_thread' and t.is_alive() for t in threading.enumerate()):
                    if self.indexing_queue.empty(): # Double check if really empty
                        break
                pass # Otherwise, continue waiting for items or the sentinel

        self.update_chat_history("Indexing process finished.")
        self.root.after(0, lambda: self.index_button.config(state="normal"))
        self.root.after(0, lambda: self.progress_label.config(text="Progress: Idle"))

    def _update_progress_label(self, count, total):
        """Updates the progress label in the main thread."""
        # If total is 0 (e.g., single file or error in calculation), show just count
        if total <= 0 and self.processing_thread and self.processing_thread.is_alive():
            total_str = "?" # Still discovering or issue with total calculation
        else:
            total_str = str(total)
        self.progress_label.config(text=f"Progress: Indexed {count}/{total_str} files...")

    def send_query_event(self, event):
        """Handles sending query when Enter key is pressed."""
        self.send_query()

    def send_query(self):
        """Processes the user query and displays the response."""
        query = self.query_entry.get()
        if not query:
            return

        self.update_chat_history(f"You: {query}")
        self.query_entry.delete(0, tk.END)

        if not self.embedding_model or not self.chat_model:
            self.update_chat_history("Please select both embedding and chat models.")
            return

        # Run query processing in a separate thread
        query_thread = threading.Thread(target=self._process_query, args=(query,))
        query_thread.start()

    def _process_query(self, query):
        """Processes the query, searches documents, and gets chat response (in separate thread)."""
        query_embedding = generate_embedding(self.embedding_model, query)

        if query_embedding is not None:
            # Perform document search
            k_value = self.top_k_documents.get()
            search_results = search_documents(DATABASE_FILE, query_embedding, k=k_value)

            if search_results:
                # Prepare context for the chat model
                context_parts = []
                for path, score in search_results:
                    file_content = process_file(path) # Extract file content
                    if file_content:
                        context_parts.append(f"File: {path}\nScore: {score:.4f}\nContent:\n{file_content[:1000]}\n---") # Limit content length for context
                    else:
                        context_parts.append(f"File: {path}\nScore: {score:.4f}\nContent: [Could not extract content]")

                context_docs = "\n\n".join(context_parts)
                messages = [
                    {"role": "system", "content": "You are a helpful assistant. Use the provided document information (including file content) to answer the user's query."},
                    {"role": "user", "content": f"Documents:\n{context_docs}\n\nQuery: {query}"}
                ]

                response = chat_with_ollama(self.chat_model, messages)

                if response:
                    self.update_chat_history(f"Diskuss: {response}")
                else:
                    self.update_chat_history("Diskuss: Sorry, I could not get a response from the chat model.")
            else:
                self.update_chat_history("Diskuss: No relevant documents found.")
        else:
            self.update_chat_history("Diskuss: Failed to generate embedding for the query.")

    def update_chat_history(self, message):
        """Updates the chat history text widget (thread-safe)."""
        self.root.after(0, self._insert_chat_message, message)

    def _insert_chat_message(self, message):
        """Inserts a message into the chat history text widget."""
        self.chat_history.config(state='normal')
        self.chat_history.insert(tk.END, message + "\n")
        self.chat_history.config(state='disabled')
        self.chat_history.see(tk.END) # Auto-scroll to the bottom

def main():
    root = tk.Tk()
    app = DiskussApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 