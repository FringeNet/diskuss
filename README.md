# Diskuss

Diskuss is a Python-based desktop application that allows users to interact with their local file system using natural language. It leverages Ollama for both embedding generation and chat functionalities, creating a local embedding database for fast document lookup and retrieval.

## Features

*   **Local Embedding Database**: Stores document embeddings and metadata locally (using SQLite).
*   **Ollama Integration**: Uses Ollama's local models for:
    *   Generating embeddings for documents and queries.
    *   Processing natural language queries via a local chat model.
*   **File Type Support**: Handles various file types including DOCX, PDF, TXT, and CSV.
*   **User Interface**: Provides a Tkinter-based graphical interface for:
    *   File and directory selection for indexing.
    *   Selection of Ollama embedding and chat models.
    *   Chat window for queries and responses.
    *   Status check for Ollama service.
*   **Local Operations**: All operations are performed locally, with no reliance on external APIs (Ollama runs locally).

## Prerequisites

*   **Python 3.8+**
*   **Ollama**: You must have Ollama installed and running on your machine. Download it from [https://ollama.com/](https://ollama.com/).
    *   Ensure you have downloaded at least one embedding model (e.g., `nomic-embed-text`) and one chat model (e.g., `llama3`) through Ollama:
        ```bash
        ollama pull nomic-embed-text
        ollama pull llama3
        ```

## Setup and Installation

1.  **Clone the repository (if applicable) or download the source files.**

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install the required Python libraries:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Ensure Ollama is running.** You can check its status from the Diskuss application interface.

## Running the Application

Once the setup is complete and Ollama is running with the necessary models, you can start Diskuss:

```bash
python diskuss_app.py
```

## How to Use

1.  **Launch Diskuss.**
2.  **Check Ollama Status**: Ensure Ollama is running. If not, the application will provide guidance.
3.  **Select Models**: Choose an available Ollama embedding model and a chat model from the dropdown menus.
4.  **Index Files/Directories**:
    *   Use the "Browse File" button to select a single document for indexing.
    *   Use the "Browse Directory" button to select an entire folder. All supported files within the folder (and its subfolders) will be indexed.
    *   Click "Start Indexing". The progress will be displayed.
5.  **Chat with Your Documents**:
    *   Once indexing is complete for your selected files/directories, type your query into the chat input field (e.g., "What are the main points in my project reports?").
    *   Press Enter or click "Send".
    *   Diskuss will find relevant documents based on your query and use the selected Ollama chat model to generate a response based on those documents.

## Project Structure

*   `diskuss_app.py`: Main application file with the Tkinter GUI and core logic.
*   `database.py`: Module for managing the SQLite database for embeddings.
*   `file_processor.py`: Module for extracting text from different file types.
*   `ollama_integration.py`: Module for interacting with the Ollama API.
*   `requirements.txt`: Lists all Python dependencies.
*   `requirements.md`: Detailed list of dependencies and setup instructions (you are reading it or a version of it).
*   `LICENSE`: MIT License file.

## Contributing

Contributions are welcome! If you'd like to contribute to Diskuss, please follow these general steps:

1.  **Fork the Repository**: Click the 'Fork' button at the top right of the repository page.
2.  **Clone Your Fork**: Clone your forked repository to your local machine.
    ```bash
    git clone https://github.com/YOUR_USERNAME/Diskuss.git
    cd Diskuss
    ```
3.  **Create a New Branch**: Create a new branch for your feature or bug fix.
    ```bash
    git checkout -b your-feature-branch-name
    ```
4.  **Make Your Changes**: Implement your feature or fix the bug. Ensure your code follows the project's style and a clear commit history.
5.  **Commit Your Changes**:
    ```bash
    git add .
    git commit -m "feat: Describe your feature or fix"
    ```
6.  **Push to Your Fork**:
    ```bash
    git push origin your-feature-branch-name
    ```
7.  **Open a Pull Request (PR)**: Go to the original Diskuss repository on GitHub and click the 'New pull request' button. Choose your fork and branch to compare with the main branch of the original repository.
    *   Provide a clear title and description for your PR, explaining the changes you've made.

We will review your PR as soon as possible. Thank you for your contribution!

## License

This project is licensed under the MIT License - see the `LICENSE` file for details. 