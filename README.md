# Semantic Search

Semantic Search is a web application designed to help you find the information you need quickly and efficiently using advanced search algorithms. This project utilizes Flask for the web framework, LangChain for text processing, HuggingFace for embeddings, and Qdrant for the vector database.


https://github.com/harshk04/Semantic-Search-Flask/assets/115946158/d4fe25ac-685c-4721-817a-4e2c3c18ad6a



## Features

- Load and embed data from URLs
- Perform semantic searches on the embedded data
- Display search results with relevance scores
- User-friendly interface

## Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

- Python 3.7+
- Pip (Python package manager)
- Qdrant server running locally

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/harshk04/semantic-search.git
    cd semantic-search
    ```

2. Create and activate a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

4. Make sure Qdrant server is running locally on port 6333. You can download and run Qdrant using Docker:

    ```bash
    docker run -p 6333:6333 qdrant/qdrant
    ```

### Usage

1. Run the `embed-data.py` script to load and embed the data:

    ```bash
    python embed-data.py
    ```

    This script will:
    - Download necessary NLTK resources
    - Load data from the specified URLs
    - Split the documents into chunks
    - Embed the text data
    - Load the embeddings into Qdrant

2. Start the Flask application:

    ```bash
    python app.py
    ```

3. Open your web browser and navigate to `http://localhost:5000` to use the application.

### Project Structure
semantic-search/


├── templates/

  └── form.html     # HTML template for the web application

├── embed-data.py    # Script to load and embed data

├── app.py           # Flask application

├── requirements.txt # Python dependencies

└── README.md         # Project documentation


### Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### License

This project is licensed under the MIT License.

### Contact

If you have any questions or need support, feel free to reach out to us at [kumawatharsh2004@gmail.com](mailto:kumawatharsh2004@gmail.com).

Made with ❤️ by [Harsh Kumawat](https://www.linkedin.com/in/harsh-k04/)
