# YouTube Q&A Streamlit Application

This application allows users to input questions and receive relevant YouTube video recommendations. It leverages the Pinecone Vector Database for efficient similarity search and Sentence Transformers for generating semantic embeddings from user queries.

## Technologies Used

### Pinecone Vector Database

[Pinecone](https://www.pinecone.io/) is a vector database designed for machine learning applications. It enables efficient storage and retrieval of high-dimensional vectors, making it ideal for similarity search, recommendation systems, and natural language processing tasks. In this application, Pinecone serves as the backbone for querying similar content based on vector embeddings of user questions, allowing for fast and relevant video recommendations.

### Sentence Transformers

[Sentence Transformers](https://www.sbert.net/) is a library for generating sentence embeddings. It uses transformer models pre-trained on large datasets to produce high-quality vector representations of text. These embeddings capture the semantic meaning of sentences, enabling similarity comparisons and clustering. In our application, Sentence Transformers convert user queries into embeddings, which are then used to search the Pinecone vector database for the most relevant YouTube videos.

## Features

- **Question Input**: Users can input their questions into the application.
- **Relevant Video Recommendations**: The app retrieves relevant YouTube video recommendations based on the question asked.
- **Responsive Cards**: Each recommendation is displayed in a card format with a thumbnail, title, and a snippet of the video description.

3. Install the required Python packages.
    ```
    pip install -r requirements.txt
    ```

## Usage

To run the application:

1. Set your Pinecone API key as an environment variable (replace `YOUR_API_KEY` with your actual API key).
    ```
    export PINECONE_API_KEY=YOUR_API_KEY
    ```
    or on Windows,
    ```
    set PINECONE_API_KEY=YOUR_API_KEY
    ```

2. Start the Streamlit application.
    ```
    streamlit run youtube_qa.py
    ```

3. Open the URL displayed in your terminal in a web browser to view the application.

## Configuration

- **Pinecone API Key**: Ensure your API key is set as mentioned in the usage section.
- **Pinecone Index Name**: The application is configured to use an index named `youtube-search`. If you use a different name, update the index name in the code.

