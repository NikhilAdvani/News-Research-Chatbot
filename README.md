# News Research Chatbot 📈

This is a Streamlit-based news research chatbot that utilizes large language models (LLMs) to answer questions based on news articles. The project uses the LangChain framework to process and handle news articles, create embeddings, and retrieve relevant information to answer user queries.

## Features

- **Load News Articles**: Input URLs of news articles to be processed.
- **Data Processing**: Automatically load, split, and process the articles.
- **Embeddings Creation**: Convert article chunks into embeddings using OpenAI.
- **FAISS Indexing**: Store embeddings in a FAISS index for efficient retrieval.
- **Question Answering**: Retrieve relevant chunks based on user queries and use LLM to provide answers.
- **Source Display**: Display sources of the information provided in the answers.

## Project Structure

1. **Input**: Provide news article URLs.
2. **Load Data**: Use `UnstructuredURLLoader` to load article content.
3. **Text Splitting**: Split content into manageable chunks using `RecursiveCharacterTextSplitter`.
4. **Embeddings**: Create embeddings from text chunks using `OpenAIEmbeddings`.
5. **Indexing**: Store embeddings in a FAISS index.
6. **Retrieval**: Retrieve relevant chunks using `RetrievalQAWithSourcesChain`.
7. **Answering**: Use an LLM to generate answers based on retrieved chunks.

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/NikhilAdvani/financial-research-chatbot.git
    cd financial-research-chatbot
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Create a `.env` file in the root directory and add your OpenAI API key:
    ```env
    OPENAI_API_KEY=your_openai_api_key
    ```

### Running the Application

1. Run the Streamlit application:
    ```bash
    streamlit run main.py
    ```

2. Open your browser and go to `http://localhost:8501` to use the application.

### Usage

1. Enter the URLs of the news articles you want to analyze in the sidebar.
2. Click "Process URLs" to load and process the articles.
3. Enter your question in the "Question" text box.
4. View the answer and the sources displayed by the application.


## Code Explanation

Here's a brief overview of the main code components:

- **Loading Data**: The `UnstructuredURLLoader` loads the content from the provided URLs.
- **Text Splitting**: `RecursiveCharacterTextSplitter` splits the loaded content into smaller chunks.
- **Embeddings Creation**: `OpenAIEmbeddings` converts these chunks into embeddings.
- **FAISS Indexing**: The embeddings are stored in a FAISS index for efficient retrieval.
- **Retrieval and Answering**: `RetrievalQAWithSourcesChain` retrieves relevant chunks and uses an LLM to generate answers based on them.
- **Displaying Results**: The application displays the generated answer and its sources.
