# Rufus - Intelligent Web Data Extraction for RAG Systems

Rufus is an AI-powered web scraping tool designed specifically for preparing web content for Retrieval-Augmented Generation (RAG) systems. It intelligently crawls websites based on user instructions and extracts relevant content in a structured format suitable for RAG pipelines.

## Features

- 🤖 LLM-powered content relevance analysis
- 🌐 Intelligent web crawling with domain scope
- 📑 Smart content extraction and structuring
- 🚀 Concurrent processing for improved performance
- 🛡️ Built-in rate limiting and error handling
- 📊 Relevance scoring for extracted content
- 🎯 Instruction-based targeted scraping

## Setup

Create the environment from the environment.yml file:

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate ruf
```

## Quick Start

```python
from rufus import RufusClient

# Initialize the client
client = RufusClient(api_key="your-openai-api-key")

# Scrape website with specific instructions
instructions = "Find information about product features and pricing"
results = client.scrape("https://example.com", instructions)

# Save results
client.save_results(results, "output.json")
```

## Integration with RAG Pipeline

Rufus is designed to seamlessly integrate with RAG systems. Here's an example using LangChain:

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from rufus import RufusClient

# 1. Scrape and prepare data
client = RufusClient(api_key="your-openai-api-key")
results = client.scrape("https://example.com", "Find product documentation")

# 2. Process results for RAG
documents = []
for result in results:
    if result["is_relevant"]:
        for chunk in result["data"]:
            documents.append({
                "page_content": chunk["text"],
                "metadata": {
                    "url": result["url"],
                    "relevance_score": result["relevance_score"],
                    "title": chunk.get("title", ""),
                }
            })

# 3. Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embeddings)

# 4. Create retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)
```

## Configuration Options

```python
client = RufusClient(
    api_key="your-openai-api-key",
    max_pages=10,  # Maximum pages to crawl
    concurrent_requests=3  # Number of concurrent requests
)
```

## Output Format

Rufus provides structured output in JSON format:

```json
[
    {
        "url": "https://example.com/page",
        "data": [
            {
                "text": "Extracted content...",
                "title": "Section title"
            }
        ],
        "is_relevant": true,
        "relevance_score": 0.85,
        "llm_extracted_data": {
            // Structured data extracted by LLM
        }
    }
]
```

## Technical Approach & Challenges

### Architecture Overview

Rufus implements a sophisticated pipeline for web content extraction:

1. **URL Management**
   - Domain-scoped crawling
   - Duplicate URL detection
   - URL validation and filtering

2. **Content Processing**
   - HTML parsing with BeautifulSoup
   - Content chunking with unstructured
   - LLM-powered relevance analysis

3. **Data Extraction**
   - Few-shot learning for content analysis
   - Structured JSON output
   - Metadata preservation

### Challenges & Solutions

1. **LLM Response Consistency**
   - Challenge: Inconsistent JSON outputs from LLM
   - Solution: Implemented few-shot learning with explicit examples and strict JSON validation

2. **Content Quality**
   - Challenge: Extracting meaningful content from diverse web structures
   - Solution: Used unstructured library for intelligent content parsing and chunking

3. **Performance**
   - Challenge: Slow sequential processing
   - Solution: Implemented concurrent processing with ThreadPoolExecutor

4. **Rate Limiting**
   - Challenge: Avoiding server overload and blocks
   - Solution: Added configurable rate limiting and random user agents

5. **Error Handling**
   - Challenge: Various failure points in the pipeline
   - Solution: Comprehensive error handling and logging system

## Best Practices

1. Start with specific instructions for better content targeting
2. Adjust max_pages based on website size and content distribution
3. Configure concurrent_requests based on website's rate limits
4. Monitor the logs for crawling performance and errors
5. Use relevance_score to filter content for RAG systems
