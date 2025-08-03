# 🎙️ Podcast RAG Search System

A **multimodal RAG system** that processes audio podcasts, converts them to searchable text, and allows users to query specific topics mentioned across multiple episodes with timestamp references.

## 🎯 Core Features

### ✅ **Audio-to-Text Conversion with High Accuracy**
- Supports multiple audio formats (MP3, WAV, M4A, FLAC, OGG)
- Uses OpenAI Whisper for accurate speech-to-text conversion
- Includes word-level timestamps for precise audio segment mapping
- Audio preprocessing with noise reduction and format conversion

### ✅ **Searchable Text Indexing**
- Indexes transcript chunks across multiple podcast episodes
- Uses ChromaDB for efficient vector storage and retrieval
- Supports semantic, keyword, and hybrid search strategies
- **Multi-episode storage** - can handle 3+ episodes simultaneously

### ✅ **Topic-Based Querying with Contextual Understanding**
- **🤖 Gemini 1.5 Flash** (default) - Google's latest advanced language model
- **🤖 OpenAI GPT-3.5-turbo** - Alternative AI model option
- Contextual understanding of podcast content
- Provides direct quotes and relevant excerpts with timestamps

### ✅ **Enhanced Speaker Identification & Timestamp Referencing**
- **Speaker diarization** with automatic speaker detection
- **Precise timestamp mapping** for audio segments (MM:SS format)
- **Speaker information** included in search results
- **Temporal chunking** with speaker-aware segmentation

### ✅ **Multi-Episode Search Capabilities**
- **Search across all processed episodes** simultaneously
- **Cross-episode topic correlation** and analysis
- **Episode-wise result grouping** and statistics
- **Storage capacity** for 3+ episodes (up to 100 episodes)

### ✅ **Persistent Storage System**
- **Saves processed episodes** for future use
- **Stores transcripts** with full metadata
- **Cross-session persistence** - episodes remain available after restart
- **Storage management** with statistics and cleanup options
- **Episode statistics** showing total episodes, duration, and chunks

## 🚀 Quick Start

### 1. **Installation**
```bash
# Clone the repository
git clone <repository-url>
cd podcast-rag

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. **Configuration**
Edit `config.json` to add your API keys:
```json
{
    "api_keys": {
        "openai": "your-openai-api-key-here",
        "gemini": "your-gemini-api-key-here"
    },
    "model": {
        "default": "gemini",
        "available": ["gemini", "openai"]
    },
    "audio": {
        "supported_formats": ["mp3", "wav", "m4a", "flac", "ogg"],
        "chunk_duration": 30
    },
    "search": {
        "default_strategy": "semantic",
        "default_results": 5,
        "max_results": 20
    },
    "storage": {
        "episodes_file": "data/processed_episodes.json",
        "transcripts_dir": "data/transcripts"
    }
}
```

### 3. **Run the Application**
```bash
streamlit run app.py
```

## 📁 Project Structure

```
podcast-rag/
├── app.py                 # Main Streamlit application
├── config.json           # Configuration file
├── requirements.txt      # Python dependencies
├── src/
│   ├── audio_processor.py  # Audio-to-text conversion + speaker identification
│   ├── text_indexer.py     # Searchable text indexing (ChromaDB)
│   ├── rag_engine.py       # RAG functionality (Gemini 1.5 Flash + OpenAI)
│   └── utils.py           # Utility functions + persistent storage
├── data/                 # Persistent storage directory
│   ├── processed_episodes.json  # Saved episode metadata
│   └── transcripts/      # Saved transcript files
├── chroma_db/           # Vector database storage
└── temp/                # Temporary file storage
```

## 🔧 Technical Implementation

### **Audio Processing Pipeline**
1. **Audio Preprocessing**: Noise reduction and format conversion to mono WAV 16kHz
2. **Speech-to-Text**: High-accuracy transcription with word-level timestamps
3. **Speaker Identification**: Basic speaker diarization using pause detection
4. **Temporal Chunking**: Time-based segment creation with speaker information

### **Search & Retrieval System**
1. **Text Indexing**: Vector embeddings using Sentence Transformers (`all-MiniLM-L6-v2`)
2. **Search Strategies**: Semantic, keyword, and hybrid approaches
3. **Context Retrieval**: Relevant segment extraction with metadata
4. **Multi-Episode Support**: Indexes content from multiple episodes simultaneously

### **RAG Pipeline**
1. **Query Processing**: Understanding user intent
2. **Context Formation**: Relevant transcript segments with speaker info
3. **Response Generation**: AI-powered answers with timestamps (Gemini 1.5 Flash/OpenAI)

### **Persistent Storage System**
1. **Episode Metadata**: Saves episode information with timestamps and statistics
2. **Transcript Storage**: Stores full transcripts for future reference
3. **Cross-Session Persistence**: Episodes remain available after restart
4. **Storage Capacity**: Supports 3+ episodes (up to 100 episodes)

## 🎛️ Usage

### **Uploading Episodes**
1. Use the sidebar to upload audio files (MP3, WAV, M4A, FLAC, OGG)
2. Enter episode title and click "Process Episode"
3. System will transcribe, identify speakers, index, and **save for future use**
4. Episodes persist across application restarts
5. **Multi-episode support** - upload 3+ episodes for cross-episode analysis

### **Searching Content**
1. Select your preferred AI model (Gemini 1.5 Flash or OpenAI)
2. Enter your topic or question in the search box
3. Choose search strategy (semantic/keyword/hybrid)
4. View AI-generated responses with timestamps and speaker information
5. Explore cross-episode analysis for broader insights

### **Storage Management**
- View total episodes, duration, and chunks processed
- **Storage capacity indicators**: Shows readiness for additional episodes
- Clear all data if needed
- Episodes are automatically saved and persist across sessions

### **AI Models**
- **🤖 Gemini 1.5 Flash** (Default): Google's latest advanced language model
- **🤖 OpenAI GPT-3.5-turbo**: Alternative AI model option
- Model status is shown with availability indicators

### **Search Strategies**
- **Semantic**: Best for conceptual questions and understanding context
- **Keyword**: Best for finding specific terms and exact matches  
- **Hybrid**: Combines semantic understanding with keyword precision

## 🔍 Example Queries

- "What did they say about machine learning algorithms?"
- "Find discussions about startup funding"
- "What are the main points about climate change solutions?"
- "Show me all mentions of artificial intelligence"
- "What did Speaker_1 say about productivity tips?"

## 📊 Features Overview

| Feature | Description | Status |
|---------|-------------|--------|
| Audio-to-Text | High-accuracy transcription with timestamps | ✅ |
| Multi-Episode Search | Search across all processed episodes (3+ episodes) | ✅ |
| Topic-Based Querying | AI-powered contextual responses (Gemini 1.5 Flash/OpenAI) | ✅ |
| Timestamp Referencing | Precise audio segment mapping (MM:SS format) | ✅ |
| Speaker Identification | Basic speaker diarization with pause detection | ✅ |
| Cross-Episode Analysis | Topic correlation across episodes | ✅ |
| Persistent Storage | Episodes saved for future use | ✅ |
| Model Selection | Choose between Gemini 1.5 Flash and OpenAI | ✅ |
| Storage Capacity | Supports 3+ episodes (up to 100 episodes) | ✅ |

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **Audio Processing**: OpenAI Whisper, Soundfile, Librosa
- **Vector Database**: ChromaDB
- **Embeddings**: Sentence Transformers (`all-MiniLM-L6-v2`)
- **AI Models**: Google Gemini 1.5 Flash, OpenAI GPT-3.5-turbo
- **Storage**: JSON-based persistent storage
- **Audio Formats**: MP3, WAV, M4A, FLAC, OGG

## 📝 Requirements Met

✅ **Audio-to-text conversion with high accuracy**  
✅ **Searchable text indexing across multiple podcast episodes**  
✅ **Topic-based querying with contextual understanding**  
✅ **Timestamp referencing for audio segments**  
✅ **Multi-episode search capabilities**  
✅ **Audio preprocessing and noise reduction**  
✅ **Speech-to-text accuracy optimization**  
✅ **Speaker identification and diarization**  
✅ **Temporal indexing and timestamp mapping**  
✅ **Cross-episode topic correlation**  
✅ **Persistent storage for processed episodes**  
✅ **Gemini 1.5 Flash integration as default model**  
✅ **Multi-episode storage (3+ episodes)**  

## 🔑 API Key Setup

### **Gemini API Key**
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Add it to `config.json` under `api_keys.gemini`

### **OpenAI API Key**
1. Go to [OpenAI Platform](https://platform.openai.com/api-keys)
2. Create a new API key
3. Add it to `config.json` under `api_keys.openai`

## 🎉 Ready to Use!

The system now includes:
- **Gemini 1.5 Flash as the default AI model** for superior responses
- **Multi-episode storage** supporting 3+ episodes simultaneously
- **Enhanced speaker identification** with diarization
- **Persistent storage** so processed episodes are saved for future use
- **Cross-session persistence** - your episodes remain available after restart
- **All core multimodal RAG functionality** optimized for podcast analysis

**Upload your podcast episodes and start searching for specific topics with precise timestamp references and speaker information!** 🚀

## 🔧 Troubleshooting

### **Common Issues**
- **API Key Errors**: Ensure your Gemini/OpenAI API keys are valid and properly configured
- **Audio Processing**: Supported formats are MP3, WAV, M4A, FLAC, OGG
- **Storage**: System can handle 3+ episodes with automatic persistence
- **Speaker Identification**: Uses pause detection for basic speaker diarization

### **Performance Tips**
- **Multiple Episodes**: Upload 3+ episodes for cross-episode analysis
- **Search Strategies**: Use semantic for conceptual questions, keyword for specific terms
- **Model Selection**: Gemini 1.5 Flash provides faster, more accurate responses
# Audio-to-Text-RAG-for-Podcast_Search
