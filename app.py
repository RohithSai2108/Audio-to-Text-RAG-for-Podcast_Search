import streamlit as st
import os
from src.audio_processor import AudioProcessor
from src.text_indexer import TextIndexer
from src.rag_engine import RAGEngine
from src.utils import (
    setup_logging, validate_audio_file, save_uploaded_file, format_time,
    save_episode_data, load_episode_data, save_transcript, get_episode_stats,
    clear_storage
)

# Page config
st.set_page_config(
    page_title="Podcast RAG Search",
    page_icon="🎙️",
    layout="wide"
)

setup_logging()

@st.cache_resource
def init_components():
    audio_processor = AudioProcessor()
    text_indexer = TextIndexer()
    rag_engine = RAGEngine()
    return audio_processor, text_indexer, rag_engine

audio_processor, text_indexer, rag_engine = init_components()

def main():
    st.title("🎙️ Podcast RAG Search System")
    st.markdown("**Multimodal RAG system for audio podcast processing and topic-based search with timestamps**")

    # Sidebar: Episode Management
    with st.sidebar:
        st.header("📁 Episode Management")
        
        # Upload section
        uploaded_file = st.file_uploader(
            "Upload Podcast Episode",
            type=['mp3', 'wav', 'm4a', 'flac', 'ogg'],
            help="Upload an audio file to process"
        )
        
        if uploaded_file:
            episode_title = st.text_input("Episode Title", value=uploaded_file.name)
            if st.button("🔄 Process Episode", type="primary"):
                with st.spinner("Processing audio file..."):
                    try:
                        if not validate_audio_file(uploaded_file):
                            st.error("Unsupported audio format.")
                        else:
                            file_path = save_uploaded_file(uploaded_file)
                            
                            # Audio-to-text conversion with high accuracy
                            st.info("🎤 Transcribing audio with high accuracy...")
                            transcript = audio_processor.transcribe_with_timestamps(file_path)
                            
                            if transcript and transcript.get("segments"):
                                # Speaker identification and diarization
                                st.info("👥 Identifying speakers...")
                                speakers = audio_processor.identify_speakers(file_path, transcript)
                                
                                # Temporal indexing and chunking
                                st.info("⏰ Creating temporal chunks...")
                                chunks = audio_processor.chunk_transcript_by_time(transcript, speakers)
                                
                                # Searchable text indexing
                                st.info("🔍 Indexing content for search...")
                                episode_id = f"episode_{len(st.session_state.get('episodes', []))}"
                                text_indexer.add_transcript_chunks(chunks, episode_id, episode_title)
                                
                                # Store episode info with persistent storage
                                episode_data = {
                                    'id': episode_id,
                                    'title': episode_title,
                                    'chunks': len(chunks),
                                    'duration': transcript.get('duration', 0),
                                    'speakers': len(set(speakers.values())) if speakers else 0,
                                    'file_name': uploaded_file.name,
                                    'processed_at': None  # Will be set by save function
                                }
                                
                                # Save to persistent storage
                                if save_episode_data(episode_data):
                                    st.success(f"✅ Successfully processed and saved '{episode_title}'!")
                                    st.info(f"📊 Created {len(chunks)} searchable chunks with {len(set(speakers.values())) if speakers else 0} speakers")
                                    
                                    # Save transcript for future use
                                    save_transcript(episode_id, transcript)
                                    
                                    # Update session state
                                    if 'episodes' not in st.session_state:
                                        st.session_state.episodes = []
                                    st.session_state.episodes.append(episode_data)
                                else:
                                    st.error("❌ Failed to save episode data")
                            else:
                                st.error("❌ Transcription failed or no segments found.")
                            os.remove(file_path)
                    except Exception as e:
                        st.error(f"❌ Error processing episode: {e}")

        # Load and display processed episodes from persistent storage
        st.subheader("📚 Processed Episodes")
        episodes_data = load_episode_data()
        
        if episodes_data:
            for episode_id, episode in episodes_data.items():
                with st.expander(f"🎧 {episode['title']}"):
                    st.write(f"**Episode ID:** {episode_id}")
                    st.write(f"**Chunks:** {episode['chunks']}")
                    st.write(f"**Duration:** {format_time(episode['duration'])}")
                    st.write(f"**Speakers:** {episode['speakers']}")
                    if episode.get('processed_at'):
                        st.write(f"**Processed:** {episode['processed_at'][:19]}")
        else:
            st.info("No episodes processed yet")
        
        # Storage management
        with st.expander("🗄️ Storage Management"):
            stats = get_episode_stats()
            st.write(f"**Total Episodes:** {stats['total_episodes']}")
            st.write(f"**Total Duration:** {format_time(stats['total_duration'])}")
            st.write(f"**Total Chunks:** {stats['total_chunks']}")
            
            if st.button("🗑️ Clear All Data"):
                if clear_storage():
                    st.success("✅ All data cleared successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to clear data")

    # Main search interface
    st.header("🔍 Topic-Based Search")
    
    # Model selection
    available_models = rag_engine.get_available_models()
    model_options = []
    for model_id, model_info in available_models.items():
        status_icon = "✅" if model_info['status'] == 'available' else "❌"
        model_options.append(f"{status_icon} {model_info['name']} ({model_id})")
    
    selected_model = st.selectbox(
        "🤖 AI Model",
        ["gemini", "openai"],
        format_func=lambda x: f"✅ {available_models[x]['name']} ({x})" if available_models[x]['status'] == 'available' else f"❌ {available_models[x]['name']} ({x})"
    )
    
    # Search configuration
    with st.expander("⚙️ Search Settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            n_results = st.slider("Number of results", 1, 20, 5)
            search_strategy = st.selectbox(
                "Search Strategy",
                ["semantic", "keyword", "hybrid"],
                format_func=lambda x: {
                    "semantic": "Semantic - Meaning-based search",
                    "keyword": "Keyword - Exact term matching",
                    "hybrid": "Hybrid - Combines semantic and keyword"
                }[x]
            )
        with col2:
            show_timestamps = st.checkbox("Show timestamps", value=True)
            show_speakers = st.checkbox("Show speaker info", value=True)
    
    # Query input 
    query = st.text_input(
        "What topic are you looking for?",
        placeholder="e.g., 'machine learning algorithms', 'startup funding', 'climate change solutions'",
        help="Search across all processed episodes for specific topics"
    )

    # Search buttons in the streamlit
    col_search1, col_search2 = st.columns(2)
    
    with col_search1:
        if st.button("🔍 Search Episodes", type="primary"):
            if query:
                with st.spinner("Searching across episodes..."):
                    try:
                        results = rag_engine.query_podcasts(
                            query, text_indexer, n_results, 
                            search_strategy=search_strategy,
                            model=selected_model
                        )
                        
                        st.subheader("📝 AI Response")
                        st.write(results['response'])
                        
                        # Show model info
                        if results.get('search_info'):
                            search_info = results['search_info']
                            st.caption(f"Model: {search_info.get('model', 'Unknown')} | Strategy: {search_info.get('strategy', 'Unknown')}")
                        
                        # Show sources with timestamps
                        if show_timestamps:
                            with st.expander("📚 Sources & Timestamps"):
                                sources = results['sources']
                                documents = sources.get('documents', [[]])[0]
                                metadatas = sources.get('metadatas', [[]])[0]
                                distances = sources.get('distances', [[]])[0]
                                
                                for i, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
                                    if doc and metadata:
                                        st.markdown(f"**Source {i+1}:**")
                                        st.markdown(f"*Episode:* {metadata.get('episode_title', 'Unknown')}")
                                        
                                        start_time = metadata.get('start_time', 0)
                                        end_time = metadata.get('end_time', 0)
                                        st.markdown(f"*Timestamp:* {format_time(start_time)} - {format_time(end_time)}")
                                        
                                        if show_speakers and metadata.get('speaker'):
                                            st.markdown(f"*Speaker:* {metadata.get('speaker', 'Unknown')}")
                                        
                                        st.markdown(f"*Relevance:* {(1-distance)*100:.1f}%")
                                        st.markdown(f"*Content:* {doc[:300]}...")
                                        st.markdown("---")
                        
                        # Store in recent searches
                        if 'recent_searches' not in st.session_state:
                            st.session_state.recent_searches = []
                        st.session_state.recent_searches.append(f"{query} ({selected_model})")
                        
                    except Exception as e:
                        st.error(f"❌ Search error: {e}")
            else:
                st.warning("Please enter a search query")
    
    with col_search2:
        if st.button("🔬 Cross-Episode Analysis"):
            if query:
                with st.spinner("Analyzing across all episodes..."):
                    try:
                        # Get statistics across episodes
                        episodes_data = load_episode_data()
                        if episodes_data:
                            st.subheader("📊 Cross-Episode Analysis")
                            
                            # Episode statistics
                            total_duration = sum(ep['duration'] for ep in episodes_data.values())
                            total_chunks = sum(ep['chunks'] for ep in episodes_data.values())
                            total_speakers = sum(ep['speakers'] for ep in episodes_data.values())
                            
                            col_stats1, col_stats2, col_stats3 = st.columns(3)
                            with col_stats1:
                                st.metric("Total Episodes", len(episodes_data))
                            with col_stats2:
                                st.metric("Total Duration", format_time(total_duration))
                            with col_stats3:
                                st.metric("Total Chunks", total_chunks)
                            
                            # Topic frequency across episodes
                            st.subheader("🎯 Topic Frequency")
                            st.info(f"Searching for '{query}' across {len(episodes_data)} episodes...")
                            
                            # Perform search across all episodes
                            results = rag_engine.query_podcasts(
                                query, text_indexer, n_results * 2, 
                                search_strategy=search_strategy,
                                model=selected_model
                            )
                            
                            if results.get('sources'):
                                sources = results['sources']
                                documents = sources.get('documents', [[]])[0]
                                metadatas = sources.get('metadatas', [[]])[0]
                                
                                # Group by episode
                                episode_mentions = {}
                                for doc, metadata in zip(documents, metadatas):
                                    if doc and metadata:
                                        episode_title = metadata.get('episode_title', 'Unknown')
                                        if episode_title not in episode_mentions:
                                            episode_mentions[episode_title] = []
                                        episode_mentions[episode_title].append({
                                            'content': doc,
                                            'timestamp': metadata.get('start_time', 0),
                                            'speaker': metadata.get('speaker', 'Unknown')
                                        })
                                
                                # Display episode-wise mentions
                                for episode_title, mentions in episode_mentions.items():
                                    with st.expander(f"📺 {episode_title} ({len(mentions)} mentions)"):
                                        for mention in mentions:
                                            st.markdown(f"**{format_time(mention['timestamp'])}** - {mention['speaker']}")
                                            st.markdown(f"*{mention['content'][:200]}...*")
                                            st.markdown("---")
                        else:
                            st.warning("No episodes processed yet")
                            
                    except Exception as e:
                        st.error(f"❌ Analysis error: {e}")
            else:
                st.warning("Please enter a search query")
    
    # Recent searches
    st.subheader("🕒 Recent Searches")
    if 'recent_searches' in st.session_state and st.session_state.recent_searches:
        for search in st.session_state.recent_searches[-5:]:
            st.write(f"• {search}")
    else:
        st.info("No recent searches")

if __name__ == "__main__":
    main()
