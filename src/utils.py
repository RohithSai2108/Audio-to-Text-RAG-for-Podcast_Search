import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

def setup_logging():
    """Setup basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('podcast_rag.log'),
            logging.StreamHandler()
        ]
    )

def validate_audio_file(uploaded_file) -> bool:
    """Validate uploaded audio file format."""
    if uploaded_file is None:
        return False
    
    supported_formats = ['mp3', 'wav', 'm4a', 'flac', 'ogg']
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    return file_extension in supported_formats

def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file to temp directory and return path."""
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path

def format_time(seconds: float) -> str:
    """Format seconds to MM:SS format."""
    try:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    except:
        return "00:00"

def load_config() -> Dict[str, Any]:
    """Load configuration from config.json."""
    try:
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        logging.error(f"Error loading config: {e}")
    return {}

def save_episode_data(episode_data: Dict[str, Any]) -> bool:
    """Save processed episode data to persistent storage."""
    try:
        config = load_config()
        episodes_file = config.get('storage', {}).get('episodes_file', 'data/processed_episodes.json')
        
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(episodes_file), exist_ok=True)
        
        # Load existing episodes
        existing_episodes = load_episode_data()
        
        # Add new episode
        episode_id = episode_data.get('id')
        if episode_id:
            existing_episodes[episode_id] = episode_data
            existing_episodes[episode_id]['processed_at'] = datetime.now().isoformat()
            
            # Ensure we don't exceed reasonable limits (safety check)
            if len(existing_episodes) > 100:  # Allow up to 100 episodes
                logging.warning(f"Storage limit reached. Consider clearing old episodes.")
        
        # Save updated episodes
        with open(episodes_file, 'w') as f:
            json.dump(existing_episodes, f, indent=2)
        
        logging.info(f"Saved episode data for: {episode_data.get('title', 'Unknown')}. Total episodes: {len(existing_episodes)}")
        return True
        
    except Exception as e:
        logging.error(f"Error saving episode data: {e}")
        return False

def load_episode_data() -> Dict[str, Any]:
    """Load processed episode data from persistent storage."""
    try:
        config = load_config()
        episodes_file = config.get('storage', {}).get('episodes_file', 'data/processed_episodes.json')
        
        if os.path.exists(episodes_file):
            with open(episodes_file, 'r') as f:
                return json.load(f)
        else:
            return {}
            
    except Exception as e:
        logging.error(f"Error loading episode data: {e}")
        return {}

def save_transcript(episode_id: str, transcript: Dict[str, Any]) -> bool:
    """Save transcript data to persistent storage."""
    try:
        config = load_config()
        transcripts_dir = config.get('storage', {}).get('transcripts_dir', 'data/transcripts')
        
        # Create transcripts directory if it doesn't exist
        os.makedirs(transcripts_dir, exist_ok=True)
        
        # Save transcript
        transcript_file = os.path.join(transcripts_dir, f"{episode_id}_transcript.json")
        with open(transcript_file, 'w') as f:
            json.dump(transcript, f, indent=2)
        
        logging.info(f"Saved transcript for episode: {episode_id}")
        return True
        
    except Exception as e:
        logging.error(f"Error saving transcript: {e}")
        return False

def load_transcript(episode_id: str) -> Optional[Dict[str, Any]]:
    """Load transcript data from persistent storage."""
    try:
        config = load_config()
        transcripts_dir = config.get('storage', {}).get('transcripts_dir', 'data/transcripts')
        
        transcript_file = os.path.join(transcripts_dir, f"{episode_id}_transcript.json")
        if os.path.exists(transcript_file):
            with open(transcript_file, 'r') as f:
                return json.load(f)
        else:
            return None
            
    except Exception as e:
        logging.error(f"Error loading transcript: {e}")
        return None

def get_episode_stats() -> Dict[str, Any]:
    """Get statistics about processed episodes."""
    try:
        episodes = load_episode_data()
        
        if not episodes:
            return {
                "total_episodes": 0,
                "total_duration": 0,
                "total_chunks": 0,
                "episodes": [],
                "storage_capacity": "Ready for 3+ episodes"
            }
        
        total_duration = sum(ep.get('duration', 0) for ep in episodes.values())
        total_chunks = sum(ep.get('chunks', 0) for ep in episodes.values())
        
        # Calculate storage capacity info
        capacity_status = "Ready for more episodes"
        if len(episodes) >= 3:
            capacity_status = f"Successfully storing {len(episodes)} episodes"
        elif len(episodes) == 2:
            capacity_status = "Ready for 3rd episode"
        elif len(episodes) == 1:
            capacity_status = "Ready for 2nd and 3rd episodes"
        
        return {
            "total_episodes": len(episodes),
            "total_duration": total_duration,
            "total_chunks": total_chunks,
            "episodes": list(episodes.values()),
            "storage_capacity": capacity_status,
            "max_capacity": 100  # Maximum episodes allowed
        }
        
    except Exception as e:
        logging.error(f"Error getting episode stats: {e}")
        return {
            "total_episodes": 0,
            "total_duration": 0,
            "total_chunks": 0,
            "episodes": [],
            "storage_capacity": "Ready for 3+ episodes"
        }

def clear_storage() -> bool:
    """Clear all stored episode data and transcripts."""
    try:
        config = load_config()
        episodes_file = config.get('storage', {}).get('episodes_file', 'data/processed_episodes.json')
        transcripts_dir = config.get('storage', {}).get('transcripts_dir', 'data/transcripts')
        
        # Remove episodes file
        if os.path.exists(episodes_file):
            os.remove(episodes_file)
        
        # Remove transcripts directory
        if os.path.exists(transcripts_dir):
            import shutil
            shutil.rmtree(transcripts_dir)
        
        logging.info("Cleared all stored data")
        return True
        
    except Exception as e:
        logging.error(f"Error clearing storage: {e}")
        return False
