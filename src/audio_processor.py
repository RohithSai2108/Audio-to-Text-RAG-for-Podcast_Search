import whisper
import os
import logging
from typing import List, Dict
import tempfile
import soundfile as sf

class AudioProcessor:
    def __init__(self, model_size="base"):
        """Initialize the audio processor with Whisper model."""
        try:
            self.model = whisper.load_model(model_size)
        except Exception as e:
            logging.error(f"Error loading Whisper model: {e}")
            raise

    def preprocess_audio(self, audio_path: str) -> str:
        """
        Convert audio to mono wav 16kHz using soundfile (no ffmpeg/pydub needed).
        Returns path to preprocessed temp file.
        """
        try:
            # Load audio using soundfile
            data, samplerate = sf.read(audio_path)
            # If not mono, average channels
            if len(data.shape) > 1:
                data = data.mean(axis=1)
            # If not 16000Hz, resample using librosa
            if samplerate != 16000:
                import librosa
                data = librosa.resample(data, orig_sr=samplerate, target_sr=16000)
                samplerate = 16000
            temp_fd, temp_path = tempfile.mkstemp(suffix=".wav")
            sf.write(temp_path, data, samplerate)
            os.close(temp_fd)
            return temp_path
        except Exception as e:
            logging.error(f"Error preprocessing audio: {e}")
            raise

    def transcribe_with_timestamps(self, audio_path: str) -> Dict:
        """
        Transcribe audio with word-level timestamps using Whisper.
        Returns transcript dict with segments, duration, text.
        """
        try:
            processed_path = self.preprocess_audio(audio_path)
            # Check file size and extension
            if not os.path.exists(processed_path) or os.path.getsize(processed_path) == 0:
                logging.error("Audio file is missing or empty.")
                return {}
            result = self.model.transcribe(processed_path, word_timestamps=True, verbose=False)
            os.remove(processed_path)
            transcript = {
                "segments": result.get("segments", []),
                "duration": result.get("duration", 0),
                "text": result.get("text", "")
            }
            if not transcript["segments"]:
                logging.error("Whisper transcription returned no segments.")
                return {}
            return transcript
        except Exception as e:
            logging.error(f"Error transcribing audio: {e}")
            return {}

    def chunk_transcript_by_time(self, transcript: Dict, speakers: Dict[int, str] = None, chunk_duration: int = 30) -> List[Dict]:
        """
        Split transcript into time-based chunks with speaker information.
        Returns list of dicts: text, start_time, end_time, words, speaker.
        """
        try:
            segments = transcript.get("segments", [])
            chunks = []
            current_chunk = {
                'text': '',
                'start_time': None,
                'end_time': None,
                'words': [],
                'speaker': 'Unknown'
            }
            
            for i, segment in enumerate(segments):
                seg_start = segment['start']
                seg_end = segment['end']
                
                # Get speaker for this segment
                speaker = speakers.get(i, 'Unknown') if speakers else 'Unknown'
                
                if current_chunk['start_time'] is None:
                    current_chunk['start_time'] = seg_start
                    current_chunk['speaker'] = speaker
                
                current_chunk['end_time'] = seg_end
                current_chunk['text'] += ' ' + segment['text']
                current_chunk['words'].extend(segment.get('words', []))
                
                # If speaker changes, create new chunk
                if speaker != current_chunk['speaker'] and current_chunk['text'].strip():
                    chunks.append(current_chunk.copy())
                    current_chunk = {
                        'text': '',
                        'start_time': seg_start,
                        'end_time': seg_end,
                        'words': segment.get('words', []),
                        'speaker': speaker
                    }
                    current_chunk['text'] = segment['text']
                elif current_chunk['end_time'] - current_chunk['start_time'] >= chunk_duration:
                    chunks.append(current_chunk.copy())
                    current_chunk = {
                        'text': segment['text'],
                        'start_time': seg_start,
                        'end_time': seg_end,
                        'words': segment.get('words', []),
                        'speaker': speaker
                    }
            
            if current_chunk['text'].strip():
                chunks.append(current_chunk.copy())
            
            return chunks
        except Exception as e:
            logging.error(f"Error chunking transcript: {e}")
            return []

    def identify_speakers(self, audio_path: str, transcript: Dict) -> Dict[int, str]:
        """
        Basic speaker identification using pause detection.
        Returns dict mapping segment index to speaker label.
        """
        try:
            segments = transcript.get('segments', [])
            speakers = {}
            speaker_count = 0
            
            if not segments:
                return {}
            
            for i, segment in enumerate(segments):
                if i == 0:
                    speakers[i] = f"Speaker_{speaker_count}"
                else:
                    prev_end = segments[i-1]['end']
                    curr_start = segment['start']
                    pause = curr_start - prev_end
                    if pause > 2.0:
                        speaker_count += 1
                        speakers[i] = f"Speaker_{speaker_count}"
                    else:
                        speakers[i] = speakers[i-1]
            
            return speakers
        except Exception as e:
            logging.error(f"Error identifying speakers: {e}")
            return {}
