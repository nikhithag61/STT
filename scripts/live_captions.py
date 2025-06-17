import whisper
import pyaudio
import wave
import threading
import time
import numpy as np
import tempfile
import os
from collections import deque

class LiveCaptioning:
    def __init__(self, model_size="base"):
        """
        Initialize the live captioning system
        model_size: 'tiny', 'base', 'small', 'medium', 'large' (base recommended for CPU)
        """
        print("Loading Whisper model...")
        
        # Map model sizes to local file paths
        model_paths = {
            'tiny': '../models/tiny.pt',
            'base': '../models/base.pt',
            'small': '../models/small.pt',
            'medium': '../models/medium.pt',
            'large': '../models/large-v3.pt'
        }
        
        model_path = model_paths.get(model_size, 'models/base.pt')
        self.model = whisper.load_model(model_path)
        print(f"Whisper {model_size} model loaded successfully from {model_path}!")
        
        # Audio settings
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000  # Whisper works best with 16kHz
        self.RECORD_SECONDS = 3  # Process every 3 seconds
        
        # Audio buffer
        self.audio_buffer = deque(maxlen=int(self.RATE * 10))  # 10 second buffer
        self.is_recording = False
        
        # PyAudio setup
        self.audio = pyaudio.PyAudio()
        
    def start_audio_stream(self):
        """Start the audio input stream"""
        try:
            self.stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK,
                stream_callback=self.audio_callback
            )
            print("Audio stream started successfully!")
            return True
        except Exception as e:
            print(f"Error starting audio stream: {e}")
            return False
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for audio stream"""
        # Convert bytes to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        
        # Add to buffer
        self.audio_buffer.extend(audio_data)
        
        return (in_data, pyaudio.paContinue)
    
    def get_audio_chunk(self):
        """Get audio chunk for processing"""
        if len(self.audio_buffer) < self.RATE * self.RECORD_SECONDS:
            return None
            
        # Get last N seconds of audio
        chunk_size = int(self.RATE * self.RECORD_SECONDS)
        audio_chunk = list(self.audio_buffer)[-chunk_size:]
        
        # Convert to numpy array and normalize
        audio_array = np.array(audio_chunk, dtype=np.float32)
        audio_array = audio_array / 32768.0  # Normalize to [-1, 1]
        
        return audio_array
    
    def transcribe_audio(self, audio_data):
        """Transcribe audio using Whisper"""
        try:
            # Use Whisper to transcribe
            result = self.model.transcribe(
                audio_data,
                language="en",  # Set to English for better performance
                task="transcribe",
                fp16=False,  # Set to False for CPU
                verbose=False
            )
            
            return result["text"].strip()
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""
    
    def start_live_captioning(self):
        """Main function to start live captioning"""
        print("\n" + "="*50)
        print("LIVE CAPTIONING STARTED")
        print("="*50)
        print("Speak into your microphone...")
        print("Press Ctrl+C to stop")
        print("="*50 + "\n")
        
        if not self.start_audio_stream():
            return
        
        self.is_recording = True
        self.stream.start_stream()
        
        try:
            while self.is_recording:
                # Get audio chunk
                audio_chunk = self.get_audio_chunk()
                
                if audio_chunk is not None:
                    # Check if there's actually audio (not silence)
                    if np.max(np.abs(audio_chunk)) > 0.01:  # Threshold for silence detection
                        # Transcribe
                        text = self.transcribe_audio(audio_chunk)
                        
                        if text and len(text.strip()) > 0:
                            # Display with timestamp
                            timestamp = time.strftime("%H:%M:%S")
                            print(f"[{timestamp}] {text}")
                
                # Small delay to prevent excessive CPU usage
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n\nStopping live captioning...")
        finally:
            self.stop_captioning()
    
    def stop_captioning(self):
        """Stop the captioning system"""
        self.is_recording = False
        
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        
        self.audio.terminate()
        print("Live captioning stopped.")

def main():
    print("Live Captioning System")
    print("======================")
    
    # Choose model size (smaller = faster, larger = more accurate)
    model_choices = {
        '1': 'tiny',    # Fastest, least accurate
        '2': 'base',    # Good balance (recommended)
        '3': 'small',   # Better accuracy, slower
        '4': 'medium',  # High accuracy, much slower
        '5': 'large'    # Best accuracy, very slow
    }
    
    print("\nChoose Whisper model size:")
    print("1. Tiny (fastest)")
    print("2. Base (recommended for CPU)")
    print("3. Small (better accuracy)")
    print("4. Medium (high accuracy, slower)")
    print("5. Large (best accuracy, very slow)")
    
    choice = input("\nEnter choice (1-5) [default: 2]: ").strip()
    if choice not in model_choices:
        choice = '2'
    
    model_size = model_choices[choice]
    print(f"\nUsing {model_size} model...")
    
    # Initialize and start live captioning
    captioner = LiveCaptioning(model_size=model_size)
    captioner.start_live_captioning()

if __name__ == "__main__":
    main()