#!/usr/bin/env python3
"""
Push-to-talk Voice-to-Text Transcription
Press hotkey to start/stop recording, auto-types transcribed text at cursor
"""

import pyaudio
import numpy as np
from faster_whisper import WhisperModel
import subprocess
from pynput import keyboard
from pynput.keyboard import Controller, Key
import threading
import time

# Configuration
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
HOTKEY = keyboard.Key.f9

class DictationTranscriber:
    def __init__(self, model_size="base"):
        print(f"Loading Whisper {model_size} model...")
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        print("Model loaded!")
        
        self.is_recording = False
        self.audio_frames = []
        self.keyboard_controller = Controller()
        self.stream = None
        self.p = None
        self.recording_error = False
   
    def start_recording(self):
        """Start recording audio"""
        if self.is_recording:
            return

        self.is_recording = True
        self.audio_frames = []
        self.recording_error = False

        # Initialize PyAudio
        self.p = pyaudio.PyAudio()

        try:
            self.stream = self.p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=None,
                frames_per_buffer=CHUNK
            )

            # First beep to wake up Bluetooth (you won't hear this one)
            print("\n🔔 *beep* (waking up Bluetooth...)")
            subprocess.run(['paplay', '/usr/share/sounds/freedesktop/stereo/message.oga'],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Wait half a second
            time.sleep(0.5)

            # Second beep - you'll hear this one!
            print("🔔 *beep* Ready!")
            subprocess.run(['paplay', '/usr/share/sounds/freedesktop/stereo/message.oga'],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Discard 1 more second of audio to be safe
            warmup_chunks = int(RATE / CHUNK * 1.0)
            for _ in range(warmup_chunks):
                try:
                    self.stream.read(CHUNK, exception_on_overflow=False)
                except:
                    pass

            print("🔴 Recording NOW! Speak clearly... (Press F9 to stop)")

            # Now start actual recording thread
            record_thread = threading.Thread(target=self._record)
            record_thread.daemon = True
            record_thread.start()

        except Exception as e:
            print(f"❌ Error opening audio stream: {e}")
            self.is_recording = False
            if self.p:
                self.p.terminate()
    
    def _record(self):
        """Internal recording loop"""
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while self.is_recording:
            try:
                data = self.stream.read(CHUNK, exception_on_overflow=False)
                self.audio_frames.append(data)
                consecutive_errors = 0  # Reset error counter on success
            except OSError as e:
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    print(f"\n⚠️  Audio stream error (Bluetooth hiccup?). Try recording again.")
                    self.recording_error = True
                    self.is_recording = False
                    break
                time.sleep(0.01)  # Brief pause before retry
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
                self.recording_error = True
                self.is_recording = False
                break
    
    def stop_recording(self):
        """Stop recording and transcribe"""
        if not self.is_recording:
            return
            
        self.is_recording = False
        
        # Give recording thread a moment to finish
        time.sleep(0.1)
        
        print("⏹️  Recording stopped. Transcribing...")
        
        # Close audio stream
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
        except:
            pass
            
        if self.p:
            self.p.terminate()
        
        # Check for recording errors
        if self.recording_error:
            print("❌ Recording had errors. Please try again.")
            self.audio_frames = []
            return
        
        # Check if we have audio
        if not self.audio_frames:
            print("❌ No audio recorded")
            return
        
        # Transcribe
        try:
            # Convert frames to numpy array
            audio_data = b''.join(self.audio_frames)
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Check if audio has any content
            volume = np.abs(audio_np).mean()
            if volume < 0.001:  # Very quiet
                print("❌ No speech detected (too quiet)")
                return
            
            # Transcribe
            print(f"Processing {len(audio_np)/RATE:.1f} seconds of audio...")
            start_time = time.time()
            segments, info = self.model.transcribe(
                audio_np,
                beam_size=5,
                language="en",
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=300)
            )
            
            text = " ".join([segment.text.strip() for segment in segments])
            elapsed = time.time() - start_time
            
            if text:
                print(f"✅ Transcribed in {elapsed:.2f}s: {text}")
                # Type the text at cursor position
                self.type_text(text)
            else:
                print("❌ No speech detected in audio")
                
        except Exception as e:
            print(f"❌ Error during transcription: {e}")
            import traceback
            traceback.print_exc()
    
    def type_text(self, text):
        """Type the transcribed text at the current cursor position"""
        try:
            # Small delay to ensure cursor is ready
            time.sleep(0.1)
            self.keyboard_controller.type(text)
            print("⌨️  Text typed!")
        except Exception as e:
            print(f"❌ Error typing text: {e}")
            print(f"Text was: {text}")
    
    def on_hotkey(self, key):
        """Handle hotkey press"""
        try:
            if key == HOTKEY:
                if not self.is_recording:
                    self.start_recording()
                else:
                    self.stop_recording()
        except Exception as e:
            print(f"Error handling hotkey: {e}")
    
    def start(self):
        """Start listening for hotkey"""
        print("\n" + "="*60)
        print("🎙️  Voice-to-Text Dictation Ready")
        print("="*60)
        print(f"Hotkey: F9 (change HOTKEY variable to customize)")
        print(f"Model: Whisper base (CPU/int8)")
        print(f"Audio input: Device 15 (pulse) → Bluetooth Headset")
        print("\nUsage:")
        print("  1. Click where you want to type")
        print("  2. Press F9 (you'll hear a beep)")
        print("  3. Wait for 'Recording NOW!' message")
        print("  4. Speak your text clearly")
        print("  5. Press F9 again to stop and transcribe")
        print("  6. Text will be typed at your cursor\n")
        print("Tips:")
        print("  - The beep wakes up your Bluetooth headset")
        print("  - Wait for 'Recording NOW!' before speaking")
        print("  - Speak clearly and at normal pace")
        print("  - If you get audio errors, just try again\n")
        print("Press Ctrl+C to exit\n")
        
        # Start keyboard listener
        with keyboard.Listener(on_press=self.on_hotkey) as listener:
            try:
                listener.join()
            except KeyboardInterrupt:
                print("\n\n👋 Shutting down...")
                if self.is_recording:
                    self.stop_recording()

if __name__ == "__main__":
    transcriber = DictationTranscriber(model_size="small")
    transcriber.start()
