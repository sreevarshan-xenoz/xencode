#!/usr/bin/env python3
"""
Voice Input with Whisper MVP for Xencode

Provides push-to-talk voice transcription with:
- Local Whisper model support
- Edit confirmation before submit
- Command grammar for shortcuts
- Audio recording and processing
"""

import asyncio
import io
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable

from rich.console import Console

console = Console()

# Check for whisper availability
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

# Check for sounddevice (audio recording)
try:
    import sounddevice as sd
    import numpy as np
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    sd = None
    np = None


class VoiceCommand(Enum):
    """Voice command shortcuts"""
    SUBMIT = "submit"
    CANCEL = "cancel"
    CLEAR = "clear"
    HELP = "help"
    EXPLAIN = "explain"
    REFACTOR = "refactor"
    TEST = "test"


@dataclass
class VoiceInput:
    """Voice input data structure"""
    text: str
    confidence: float = 1.0
    language: str = "en"
    duration: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    detected_commands: List[VoiceCommand] = field(default_factory=list)
    raw_audio: Optional[bytes] = None
    
    @property
    def is_command(self) -> bool:
        """Check if input is a command"""
        return len(self.detected_commands) > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "text": self.text,
            "confidence": self.confidence,
            "language": self.language,
            "duration": self.duration,
            "timestamp": self.timestamp.isoformat(),
            "detected_commands": [c.value for c in self.detected_commands],
            "is_command": self.is_command,
        }


@dataclass
class VoiceConfig:
    """Voice input configuration"""
    model_size: str = "base"  # tiny, base, small, medium, large
    language: Optional[str] = None  # None = auto-detect
    recording_duration: float = 10.0  # max seconds
    sample_rate: int = 16000
    silence_threshold: float = 0.01
    command_mode: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "model_size": self.model_size,
            "language": self.language,
            "recording_duration": self.recording_duration,
            "sample_rate": self.sample_rate,
            "silence_threshold": self.silence_threshold,
            "command_mode": self.command_mode,
        }


class VoiceTranscriber:
    """
    Local Whisper-based voice transcriber
    
    Usage:
        transcriber = VoiceTranscriber()
        result = await transcriber.transcribe(audio_data)
    """
    
    def __init__(self, config: Optional[VoiceConfig] = None):
        """
        Initialize voice transcriber
        
        Args:
            config: Voice configuration
        """
        self.config = config or VoiceConfig()
        self._model = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize Whisper model"""
        if self._initialized:
            return
        
        if not WHISPER_AVAILABLE:
            console.print("[yellow]Warning: Whisper not installed. Run: pip install openai-whisper[/yellow]")
            return
        
        try:
            console.print(f"[dim]Loading Whisper model: {self.config.model_size}...[/dim]")
            self._model = whisper.load_model(self.config.model_size)
            self._initialized = True
            console.print("[green]✓ Whisper model loaded[/green]")
        except Exception as e:
            console.print(f"[red]Failed to load Whisper model: {e}[/red]")
    
    async def transcribe(
        self,
        audio_data: bytes,
        language: Optional[str] = None,
    ) -> VoiceInput:
        """
        Transcribe audio to text
        
        Args:
            audio_data: Raw audio bytes (16kHz, 16-bit mono PCM)
            language: Optional language code
            
        Returns:
            VoiceInput with transcribed text
        """
        if not self._initialized:
            await self.initialize()
        
        if not self._model:
            return VoiceInput(
                text="",
                confidence=0.0,
                detected_commands=[],
            )
        
        try:
            # Convert bytes to numpy array
            if np is None:
                return VoiceInput(text="", confidence=0.0)
            
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Transcribe
            result = self._model.transcribe(
                audio_array,
                language=language or self.config.language,
            )
            
            text = result.get("text", "").strip()
            confidence = result.get("segments", [{}])[0].get("avg_logprob", 0.0)
            
            # Detect commands
            commands = self._detect_commands(text)
            
            return VoiceInput(
                text=text,
                confidence=confidence,
                language=language or self.config.language or "en",
                detected_commands=commands,
                raw_audio=audio_data,
            )
            
        except Exception as e:
            console.print(f"[red]Transcription error: {e}[/red]")
            return VoiceInput(
                text="",
                confidence=0.0,
                detected_commands=[],
            )
    
    def _detect_commands(self, text: str) -> List[VoiceCommand]:
        """Detect voice commands in transcribed text"""
        if not self.config.command_mode:
            return []
        
        commands = []
        text_lower = text.lower()
        
        command_map = {
            "submit": ["submit", "send", "go ahead", "do it"],
            "cancel": ["cancel", "nevermind", "stop", "abort"],
            "clear": ["clear", "delete", "erase"],
            "help": ["help", "what can you do", "commands"],
            "explain": ["explain this", "explain code"],
            "refactor": ["refactor", "improve code", "clean up"],
            "test": ["write tests", "generate tests", "test this"],
        }
        
        for command, keywords in command_map.items():
            if any(keyword in text_lower for keyword in keywords):
                try:
                    commands.append(VoiceCommand(command))
                except ValueError:
                    pass
        
        return commands


class AudioRecorder:
    """
    Audio recording utility
    
    Usage:
        recorder = AudioRecorder()
        audio = await recorder.record(duration=5.0)
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
    ):
        """
        Initialize audio recorder
        
        Args:
            sample_rate: Audio sample rate (Hz)
            channels: Number of audio channels
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self._recording = False
    
    async def record(self, duration: float = 10.0) -> Optional[bytes]:
        """
        Record audio for specified duration
        
        Args:
            duration: Recording duration in seconds
            
        Returns:
            Raw audio bytes or None if recording failed
        """
        if not AUDIO_AVAILABLE:
            console.print("[yellow]Audio recording not available. Install: pip install sounddevice numpy[/yellow]")
            return None
        
        try:
            console.print(f"[dim]Recording for {duration} seconds...[/dim]")
            
            # Record audio
            recording = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.int16,
            )
            
            self._recording = True
            
            # Wait for recording to complete
            await asyncio.sleep(duration)
            sd.wait()
            
            self._recording = False
            
            # Convert to bytes
            audio_bytes = recording.tobytes()
            
            console.print("[green]✓ Recording complete[/green]")
            return audio_bytes
            
        except Exception as e:
            console.print(f"[red]Recording error: {e}[/red]")
            self._recording = False
            return None
    
    async def record_until_silence(
        self,
        max_duration: float = 30.0,
        silence_duration: float = 2.0,
    ) -> Optional[bytes]:
        """
        Record until silence detected
        
        Args:
            max_duration: Maximum recording duration
            silence_duration: Silence duration to stop recording
            
        Returns:
            Raw audio bytes or None
        """
        if not AUDIO_AVAILABLE:
            return None
        
        try:
            console.print("[dim]Recording... (stop speaking to finish)[/dim]")
            
            all_audio = []
            elapsed = 0.0
            chunk_duration = 0.5  # Process in 0.5s chunks
            silence_start = None
            
            while elapsed < max_duration:
                # Record chunk
                chunk = sd.rec(
                    int(chunk_duration * self.sample_rate),
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype=np.int16,
                )
                sd.wait()
                
                all_audio.append(chunk.copy())
                elapsed += chunk_duration
                
                # Check for silence
                rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))
                
                if rms < 100:  # Silence threshold
                    if silence_start is None:
                        silence_start = elapsed
                    elif elapsed - silence_start >= silence_duration:
                        console.print("[green]✓ Silence detected, stopping[/green]")
                        break
                else:
                    silence_start = None
            
            # Combine all audio
            if all_audio:
                full_recording = np.concatenate(all_audio)
                return full_recording.tobytes()
            
            return None
            
        except Exception as e:
            console.print(f"[red]Recording error: {e}[/red]")
            return None
    
    def is_recording(self) -> bool:
        """Check if currently recording"""
        return self._recording


class VoiceInputPanel:
    """
    Voice input panel for TUI
    
    Integrates transcription with edit confirmation
    
    Usage:
        panel = VoiceInputPanel()
        result = await panel.capture_and_confirm()
    """
    
    def __init__(
        self,
        config: Optional[VoiceConfig] = None,
        confirm_callback: Optional[Callable] = None,
    ):
        """
        Initialize voice input panel
        
        Args:
            config: Voice configuration
            confirm_callback: Callback for confirmation UI
        """
        self.config = config or VoiceConfig()
        self.confirm_callback = confirm_callback
        self.transcriber = VoiceTranscriber(self.config)
        self.recorder = AudioRecorder(sample_rate=self.config.sample_rate)
        self._history: List[VoiceInput] = []
    
    async def capture_and_confirm(
        self,
        use_silence_detection: bool = True,
    ) -> Optional[VoiceInput]:
        """
        Capture voice input and confirm before submit
        
        Args:
            use_silence_detection: Use silence detection vs fixed duration
            
        Returns:
            Confirmed VoiceInput or None if cancelled
        """
        # Record audio
        if use_silence_detection:
            audio = await self.recorder.record_until_silence(
                max_duration=self.config.recording_duration,
            )
        else:
            audio = await self.recorder.record(self.config.recording_duration)
        
        if not audio:
            return None
        
        # Transcribe
        voice_input = await self.transcriber.transcribe(audio)
        
        if not voice_input.text:
            console.print("[yellow]No speech detected[/yellow]")
            return None
        
        # Show for confirmation
        console.print("\n[bold]Transcribed text:[/bold]")
        console.print(f"  {voice_input.text}")
        
        if voice_input.detected_commands:
            console.print(f"\n[yellow]Commands detected: {[c.value for c in voice_input.detected_commands]}[/yellow]")
        
        # Confirm (would show UI in TUI)
        if self.confirm_callback:
            confirmed = await self.confirm_callback(voice_input)
            if not confirmed:
                console.print("[yellow]Cancelled[/yellow]")
                return None
        
        self._history.append(voice_input)
        return voice_input
    
    async def quick_capture(self) -> Optional[VoiceInput]:
        """
        Quick capture without confirmation
        
        Returns:
            VoiceInput or None
        """
        audio = await self.recorder.record(5.0)
        
        if not audio:
            return None
        
        return await self.transcriber.transcribe(audio)
    
    def get_history(self, count: int = 10) -> List[VoiceInput]:
        """Get recent voice inputs"""
        return self._history[-count:]
    
    def clear_history(self):
        """Clear voice input history"""
        self._history = []


# Global panel instance
_panel: Optional[VoiceInputPanel] = None


def get_voice_panel(config: Optional[VoiceConfig] = None) -> VoiceInputPanel:
    """Get or create global voice panel"""
    global _panel
    if _panel is None:
        _panel = VoiceInputPanel(config=config)
    return _panel


# Convenience functions
async def capture_voice_input(
    confirm: bool = True,
    use_silence_detection: bool = True,
) -> Optional[VoiceInput]:
    """
    Capture voice input
    
    Args:
        confirm: Require confirmation before submit
        use_silence_detection: Use silence detection
        
    Returns:
        VoiceInput or None
    """
    panel = get_voice_panel()
    
    if confirm:
        return await panel.capture_and_confirm(use_silence_detection)
    else:
        return await panel.quick_capture()


async def transcribe_audio(audio_data: bytes) -> VoiceInput:
    """
    Transcribe audio data
    
    Args:
        audio_data: Raw audio bytes
        
    Returns:
        VoiceInput with transcription
    """
    transcriber = VoiceTranscriber()
    return await transcriber.transcribe(audio_data)


def is_voice_available() -> bool:
    """Check if voice input is available"""
    return WHISPER_AVAILABLE and AUDIO_AVAILABLE


if __name__ == "__main__":
    # Demo
    async def demo():
        console.print("[bold blue]Voice Input Demo (Whisper MVP)[/bold blue]\n")
        
        console.print(f"Whisper available: {WHISPER_AVAILABLE}")
        console.print(f"Audio recording available: {AUDIO_AVAILABLE}")
        
        if not is_voice_available():
            console.print("\n[yellow]Voice input requires:[/yellow]")
            if not WHISPER_AVAILABLE:
                console.print("  pip install openai-whisper")
            if not AUDIO_AVAILABLE:
                console.print("  pip install sounddevice numpy")
            return
        
        # Demo transcription with simulated audio
        console.print("\n[bold]Demo mode (no actual recording):[/bold]")
        
        transcriber = VoiceTranscriber()
        await transcriber.initialize()
        
        # Simulate transcription
        if np:
            # Create silent audio for demo
            silent_audio = np.zeros(16000, dtype=np.int16).tobytes()
            result = await transcriber.transcribe(silent_audio)
            console.print(f"Transcription result: {result.to_dict()}")
    
    asyncio.run(demo())
