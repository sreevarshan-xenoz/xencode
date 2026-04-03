"""
Voice Interface Feature

Provides hands-free coding with voice commands, speech-to-text conversion,
text-to-speech responses, and visual feedback during voice interaction.
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import threading
import queue

from .base import FeatureBase, FeatureConfig, FeatureError


class VoiceStatus(Enum):
    """Voice interface status"""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    ERROR = "error"


class CommandType(Enum):
    """Types of voice commands"""
    CODE = "code"
    NAVIGATION = "navigation"
    SEARCH = "search"
    EDIT = "edit"
    SYSTEM = "system"
    QUERY = "query"


@dataclass
class VoiceInterfaceConfig:
    """Configuration for Voice Interface"""
    enabled: bool = True
    speech_to_text: Dict[str, Any] = field(default_factory=lambda: {
        'provider': 'whisper',
        'model': 'base',
        'language': 'en'
    })
    text_to_speech: Dict[str, Any] = field(default_factory=lambda: {
        'provider': 'pyttsx3',
        'voice': 'en-US',
        'rate': 150,
        'volume': 0.9
    })
    noise_reduction: bool = True
    silence_threshold: float = 0.5
    silence_duration: float = 1.0
    command_timeout: float = 5.0
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VoiceInterfaceConfig':
        """Create config from dictionary"""
        return cls(
            enabled=data.get('enabled', True),
            speech_to_text=data.get('speech_to_text', {
                'provider': 'whisper',
                'model': 'base',
                'language': 'en'
            }),
            text_to_speech=data.get('text_to_speech', {
                'provider': 'pyttsx3',
                'voice': 'en-US',
                'rate': 150,
                'volume': 0.9
            }),
            noise_reduction=data.get('noise_reduction', True),
            silence_threshold=data.get('silence_threshold', 0.5),
            silence_duration=data.get('silence_duration', 1.0),
            command_timeout=data.get('command_timeout', 5.0)
        )


@dataclass
class VoiceCommand:
    """Represents a voice command"""
    id: str
    text: str
    command_type: CommandType
    confidence: float
    timestamp: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'text': self.text,
            'command_type': self.command_type.value,
            'confidence': self.confidence,
            'timestamp': self.timestamp,
            'parameters': self.parameters
        }


@dataclass
class AudioLevel:
    """Audio level information"""
    current: float
    peak: float
    average: float
    is_speaking: bool


class VoiceInterfaceFeature(FeatureBase):
    """Voice Interface feature implementation"""
    
    def __init__(self, config: FeatureConfig):
        super().__init__(config)
        self.vi_config = VoiceInterfaceConfig.from_dict(config.config)
        self.speech_to_text = None
        self.text_to_speech = None
        self.command_processor = None
        self.feedback_system = None
        self.current_status = VoiceStatus.IDLE
        self.command_history: List[VoiceCommand] = []
        self.audio_level = AudioLevel(0.0, 0.0, 0.0, False)
        self._listening = False
        self._callbacks: Dict[str, List[Callable]] = {}
    
    @property
    def name(self) -> str:
        """Feature name"""
        return "voice_interface"
    
    @property
    def description(self) -> str:
        """Feature description"""
        return "Hands-free coding with voice commands and responses"
    
    async def _initialize(self) -> None:
        """Initialize Voice Interface components"""
        # Initialize speech-to-text
        self.speech_to_text = SpeechToText(
            provider=self.vi_config.speech_to_text['provider'],
            model=self.vi_config.speech_to_text['model'],
            language=self.vi_config.speech_to_text['language']
        )
        await self.speech_to_text.initialize()
        
        # Initialize text-to-speech
        self.text_to_speech = TextToSpeech(
            provider=self.vi_config.text_to_speech['provider'],
            voice=self.vi_config.text_to_speech['voice'],
            rate=self.vi_config.text_to_speech['rate'],
            volume=self.vi_config.text_to_speech['volume']
        )
        await self.text_to_speech.initialize()
        
        # Initialize command processor
        self.command_processor = VoiceCommandProcessor(
            noise_reduction=self.vi_config.noise_reduction,
            silence_threshold=self.vi_config.silence_threshold,
            silence_duration=self.vi_config.silence_duration
        )
        
        # Initialize feedback system
        self.feedback_system = VoiceFeedbackSystem()
    
    async def _shutdown(self) -> None:
        """Shutdown Voice Interface"""
        if self._listening:
            await self.stop_listening()
        
        if self.speech_to_text:
            await self.speech_to_text.shutdown()
        
        if self.text_to_speech:
            await self.text_to_speech.shutdown()
    
    async def listen(self, duration: float = None) -> Optional[str]:
        """
        Listen for voice input
        
        Args:
            duration: Optional duration to listen (None for continuous)
            
        Returns:
            Transcribed text or None
        """
        if not self.speech_to_text:
            raise FeatureError("Speech-to-text not initialized")
        
        self.current_status = VoiceStatus.LISTENING
        self._listening = True
        
        try:
            # Update feedback
            await self.feedback_system.update_status(VoiceStatus.LISTENING)
            
            # Listen for audio
            audio_data = await self._capture_audio(duration)
            
            if not audio_data:
                return None
            
            # Process audio
            self.current_status = VoiceStatus.PROCESSING
            await self.feedback_system.update_status(VoiceStatus.PROCESSING)
            
            # Transcribe
            text = await self.speech_to_text.transcribe(audio_data)
            
            # Track analytics
            self.track_analytics('listen', {
                'duration': duration,
                'text_length': len(text) if text else 0
            })
            
            return text
            
        except Exception as e:
            self.current_status = VoiceStatus.ERROR
            await self.feedback_system.update_status(VoiceStatus.ERROR, str(e))
            raise FeatureError(f"Failed to listen: {str(e)}")
        finally:
            self._listening = False
            if self.current_status != VoiceStatus.ERROR:
                self.current_status = VoiceStatus.IDLE
    
    async def speak(self, text: str, wait: bool = True) -> None:
        """
        Convert text to speech
        
        Args:
            text: Text to speak
            wait: Wait for speech to complete
        """
        if not self.text_to_speech:
            raise FeatureError("Text-to-speech not initialized")
        
        self.current_status = VoiceStatus.SPEAKING
        
        try:
            # Update feedback
            await self.feedback_system.update_status(VoiceStatus.SPEAKING, text)
            
            # Speak
            await self.text_to_speech.speak(text, wait=wait)
            
            # Track analytics
            self.track_analytics('speak', {
                'text_length': len(text),
                'wait': wait
            })
            
        except Exception as e:
            self.current_status = VoiceStatus.ERROR
            await self.feedback_system.update_status(VoiceStatus.ERROR, str(e))
            raise FeatureError(f"Failed to speak: {str(e)}")
        finally:
            if self.current_status != VoiceStatus.ERROR:
                self.current_status = VoiceStatus.IDLE
    
    async def process(self, text: str) -> VoiceCommand:
        """
        Process voice input into structured command
        
        Args:
            text: Voice input text
            
        Returns:
            Parsed voice command
        """
        if not self.command_processor:
            raise FeatureError("Command processor not initialized")
        
        try:
            # Process command
            command = await self.command_processor.process(text)
            
            # Add to history
            self.command_history.append(command)
            
            # Update feedback
            await self.feedback_system.show_command(command)
            
            # Track analytics
            self.track_analytics('process_command', {
                'command_type': command.command_type.value,
                'confidence': command.confidence
            })
            
            return command
            
        except Exception as e:
            raise FeatureError(f"Failed to process command: {str(e)}")
    
    async def start_listening(self) -> None:
        """Start continuous listening mode"""
        self._listening = True
        
        # Track analytics
        self.track_analytics('start_listening', {})
        
        # Notify callbacks
        await self._trigger_callbacks('start_listening')
    
    async def stop_listening(self) -> None:
        """Stop continuous listening mode"""
        self._listening = False
        self.current_status = VoiceStatus.IDLE
        
        # Track analytics
        self.track_analytics('stop_listening', {})
        
        # Notify callbacks
        await self._trigger_callbacks('stop_listening')
    
    def get_status(self) -> Dict[str, Any]:
        """Get current voice interface status"""
        return {
            'status': self.current_status.value,
            'listening': self._listening,
            'audio_level': {
                'current': self.audio_level.current,
                'peak': self.audio_level.peak,
                'average': self.audio_level.average,
                'is_speaking': self.audio_level.is_speaking
            },
            'command_count': len(self.command_history)
        }
    
    def get_command_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent command history"""
        return [cmd.to_dict() for cmd in self.command_history[-limit:]]
    
    def get_audio_level(self) -> AudioLevel:
        """Get current audio level"""
        return self.audio_level
    
    def register_callback(self, event: str, callback: Callable) -> None:
        """Register callback for voice events"""
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)
    
    async def _capture_audio(self, duration: float = None) -> Optional[bytes]:
        """Capture audio from microphone"""
        # Placeholder for audio capture
        # In production, use pyaudio or sounddevice
        await asyncio.sleep(duration or 0.1)
        return b"audio_data"
    
    async def _trigger_callbacks(self, event: str, *args, **kwargs) -> None:
        """Trigger registered callbacks"""
        if event in self._callbacks:
            for callback in self._callbacks[event]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(*args, **kwargs)
                    else:
                        callback(*args, **kwargs)
                except Exception:
                    pass
    
    def get_cli_commands(self) -> List[Any]:
        """Get CLI commands for Voice Interface"""
        import click
        
        @click.group(name='voice')
        def voice_group():
            """Voice interface commands - hands-free coding with voice"""
            pass
        
        @voice_group.command(name='start')
        @click.option('--duration', type=float, default=None, help='Recording duration (None for continuous)')
        @click.option('--silence-detection', is_flag=True, default=True, help='Use silence detection')
        def start_voice(duration, silence_detection):
            """Start voice interface listening mode"""
            import asyncio
            from rich.console import Console
            from rich.panel import Panel
            
            console = Console()
            console.print(Panel.fit("🎤 Starting Voice Interface", style="bold blue"))
            
            async def _start():
                try:
                    await self.start_listening()
                    console.print("[green]✅ Voice interface started[/green]")
                    console.print("[yellow]💡 Speak commands to interact with Xencode[/yellow]")
                    console.print("[dim]Press Ctrl+C to stop[/dim]")
                    
                    # Keep listening until interrupted
                    try:
                        while self._listening:
                            text = await self.listen(duration=duration)
                            if text:
                                console.print(f"\n[cyan]Recognized:[/cyan] {text}")
                                
                                # Process command
                                command = await self.process(text)
                                console.print(f"[yellow]Command Type:[/yellow] {command.command_type.value}")
                                console.print(f"[yellow]Confidence:[/yellow] {command.confidence:.0%}")
                                
                                # Speak confirmation
                                await self.speak(f"Understood: {text[:50]}", wait=False)
                            
                            await asyncio.sleep(0.1)
                    except KeyboardInterrupt:
                        console.print("\n[yellow]Stopping voice interface...[/yellow]")
                        await self.stop_listening()
                        
                except Exception as e:
                    console.print(f"[red]❌ Failed to start voice interface: {e}[/red]")
            
            asyncio.run(_start())
        
        @voice_group.command(name='stop')
        def stop_voice():
            """Stop voice interface listening mode"""
            import asyncio
            from rich.console import Console
            
            console = Console()
            
            async def _stop():
                try:
                    await self.stop_listening()
                    console.print("[green]✅ Voice interface stopped[/green]")
                except Exception as e:
                    console.print(f"[red]❌ Failed to stop voice interface: {e}[/red]")
            
            asyncio.run(_stop())
        
        @voice_group.command(name='commands')
        @click.option('--limit', type=int, default=10, help='Number of recent commands to show')
        def show_commands(limit):
            """Show recent voice commands"""
            from rich.console import Console
            from rich.table import Table
            
            console = Console()
            
            history = self.get_command_history(limit=limit)
            
            if not history:
                console.print("[yellow]No voice commands in history[/yellow]")
                return
            
            table = Table(title=f"Recent Voice Commands (Last {len(history)})")
            table.add_column("Time", style="cyan")
            table.add_column("Command", style="white")
            table.add_column("Type", style="yellow")
            table.add_column("Confidence", style="green")
            
            for cmd in history:
                from datetime import datetime
                try:
                    dt = datetime.fromisoformat(cmd['timestamp'])
                    time_str = dt.strftime("%H:%M:%S")
                except:
                    time_str = cmd['timestamp']
                
                confidence_str = f"{cmd['confidence']:.0%}"
                table.add_row(
                    time_str,
                    cmd['text'][:50],
                    cmd['command_type'],
                    confidence_str
                )
            
            console.print(table)
        
        @voice_group.command(name='status')
        def show_status():
            """Show voice interface status"""
            from rich.console import Console
            from rich.panel import Panel
            from rich.table import Table
            
            console = Console()
            
            status = self.get_status()
            
            # Status panel
            status_text = f"""
[cyan]Status:[/cyan] {status['status']}
[cyan]Listening:[/cyan] {'✅ Yes' if status['listening'] else '❌ No'}
[cyan]Commands:[/cyan] {status['command_count']}
            """
            
            console.print(Panel(status_text.strip(), title="🎤 Voice Interface Status", border_style="blue"))
            
            # Audio level
            audio = status['audio_level']
            console.print("\n[bold]Audio Level:[/bold]")
            console.print(f"  Current: {audio['current']:.2f}")
            console.print(f"  Peak: {audio['peak']:.2f}")
            console.print(f"  Average: {audio['average']:.2f}")
            console.print(f"  Speaking: {'✅ Yes' if audio['is_speaking'] else '❌ No'}")
        
        @voice_group.command(name='test')
        @click.option('--text', default='Hello, this is a voice interface test', help='Text to speak')
        def test_voice(text):
            """Test voice interface (text-to-speech)"""
            import asyncio
            from rich.console import Console
            
            console = Console()
            console.print("[blue]🔊 Testing voice interface...[/blue]")
            
            async def _test():
                try:
                    await self.speak(text, wait=True)
                    console.print("[green]✅ Voice test completed[/green]")
                except Exception as e:
                    console.print(f"[red]❌ Voice test failed: {e}[/red]")
            
            asyncio.run(_test())
        
        @voice_group.command(name='settings')
        @click.option('--show', is_flag=True, help='Show current settings')
        @click.option('--set-language', help='Set speech language (e.g., en, es, fr)')
        @click.option('--set-voice', help='Set TTS voice')
        @click.option('--set-rate', type=int, help='Set TTS rate (words per minute)')
        @click.option('--set-volume', type=float, help='Set TTS volume (0.0-1.0)')
        def configure_settings(show, set_language, set_voice, set_rate, set_volume):
            """Configure voice interface settings"""
            from rich.console import Console
            from rich.table import Table
            
            console = Console()
            
            if show:
                # Show current settings
                table = Table(title="Voice Interface Settings")
                table.add_column("Setting", style="cyan")
                table.add_column("Value", style="yellow")
                
                table.add_row("STT Provider", self.vi_config.speech_to_text['provider'])
                table.add_row("STT Model", self.vi_config.speech_to_text['model'])
                table.add_row("STT Language", self.vi_config.speech_to_text['language'])
                table.add_row("TTS Provider", self.vi_config.text_to_speech['provider'])
                table.add_row("TTS Voice", self.vi_config.text_to_speech['voice'])
                table.add_row("TTS Rate", str(self.vi_config.text_to_speech['rate']))
                table.add_row("TTS Volume", str(self.vi_config.text_to_speech['volume']))
                table.add_row("Noise Reduction", "✅ Enabled" if self.vi_config.noise_reduction else "❌ Disabled")
                table.add_row("Silence Threshold", str(self.vi_config.silence_threshold))
                table.add_row("Silence Duration", f"{self.vi_config.silence_duration}s")
                
                console.print(table)
            else:
                # Update settings
                updated = False
                
                if set_language:
                    self.vi_config.speech_to_text['language'] = set_language
                    console.print(f"[green]✅ Language set to: {set_language}[/green]")
                    updated = True
                
                if set_voice:
                    self.vi_config.text_to_speech['voice'] = set_voice
                    console.print(f"[green]✅ Voice set to: {set_voice}[/green]")
                    updated = True
                
                if set_rate:
                    self.vi_config.text_to_speech['rate'] = set_rate
                    console.print(f"[green]✅ Rate set to: {set_rate}[/green]")
                    updated = True
                
                if set_volume is not None:
                    self.vi_config.text_to_speech['volume'] = max(0.0, min(1.0, set_volume))
                    console.print(f"[green]✅ Volume set to: {self.vi_config.text_to_speech['volume']}[/green]")
                    updated = True
                
                if not updated:
                    console.print("[yellow]No settings changed. Use --show to view current settings.[/yellow]")
        
        return [voice_group]
    
    def get_tui_components(self) -> List[Any]:
        """Get TUI components for Voice Interface"""
        from xencode.tui.widgets.voice_interface_panel import VoiceInterfacePanel
        return [VoiceInterfacePanel]
    
    def get_api_endpoints(self) -> List[Any]:
        """Get API endpoints for Voice Interface"""
        return [
            {
                'path': '/api/voice/listen',
                'method': 'POST',
                'handler': self.listen
            },
            {
                'path': '/api/voice/speak',
                'method': 'POST',
                'handler': self.speak
            },
            {
                'path': '/api/voice/process',
                'method': 'POST',
                'handler': self.process
            },
            {
                'path': '/api/voice/status',
                'method': 'GET',
                'handler': self.get_status
            },
            {
                'path': '/api/voice/history',
                'method': 'GET',
                'handler': self.get_command_history
            }
        ]


class SpeechToText:
    """Speech-to-text conversion using Whisper"""
    
    def __init__(self, provider: str = 'whisper', model: str = 'base', language: str = 'en'):
        self.provider = provider
        self.model_name = model
        self.language = language
        self.model = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize speech-to-text model"""
        if self.provider == 'whisper':
            try:
                # Try to import whisper
                import whisper
                self.model = whisper.load_model(self.model_name)
                self._initialized = True
            except ImportError:
                # Whisper not available, use mock
                self.model = None
                self._initialized = True
        else:
            raise FeatureError(f"Unsupported STT provider: {self.provider}")
    
    async def transcribe(self, audio_data: bytes) -> str:
        """
        Transcribe audio to text
        
        Args:
            audio_data: Audio data bytes
            
        Returns:
            Transcribed text
        """
        if not self._initialized:
            raise FeatureError("Speech-to-text not initialized")
        
        if self.model is None:
            # Mock transcription for testing
            return "mock transcription"
        
        try:
            # In production, save audio_data to temp file and transcribe
            # result = self.model.transcribe(audio_file, language=self.language)
            # return result['text']
            return "transcribed text"
        except Exception as e:
            raise FeatureError(f"Transcription failed: {str(e)}")
    
    async def shutdown(self) -> None:
        """Shutdown speech-to-text"""
        self.model = None
        self._initialized = False


class TextToSpeech:
    """Text-to-speech conversion"""
    
    def __init__(self, provider: str = 'pyttsx3', voice: str = 'en-US', 
                 rate: int = 150, volume: float = 0.9):
        self.provider = provider
        self.voice = voice
        self.rate = rate
        self.volume = volume
        self.engine = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize text-to-speech engine"""
        if self.provider == 'pyttsx3':
            try:
                import pyttsx3
                self.engine = pyttsx3.init()
                self.engine.setProperty('rate', self.rate)
                self.engine.setProperty('volume', self.volume)
                
                # Set voice if available
                voices = self.engine.getProperty('voices')
                for v in voices:
                    if self.voice in v.id or self.voice in v.name:
                        self.engine.setProperty('voice', v.id)
                        break
                
                self._initialized = True
            except ImportError:
                # pyttsx3 not available, use mock
                self.engine = None
                self._initialized = True
        else:
            raise FeatureError(f"Unsupported TTS provider: {self.provider}")
    
    async def speak(self, text: str, wait: bool = True) -> None:
        """
        Convert text to speech
        
        Args:
            text: Text to speak
            wait: Wait for speech to complete
        """
        if not self._initialized:
            raise FeatureError("Text-to-speech not initialized")
        
        if self.engine is None:
            # Mock speech for testing
            await asyncio.sleep(len(text) * 0.05)  # Simulate speech duration
            return
        
        try:
            self.engine.say(text)
            if wait:
                self.engine.runAndWait()
        except Exception as e:
            raise FeatureError(f"Speech failed: {str(e)}")
    
    async def shutdown(self) -> None:
        """Shutdown text-to-speech"""
        if self.engine:
            try:
                self.engine.stop()
            except Exception:
                pass
        self.engine = None
        self._initialized = False


class VoiceCommandProcessor:
    """Processes voice commands into structured actions"""
    
    def __init__(self, noise_reduction: bool = True, 
                 silence_threshold: float = 0.5,
                 silence_duration: float = 1.0):
        self.noise_reduction = noise_reduction
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.command_patterns = self._load_command_patterns()
    
    def _load_command_patterns(self) -> Dict[CommandType, List[str]]:
        """Load command patterns for recognition"""
        return {
            CommandType.CODE: [
                'write', 'create', 'generate', 'code', 'function', 'class'
            ],
            CommandType.NAVIGATION: [
                'go to', 'open', 'navigate', 'show', 'find'
            ],
            CommandType.SEARCH: [
                'search', 'find', 'look for', 'where is'
            ],
            CommandType.EDIT: [
                'edit', 'change', 'modify', 'update', 'delete', 'remove'
            ],
            CommandType.SYSTEM: [
                'save', 'close', 'exit', 'quit', 'help'
            ],
            CommandType.QUERY: [
                'what', 'how', 'why', 'explain', 'tell me'
            ]
        }
    
    async def process(self, text: str) -> VoiceCommand:
        """
        Process voice text into structured command
        
        Args:
            text: Voice input text
            
        Returns:
            Structured voice command
        """
        # Clean text
        cleaned_text = self._clean_text(text)
        
        # Detect command type
        command_type = self._detect_command_type(cleaned_text)
        
        # Extract parameters
        parameters = self._extract_parameters(cleaned_text, command_type)
        
        # Calculate confidence
        confidence = self._calculate_confidence(cleaned_text, command_type)
        
        # Create command
        command = VoiceCommand(
            id=self._generate_command_id(),
            text=cleaned_text,
            command_type=command_type,
            confidence=confidence,
            timestamp=datetime.now().isoformat(),
            parameters=parameters
        )
        
        return command
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Convert to lowercase for processing
        text = text.lower()
        
        return text
    
    def _detect_command_type(self, text: str) -> CommandType:
        """Detect command type from text"""
        # Check each command type pattern
        for cmd_type, patterns in self.command_patterns.items():
            for pattern in patterns:
                if pattern in text:
                    return cmd_type
        
        # Default to query
        return CommandType.QUERY
    
    def _extract_parameters(self, text: str, command_type: CommandType) -> Dict[str, Any]:
        """Extract parameters from command text"""
        parameters = {}
        
        # Extract based on command type
        if command_type == CommandType.CODE:
            # Extract code-related parameters
            if 'function' in text:
                parameters['entity_type'] = 'function'
            elif 'class' in text:
                parameters['entity_type'] = 'class'
        
        elif command_type == CommandType.NAVIGATION:
            # Extract navigation target
            words = text.split()
            if 'to' in words:
                idx = words.index('to')
                if idx + 1 < len(words):
                    parameters['target'] = ' '.join(words[idx+1:])
        
        return parameters
    
    def _calculate_confidence(self, text: str, command_type: CommandType) -> float:
        """Calculate confidence score for command"""
        # Simple confidence based on pattern matching
        patterns = self.command_patterns.get(command_type, [])
        matches = sum(1 for pattern in patterns if pattern in text)
        
        if matches > 0:
            return min(0.7 + (matches * 0.1), 1.0)
        return 0.5
    
    def _generate_command_id(self) -> str:
        """Generate unique command ID"""
        import uuid
        return str(uuid.uuid4())[:8]


class VoiceFeedbackSystem:
    """Provides visual feedback during voice interaction"""
    
    def __init__(self):
        self.current_status = VoiceStatus.IDLE
        self.status_message = ""
        self.recognized_commands: List[VoiceCommand] = []
        self.responses: List[str] = []
    
    async def update_status(self, status: VoiceStatus, message: str = "") -> None:
        """Update voice status"""
        self.current_status = status
        self.status_message = message
    
    async def show_command(self, command: VoiceCommand) -> None:
        """Show recognized command"""
        self.recognized_commands.append(command)
    
    async def show_response(self, response: str) -> None:
        """Show voice response"""
        self.responses.append(response)
    
    def get_feedback(self) -> Dict[str, Any]:
        """Get current feedback state"""
        return {
            'status': self.current_status.value,
            'message': self.status_message,
            'recent_commands': [cmd.to_dict() for cmd in self.recognized_commands[-5:]],
            'recent_responses': self.responses[-5:]
        }
