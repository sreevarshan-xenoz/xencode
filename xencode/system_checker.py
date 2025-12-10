import platform
import subprocess
import sys
import shutil
from typing import Dict, Any, Optional, Tuple

class SystemChecker:
    """
    Identifies the system environment (OS, Distro, Python version, etc.)
    and checks for necessary dependencies like Ollama.
    """

    def __init__(self):
        self.system_info = self._gather_system_info()

    def _gather_system_info(self) -> Dict[str, Any]:
        """Gathers detailed system information."""
        info = {
            "os": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": sys.version,
            "is_windows": platform.system() == "Windows",
            "is_linux": platform.system() == "Linux",
            "is_macos": platform.system() == "Darwin",
            "distro": None,
            "ollama_installed": False,
            "ollama_running": False
        }

        if info["is_linux"]:
            try:
                # Try to get distro info
                import distro
                info["distro"] = {
                    "id": distro.id(),
                    "name": distro.name(),
                    "version": distro.version(),
                    "like": distro.like()
                }
            except ImportError:
                # Fallback if distro package is not installed
                try:
                    with open("/etc/os-release") as f:
                        lines = f.readlines()
                        distro_info = {}
                        for line in lines:
                            if "=" in line:
                                k, v = line.strip().split("=", 1)
                                distro_info[k] = v.strip('"')
                        info["distro"] = {
                            "id": distro_info.get("ID"),
                            "name": distro_info.get("PRETTY_NAME"),
                            "version": distro_info.get("VERSION_ID"),
                        }
                except Exception:
                    info["distro"] = "Unknown Linux Distro"

        # Check for Ollama
        info["ollama_installed"] = shutil.which("ollama") is not None
        
        if info["ollama_installed"]:
            try:
                # Check if ollama is actually running by trying to list models or hit the API
                # Using a quick timeout
                import requests
                try:
                    response = requests.get("http://localhost:11434/api/tags", timeout=2)
                    if response.status_code == 200:
                        info["ollama_running"] = True
                except requests.RequestException:
                    info["ollama_running"] = False
            except ImportError:
                # If requests is not installed, we can't easily check API, 
                # but we can try subprocess if we wanted to be very thorough.
                # For now, assume if installed it might be running, or let the app handle connection errors.
                pass

        return info
    
    def ensure_ollama_available(self, auto_start: bool = True) -> Tuple[bool, str]:
        """
        Ensure Ollama is available, optionally starting it if not running.
        
        Args:
            auto_start: If True, attempt to start Ollama if not running
            
        Returns:
            Tuple of (success, message)
        """
        try:
            from .ollama_fallback import OllamaFallbackManager
            fallback = OllamaFallbackManager(auto_start=auto_start, auto_install=False)
            return fallback.ensure_ollama_available()
        except ImportError:
            # Fallback if ollama_fallback module not available
            if not self.system_info.get("ollama_installed"):
                return False, "Ollama is not installed"
            if not self.system_info.get("ollama_running"):
                return False, "Ollama is not running"
            return True, "Ollama is running"

    def get_info(self) -> Dict[str, Any]:
        """Returns the gathered system information."""
        return self.system_info

    def print_status(self):
        """Prints a formatted status report to the console."""
        print("=" * 40)
        print("🔍 System Check Report")
        print("=" * 40)
        print(f"OS: {self.system_info['os']} {self.system_info['release']}")
        if self.system_info['is_linux'] and self.system_info['distro']:
            d = self.system_info['distro']
            if isinstance(d, dict):
                 print(f"Distro: {d.get('name', 'Unknown')} ({d.get('version', '')})")
            else:
                 print(f"Distro: {d}")
        
        print(f"Python: {platform.python_version()}")
        
        ollama_status = "✅ Installed" if self.system_info['ollama_installed'] else "❌ Not Found"
        print(f"Ollama CLI: {ollama_status}")
        
        if self.system_info['ollama_installed']:
            running_status = "✅ Running" if self.system_info['ollama_running'] else "❌ Not Running (or not accessible)"
            print(f"Ollama Service: {running_status}")
        
        print("=" * 40)

if __name__ == "__main__":
    checker = SystemChecker()
    checker.print_status()
