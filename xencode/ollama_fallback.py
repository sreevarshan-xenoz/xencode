#!/usr/bin/env python3
"""
Ollama Fallback Manager for Xencode

Provides intelligent fallback mechanisms for Ollama:
1. Checks if Ollama service is running
2. Attempts to start Ollama if not running
3. Offers to install Ollama (with user consent) if not installed
"""

import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.prompt import Confirm, Prompt
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Initialize console
console = Console() if RICH_AVAILABLE else None


class OSType(Enum):
    """Supported operating systems"""
    WINDOWS = "windows"
    LINUX = "linux"
    MACOS = "macos"
    UNKNOWN = "unknown"


class PackageManager(Enum):
    """Known package managers"""
    WINGET = "winget"
    CHOCOLATEY = "chocolatey"
    APT = "apt"
    PACMAN = "pacman"
    DNF = "dnf"
    YUM = "yum"
    ZYPPER = "zypper"
    BREW = "brew"
    CURL = "curl"  # Fallback installer
    NONE = "none"


@dataclass
class OSInfo:
    """Operating system information"""
    os_type: OSType
    name: str
    version: str
    distro: Optional[str] = None
    distro_like: Optional[str] = None
    package_manager: PackageManager = PackageManager.NONE


class OllamaFallbackManager:
    """
    Manages Ollama service startup and installation with OS-aware logic.
    
    Usage:
        fallback = OllamaFallbackManager()
        success, message = fallback.ensure_ollama_available()
    """
    
    OLLAMA_API_URL = "http://localhost:11434/api/tags"
    OLLAMA_DOWNLOAD_URL = "https://ollama.ai/download"
    OLLAMA_INSTALL_SCRIPT = "https://ollama.ai/install.sh"
    
    def __init__(self, auto_start: bool = True, auto_install: bool = False):
        """
        Initialize the fallback manager.
        
        Args:
            auto_start: If True, silently try to start Ollama service
            auto_install: If True, don't prompt for install confirmation (dangerous!)
        """
        self.auto_start = auto_start
        self.auto_install = auto_install
        self.os_info = self._detect_os()
        self._start_process: Optional[subprocess.Popen] = None
    
    def _print(self, message: str, style: str = ""):
        """Print message using Rich if available, else plain print"""
        if RICH_AVAILABLE and console:
            if style:
                console.print(f"[{style}]{message}[/{style}]")
            else:
                console.print(message)
        else:
            print(message)
    
    def _print_panel(self, message: str, title: str = "", style: str = "blue"):
        """Print a panel using Rich if available"""
        if RICH_AVAILABLE and console:
            console.print(Panel(message, title=title, style=style))
        else:
            print(f"\n{'='*50}")
            if title:
                print(f" {title}")
                print(f"{'='*50}")
            print(message)
            print(f"{'='*50}\n")
    
    def _confirm(self, message: str, default: bool = False) -> bool:
        """Ask user for confirmation"""
        if RICH_AVAILABLE:
            return Confirm.ask(message, default=default)
        else:
            response = input(f"{message} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
            if not response:
                return default
            return response in ('y', 'yes')
    
    def _select_option(self, message: str, options: Dict[str, str]) -> Optional[str]:
        """Let user select from options"""
        self._print(f"\n{message}")
        for key, desc in options.items():
            self._print(f"  [{key}] {desc}")
        
        if RICH_AVAILABLE:
            choice = Prompt.ask("Select option", choices=list(options.keys()) + ['q'], default='q')
        else:
            choice = input(f"Select option [{'/'.join(options.keys())}/q]: ").strip().lower()
        
        return choice if choice in options else None
    
    def _detect_os(self) -> OSInfo:
        """Detect operating system and available package manager"""
        system = platform.system().lower()
        
        if system == "windows":
            os_type = OSType.WINDOWS
            name = "Windows"
            version = platform.version()
            pkg_mgr = self._detect_windows_package_manager()
            return OSInfo(os_type, name, version, package_manager=pkg_mgr)
        
        elif system == "darwin":
            os_type = OSType.MACOS
            name = "macOS"
            version = platform.mac_ver()[0]
            pkg_mgr = PackageManager.BREW if shutil.which("brew") else PackageManager.CURL
            return OSInfo(os_type, name, version, package_manager=pkg_mgr)
        
        elif system == "linux":
            os_type = OSType.LINUX
            name = "Linux"
            version = platform.release()
            distro, distro_like = self._detect_linux_distro()
            pkg_mgr = self._detect_linux_package_manager(distro, distro_like)
            return OSInfo(os_type, name, version, distro, distro_like, pkg_mgr)
        
        else:
            return OSInfo(OSType.UNKNOWN, system, "", package_manager=PackageManager.CURL)
    
    def _detect_windows_package_manager(self) -> PackageManager:
        """Detect available Windows package manager"""
        if shutil.which("winget"):
            return PackageManager.WINGET
        elif shutil.which("choco"):
            return PackageManager.CHOCOLATEY
        return PackageManager.NONE
    
    def _detect_linux_distro(self) -> Tuple[Optional[str], Optional[str]]:
        """Detect Linux distribution"""
        distro = None
        distro_like = None
        
        # Try /etc/os-release first
        try:
            with open("/etc/os-release") as f:
                for line in f:
                    if line.startswith("ID="):
                        distro = line.split("=")[1].strip().strip('"').lower()
                    elif line.startswith("ID_LIKE="):
                        distro_like = line.split("=")[1].strip().strip('"').lower()
        except FileNotFoundError:
            pass
        
        return distro, distro_like
    
    def _detect_linux_package_manager(self, distro: Optional[str], distro_like: Optional[str]) -> PackageManager:
        """Detect Linux package manager based on distro"""
        # Check by distro first
        distro_managers = {
            "arch": PackageManager.PACMAN,
            "manjaro": PackageManager.PACMAN,
            "endeavouros": PackageManager.PACMAN,
            "ubuntu": PackageManager.APT,
            "debian": PackageManager.APT,
            "linuxmint": PackageManager.APT,
            "pop": PackageManager.APT,
            "fedora": PackageManager.DNF,
            "rhel": PackageManager.DNF,
            "centos": PackageManager.DNF,
            "rocky": PackageManager.DNF,
            "alma": PackageManager.DNF,
            "opensuse": PackageManager.ZYPPER,
        }
        
        if distro and distro in distro_managers:
            return distro_managers[distro]
        
        # Check distro_like
        if distro_like:
            for key, mgr in distro_managers.items():
                if key in distro_like:
                    return mgr
        
        # Fallback: check which package managers are available
        if shutil.which("pacman"):
            return PackageManager.PACMAN
        elif shutil.which("apt") or shutil.which("apt-get"):
            return PackageManager.APT
        elif shutil.which("dnf"):
            return PackageManager.DNF
        elif shutil.which("yum"):
            return PackageManager.YUM
        elif shutil.which("zypper"):
            return PackageManager.ZYPPER
        
        return PackageManager.CURL
    
    def is_ollama_installed(self) -> bool:
        """Check if Ollama is installed on the system"""
        return shutil.which("ollama") is not None
    
    def is_ollama_running(self) -> bool:
        """Check if Ollama service is running and accessible"""
        if not REQUESTS_AVAILABLE:
            # Fallback: try subprocess
            try:
                result = subprocess.run(
                    ["ollama", "list"],
                    capture_output=True,
                    timeout=5
                )
                return result.returncode == 0
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                return False
        
        try:
            response = requests.get(self.OLLAMA_API_URL, timeout=3)
            return response.status_code == 200
        except (requests.ConnectionError, requests.Timeout):
            return False
        except Exception:
            return False
    
    def try_start_ollama(self) -> bool:
        """
        Attempt to start Ollama service.
        
        Returns:
            True if Ollama is now running, False otherwise
        """
        if self.is_ollama_running():
            return True
        
        if not self.is_ollama_installed():
            return False
        
        self._print("🔄 Attempting to start Ollama...", "yellow")
        
        # Platform-specific start methods
        if self.os_info.os_type == OSType.WINDOWS:
            return self._start_ollama_windows()
        elif self.os_info.os_type == OSType.LINUX:
            return self._start_ollama_linux()
        elif self.os_info.os_type == OSType.MACOS:
            return self._start_ollama_macos()
        else:
            return self._start_ollama_generic()
    
    def _start_ollama_windows(self) -> bool:
        """Start Ollama on Windows"""
        try:
            # First try to find and start Ollama app
            # Ollama on Windows typically runs as a tray application
            ollama_path = shutil.which("ollama")
            
            if ollama_path:
                # Start 'ollama serve' in background
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE
                
                self._start_process = subprocess.Popen(
                    ["ollama", "serve"],
                    startupinfo=startupinfo,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS
                )
                
                # Wait for service to become available
                return self._wait_for_ollama(timeout=15)
        except Exception as e:
            self._print(f"⚠️ Failed to start Ollama: {e}", "yellow")
        
        return False
    
    def _start_ollama_linux(self) -> bool:
        """Start Ollama on Linux"""
        # Try systemd first
        if shutil.which("systemctl"):
            try:
                # Check if ollama service exists
                result = subprocess.run(
                    ["systemctl", "list-unit-files", "ollama.service"],
                    capture_output=True,
                    text=True
                )
                
                if "ollama.service" in result.stdout:
                    # Try to start via systemctl (might need sudo)
                    subprocess.run(
                        ["systemctl", "start", "ollama"],
                        capture_output=True,
                        timeout=10
                    )
                    
                    if self._wait_for_ollama(timeout=10):
                        self._print("✅ Started Ollama via systemd", "green")
                        return True
            except (subprocess.TimeoutExpired, OSError):
                pass
        
        # Fallback: start manually in background
        return self._start_ollama_generic()
    
    def _start_ollama_macos(self) -> bool:
        """Start Ollama on macOS"""
        try:
            # Try launchctl first (if installed as service)
            result = subprocess.run(
                ["launchctl", "list"],
                capture_output=True,
                text=True
            )
            
            if "ollama" in result.stdout.lower():
                subprocess.run(
                    ["launchctl", "start", "com.ollama.ollama"],
                    capture_output=True
                )
                if self._wait_for_ollama(timeout=10):
                    return True
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass
        
        # Fallback: start manually
        return self._start_ollama_generic()
    
    def _start_ollama_generic(self) -> bool:
        """Start Ollama using generic method (ollama serve)"""
        try:
            # Start in background
            if self.os_info.os_type == OSType.WINDOWS:
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE
                
                self._start_process = subprocess.Popen(
                    ["ollama", "serve"],
                    startupinfo=startupinfo,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
            else:
                # Unix-like systems
                self._start_process = subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True
                )
            
            self._print("⏳ Waiting for Ollama to start...", "dim")
            return self._wait_for_ollama(timeout=15)
            
        except Exception as e:
            self._print(f"⚠️ Failed to start Ollama: {e}", "yellow")
            return False
    
    def _wait_for_ollama(self, timeout: int = 15) -> bool:
        """Wait for Ollama to become available"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.is_ollama_running():
                self._print("✅ Ollama is now running!", "green")
                return True
            time.sleep(1)
        
        return False
    
    def prompt_install_ollama(self) -> bool:
        """
        Prompt user to install Ollama.
        
        Returns:
            True if user wants to install, False otherwise
        """
        if self.auto_install:
            return True
        
        self._print_panel(
            "Ollama is not installed on your system.\n\n"
            "Ollama is required to run AI models locally.\n"
            f"Your system: {self.os_info.name} ({self.os_info.os_type.value})\n"
            f"Package manager: {self.os_info.package_manager.value}",
            title="🤖 Ollama Not Found",
            style="yellow"
        )
        
        return self._confirm("Would you like to install Ollama?", default=True)
    
    def install_ollama(self) -> bool:
        """
        Install Ollama based on OS and available package manager.
        
        Returns:
            True if installation successful, False otherwise
        """
        self._print_panel(
            f"Installing Ollama for {self.os_info.name}...\n"
            f"Using: {self.os_info.package_manager.value}",
            title="📦 Installing Ollama",
            style="blue"
        )
        
        if self.os_info.os_type == OSType.WINDOWS:
            return self._install_ollama_windows()
        elif self.os_info.os_type == OSType.LINUX:
            return self._install_ollama_linux()
        elif self.os_info.os_type == OSType.MACOS:
            return self._install_ollama_macos()
        else:
            self._print_panel(
                f"Please install Ollama manually from:\n{self.OLLAMA_DOWNLOAD_URL}",
                title="Manual Installation Required",
                style="yellow"
            )
            return False
    
    def _install_ollama_windows(self) -> bool:
        """Install Ollama on Windows"""
        options = {}
        
        if self.os_info.package_manager == PackageManager.WINGET:
            options["1"] = "Install via winget (recommended)"
        if self.os_info.package_manager == PackageManager.CHOCOLATEY:
            options["2"] = "Install via Chocolatey"
        options["3"] = f"Open download page ({self.OLLAMA_DOWNLOAD_URL})"
        
        choice = self._select_option("Select installation method:", options)
        
        if choice == "1" and PackageManager.WINGET in [self.os_info.package_manager]:
            return self._run_install_command(["winget", "install", "Ollama.Ollama", "-e"])
        elif choice == "2" and PackageManager.CHOCOLATEY in [self.os_info.package_manager]:
            return self._run_install_command(["choco", "install", "ollama", "-y"])
        elif choice == "3":
            import webbrowser
            webbrowser.open(self.OLLAMA_DOWNLOAD_URL)
            self._print("📂 Opening download page in browser...", "blue")
            self._print("Please install Ollama and restart this application.", "yellow")
            return False
        
        return False
    
    def _install_ollama_linux(self) -> bool:
        """Install Ollama on Linux"""
        pkg_mgr = self.os_info.package_manager
        
        options = {
            "1": "Install via official script (curl - recommended)"
        }
        
        if pkg_mgr == PackageManager.PACMAN:
            options["2"] = "Install via pacman (AUR)"
        elif pkg_mgr == PackageManager.APT:
            options["2"] = "Install via apt (if available)"
        
        choice = self._select_option("Select installation method:", options)
        
        if choice == "1":
            # Use official install script
            self._print("🔧 Running official Ollama installer...", "blue")
            self._print("This may require sudo password.", "yellow")
            
            try:
                # Download and run install script
                result = subprocess.run(
                    ["bash", "-c", f"curl -fsSL {self.OLLAMA_INSTALL_SCRIPT} | sh"],
                    check=True
                )
                return result.returncode == 0 and self.is_ollama_installed()
            except subprocess.CalledProcessError as e:
                self._print(f"⚠️ Installation failed: {e}", "red")
                return False
        
        elif choice == "2":
            if pkg_mgr == PackageManager.PACMAN:
                # Try yay or paru for AUR
                if shutil.which("yay"):
                    return self._run_install_command(["yay", "-S", "--noconfirm", "ollama"])
                elif shutil.which("paru"):
                    return self._run_install_command(["paru", "-S", "--noconfirm", "ollama"])
                else:
                    self._print("⚠️ No AUR helper found (yay/paru). Using official script...", "yellow")
                    return self._run_install_command(
                        ["bash", "-c", f"curl -fsSL {self.OLLAMA_INSTALL_SCRIPT} | sh"]
                    )
            elif pkg_mgr == PackageManager.APT:
                self._print("ℹ️ Ollama is not in default apt repos. Using official script...", "blue")
                return self._run_install_command(
                    ["bash", "-c", f"curl -fsSL {self.OLLAMA_INSTALL_SCRIPT} | sh"]
                )
        
        return False
    
    def _install_ollama_macos(self) -> bool:
        """Install Ollama on macOS"""
        options = {}
        
        if shutil.which("brew"):
            options["1"] = "Install via Homebrew (recommended)"
        options["2"] = "Install via official script (curl)"
        options["3"] = f"Open download page ({self.OLLAMA_DOWNLOAD_URL})"
        
        choice = self._select_option("Select installation method:", options)
        
        if choice == "1" and shutil.which("brew"):
            return self._run_install_command(["brew", "install", "ollama"])
        elif choice == "2":
            return self._run_install_command(
                ["bash", "-c", f"curl -fsSL {self.OLLAMA_INSTALL_SCRIPT} | sh"]
            )
        elif choice == "3":
            import webbrowser
            webbrowser.open(self.OLLAMA_DOWNLOAD_URL)
            self._print("📂 Opening download page in browser...", "blue")
            return False
        
        return False
    
    def _run_install_command(self, cmd: list) -> bool:
        """Run installation command and check result"""
        try:
            self._print(f"🔧 Running: {' '.join(cmd)}", "dim")
            
            result = subprocess.run(cmd, check=True)
            
            if self.is_ollama_installed():
                self._print("✅ Ollama installed successfully!", "green")
                return True
            else:
                self._print("⚠️ Installation completed but Ollama not found in PATH", "yellow")
                return False
                
        except subprocess.CalledProcessError as e:
            self._print(f"❌ Installation failed with exit code {e.returncode}", "red")
            return False
        except FileNotFoundError as e:
            self._print(f"❌ Command not found: {e}", "red")
            return False
    
    def ensure_ollama_available(self) -> Tuple[bool, str]:
        """
        Main entry point: Ensure Ollama is available and running.
        
        This method:
        1. Checks if Ollama is running → return success
        2. If not running but installed → try to start
        3. If not installed → prompt to install (with user consent)
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        # Step 1: Check if already running
        if self.is_ollama_running():
            return True, "Ollama is running"
        
        # Step 2: Check if installed
        if self.is_ollama_installed():
            # Try to start
            if self.auto_start:
                if self.try_start_ollama():
                    return True, "Ollama started successfully"
                else:
                    return False, (
                        "Failed to start Ollama automatically.\n"
                        "Please start it manually with: ollama serve"
                    )
            else:
                return False, (
                    "Ollama is installed but not running.\n"
                    "Start with: ollama serve"
                )
        
        # Step 3: Not installed - prompt for installation
        if self.prompt_install_ollama():
            if self.install_ollama():
                # Try to start after installation
                if self.try_start_ollama():
                    return True, "Ollama installed and started successfully!"
                else:
                    return False, (
                        "Ollama installed but failed to start.\n"
                        "Please start manually with: ollama serve"
                    )
            else:
                return False, (
                    "Installation was not completed.\n"
                    f"Please install manually from: {self.OLLAMA_DOWNLOAD_URL}"
                )
        else:
            # User declined installation
            return False, (
                "Ollama is required for AI features.\n"
                f"Install from: {self.OLLAMA_DOWNLOAD_URL}"
            )
    
    def get_status_summary(self) -> Dict:
        """Get a summary of Ollama status"""
        return {
            "installed": self.is_ollama_installed(),
            "running": self.is_ollama_running(),
            "os": self.os_info.os_type.value,
            "os_name": self.os_info.name,
            "package_manager": self.os_info.package_manager.value,
        }


# Convenience function for quick checks
def ensure_ollama() -> Tuple[bool, str]:
    """
    Quick helper to ensure Ollama is available.
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    manager = OllamaFallbackManager()
    return manager.ensure_ollama_available()


if __name__ == "__main__":
    # Demo / test the fallback manager
    manager = OllamaFallbackManager()
    
    print("\n" + "="*50)
    print("🔍 Ollama Fallback Manager - Status Check")
    print("="*50)
    
    status = manager.get_status_summary()
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*50)
    print("🚀 Running ensure_ollama_available()...")
    print("="*50 + "\n")
    
    success, message = manager.ensure_ollama_available()
    
    print("\n" + "="*50)
    print(f"Result: {'✅ Success' if success else '❌ Failed'}")
    print(f"Message: {message}")
    print("="*50)
