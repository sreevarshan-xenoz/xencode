import os
import sys
import subprocess
import shutil
from pathlib import Path

def main():
    print("🚀 Xencode Standalone Builder")
    print("===========================")

    # Define paths
    project_root = Path(__file__).parent.absolute()
    dist_dir = project_root / "dist"
    build_dir = project_root / "build"
    
    # 1. Check/Install PyInstaller
    try:
        import PyInstaller
        print("✅ PyInstaller already installed.")
    except ImportError:
        print("🔧 Installing PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller>=6.3.0"])
        print("✅ PyInstaller installed.")

    # 2. Clean previous builds
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    if build_dir.exists():
        shutil.rmtree(build_dir)
    
    print("🧹 Cleaned previous build artifacts.")

    # 3. Build Command construction
    # We point to xencode_core.py as the entry point, or creating a specific entry script?
    # xencode_core.py is the main engine. enhanced_cli_system.py is the new integration point.
    # The install.sh creates a shim that runs xencode_core.py.
    # However, enhanced_cli_system.py seems better as the entry point if we want the new features.
    # Let's check xencode_core.py's __main__ block. It runs `main()`.
    # Let's check enhanced_cli_system.py's __main__ block. It runs `main()` which prints a demo message?
    # Ah, enhanced_cli_system.py's main is a demo. We need a proper entry point script that mimics the shell shim.
    
    # Let's create a temporary entry point script for the standalone exe
    entry_script = project_root / "xencode_entry.py"
    with open(entry_script, "w", encoding="utf-8") as f:
        f.write('from xencode.enhanced_cli_system import EnhancedXencodeCLI\n')
        f.write('from xencode_core import main as core_main\n')
        f.write('import sys\n')
        f.write('import os\n\n')
        f.write('if __name__ == "__main__":\n')
        f.write('    # Determine mode based on args (similar to install.sh logic)\n')
        f.write('    # However, for simplicity, we pass everything to xencode_core for now,\n')
        f.write('    # OR we use enhanced CLI if arguments match new features.\n')
        f.write('    # Let\'s stick to wrapping xencode_core.py as the primary for stability,\n')
        f.write('    # since enhanced_cli_system isn\'t fully integrated yet as the MAIN entry.\n')
        f.write('    # Wait, the user wants the "cli tool that runs the terminal".\n')
        f.write('    # That means launching TUI/Chat mode by default.\n')
        f.write('    core_main()\n')

    print("📝 Created temporary entry point.")

    # PyInstaller arguments
    pyinstaller_args = [
        "pyinstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--console",
        "--name", "xencode",
        "--hidden-import", "rich",
        "--hidden-import", "prompt_toolkit",
        "--hidden-import", "textual",
        "--collect-all", "textual",  # Collect textual resources
        "--collect-all", "xencode",
        str(entry_script)
    ]
    
    # Run PyInstaller
    print(f"🔨 Building executable with command: {' '.join(pyinstaller_args)}")
    try:
        subprocess.check_call(pyinstaller_args)
        print("\n✅ Build successful!")
        print(f"📦 Executable location: {dist_dir / 'xencode.exe'}")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Build failed with error code {e.returncode}")
        sys.exit(1)
    finally:
        # Cleanup entry script (optional, but clean)
        if entry_script.exists():
            entry_script.unlink()

if __name__ == "__main__":
    main()
