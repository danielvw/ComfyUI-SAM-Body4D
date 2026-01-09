"""
Installation script for ComfyUI-SAM-Body4D

This script helps set up the node suite and its dependencies.
"""

import os
import sys
import subprocess
from pathlib import Path

# ANSI colors for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(msg):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}")
    print(f"{msg}")
    print(f"{'='*60}{Colors.ENDC}\n")


def print_success(msg):
    print(f"{Colors.OKGREEN}✓ {msg}{Colors.ENDC}")


def print_error(msg):
    print(f"{Colors.FAIL}✗ {msg}{Colors.ENDC}")


def print_warning(msg):
    print(f"{Colors.WARNING}⚠ {msg}{Colors.ENDC}")


def print_info(msg):
    print(f"{Colors.OKCYAN}ℹ {msg}{Colors.ENDC}")


def run_command(cmd, error_msg):
    """Run a command and handle errors."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print_error(error_msg)
        print(f"  Command: {cmd}")
        print(f"  Error: {e.stderr}")
        return False, e.stderr


def check_python_version():
    """Check Python version."""
    print_info("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print_success(f"Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print_error(f"Python 3.10+ required, found {version.major}.{version.minor}")
        return False


def check_pytorch():
    """Check if PyTorch is installed."""
    print_info("Checking PyTorch installation...")
    try:
        import torch
        print_success(f"PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print_success(f"  CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print_warning("  CUDA not available")
        return True
    except ImportError:
        print_error("PyTorch not found")
        return False


def install_requirements():
    """Install requirements from requirements.txt."""
    print_info("Installing requirements...")
    req_file = Path(__file__).parent / "requirements.txt"

    success, output = run_command(
        f"{sys.executable} -m pip install -r {req_file}",
        "Failed to install requirements"
    )

    if success:
        print_success("Requirements installed")
    return success


def check_sam_body4d():
    """Check if SAM-Body4D is installed."""
    print_info("Checking SAM-Body4D installation...")

    # Expected path (sibling to this directory)
    sam_body4d_path = Path(__file__).parent.parent / "sam-body4d"

    if sam_body4d_path.exists():
        print_success(f"SAM-Body4D found at: {sam_body4d_path}")

        # Check if installed
        try:
            sys.path.insert(0, str(sam_body4d_path))
            sys.path.insert(0, str(sam_body4d_path / "models" / "sam_3d_body"))
            from models.sam_3d_body.sam_3d_body import SAM3DBodyEstimator
            print_success("SAM-Body4D is importable")
            return True
        except ImportError as e:
            print_warning("SAM-Body4D found but not importable")
            print(f"  Error: {e}")
            print_info("  Attempting to install SAM-Body4D...")

            # Try to install
            success, _ = run_command(
                f"cd {sam_body4d_path} && {sys.executable} -m pip install -e .",
                "Failed to install SAM-Body4D"
            )
            return success
    else:
        print_error(f"SAM-Body4D not found at: {sam_body4d_path}")
        print_info("Please clone SAM-Body4D:")
        print(f"  cd {Path(__file__).parent.parent}")
        print(f"  git clone https://github.com/gaomingqi/sam-body4d.git")
        return False


def check_blender():
    """Check if Blender is accessible."""
    print_info("Checking Blender installation...")

    success, output = run_command(
        "blender --version",
        "Blender not found in PATH"
    )

    if success:
        version_line = output.split('\n')[0]
        print_success(f"Blender found: {version_line}")
        return True
    else:
        print_warning("Blender not found in PATH")
        print_info("Install Blender from https://www.blender.org/download/")
        print_info("Or specify path in Body4DExportFBX node")
        return True  # Not critical for basic functionality


def main():
    """Main installation routine."""
    print_header("ComfyUI-SAM-Body4D Installation")

    # Track status
    all_ok = True

    # 1. Check Python version
    all_ok &= check_python_version()

    # 2. Check PyTorch
    all_ok &= check_pytorch()

    # 3. Install requirements
    all_ok &= install_requirements()

    # 4. Check SAM-Body4D
    all_ok &= check_sam_body4d()

    # 5. Check Blender (optional)
    check_blender()

    # Final status
    print_header("Installation Summary")

    if all_ok:
        print_success("Installation completed successfully!")
        print_info("\nNext steps:")
        print("  1. Ensure SAM-Body4D checkpoints are downloaded")
        print("     (run: cd sam-body4d && python scripts/setup.py --ckpt-root checkpoints)")
        print("  2. Restart ComfyUI")
        print("  3. Look for 'SAM-Body4D' category in Add Node menu")
    else:
        print_error("Installation completed with errors")
        print_info("Please resolve the issues above and try again")

    return all_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
