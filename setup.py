import subprocess
import sys
import os

def install_requirements(requirements_file='requirements.txt'):
    if not os.path.isfile(requirements_file):
        print(f"{requirements_file} not found.")
        sys.exit(1)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])

if __name__ == "__main__":
    install_requirements()