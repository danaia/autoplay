import subprocess
import sys
import logging
from PyQt6.QtCore import QThread, pyqtSignal

class DependencyInstaller(QThread):
    """Thread for installing dependencies."""
    finished = pyqtSignal()
    progress = pyqtSignal(int)

    def run(self):
        """Install required dependencies."""
        try:
            dependencies = [
                'torch',
                'torchvision',
                'torchaudio',
                'transformers',
                'diffusers>=0.30.0',
                'accelerate',
                'moviepy',
                'opencv-python',
                'sentencepiece',
            ]
            total = len(dependencies)
            for i, dep in enumerate(dependencies, 1):
                subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
                self.progress.emit(int((i / total) * 100))

            self.finished.emit()
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to install dependencies: {e}")
