# resource_monitor.py
import os
import psutil
import subprocess
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QProgressBar, QLabel
from PyQt6.QtCore import QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Thread for fetching GPU data from nvidia-smi
class GpuMonitorThread(QThread):
    gpu_usage_signal = pyqtSignal(int)
    cuda_mem_signal = pyqtSignal(int)

    def run(self):
        while True:
            gpu_util, cuda_mem = self.get_gpu_usage()
            self.gpu_usage_signal.emit(gpu_util)
            self.cuda_mem_signal.emit(cuda_mem)

    def get_gpu_usage(self):
        try:
            result = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
                                              '--format=csv,noheader,nounits'])
            gpu_util, mem_used = map(int, result.decode().strip().split(', '))
            return gpu_util, mem_used
        except Exception as e:
            print(f"Error fetching GPU info: {e}")
            return 0, 0

# Thread for fetching CPU data
class CpuMonitorThread(QThread):
    cpu_usage_signal = pyqtSignal(int)

    def run(self):
        while True:
            cpu_util = psutil.cpu_percent(interval=1)
            self.cpu_usage_signal.emit(cpu_util)

class ResourceMonitor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("System Resource Monitor")
        self.setGeometry(100, 100, 600, 400)

        layout = QVBoxLayout()

        # CPU Progress Bar
        self.cpu_label = QLabel("CPU Usage")
        self.cpu_progress = QProgressBar()
        layout.addWidget(self.cpu_label)
        layout.addWidget(self.cpu_progress)

        # GPU Progress Bar
        self.gpu_label = QLabel("GPU Usage")
        self.gpu_progress = QProgressBar()
        layout.addWidget(self.gpu_label)
        layout.addWidget(self.gpu_progress)

        # CUDA Memory Progress Bar
        self.cuda_label = QLabel("CUDA Memory Usage (MB)")
        self.cuda_progress = QProgressBar()
        layout.addWidget(self.cuda_label)
        layout.addWidget(self.cuda_progress)

        # Matplotlib canvas for real-time plotting
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        self.cpu_usage_data = []
        self.gpu_usage_data = []
        self.cuda_mem_data = []

        self.setLayout(layout)

        # Start monitoring threads
        self.gpu_thread = GpuMonitorThread()
        self.gpu_thread.gpu_usage_signal.connect(self.update_gpu_usage)
        self.gpu_thread.cuda_mem_signal.connect(self.update_cuda_mem)
        self.gpu_thread.start()

        self.cpu_thread = CpuMonitorThread()
        self.cpu_thread.cpu_usage_signal.connect(self.update_cpu_usage)
        self.cpu_thread.start()

        # Matplotlib animation
        self.ani = FuncAnimation(self.fig, self.update_plot, interval=1000)

    def update_cpu_usage(self, cpu_util):
        self.cpu_progress.setValue(cpu_util)
        self.cpu_usage_data.append(cpu_util)
        if len(self.cpu_usage_data) > 60:
            self.cpu_usage_data.pop(0)

    def update_gpu_usage(self, gpu_util):
        self.gpu_progress.setValue(gpu_util)
        self.gpu_usage_data.append(gpu_util)
        if len(self.gpu_usage_data) > 60:
            self.gpu_usage_data.pop(0)

    def update_cuda_mem(self, cuda_mem):
        self.cuda_progress.setValue(cuda_mem)
        self.cuda_mem_data.append(cuda_mem)
        if len(self.cuda_mem_data) > 60:
            self.cuda_mem_data.pop(0)

    def update_plot(self, frame):
        self.ax.clear()
        self.ax.plot(self.cpu_usage_data, label='CPU Usage (%)', color='blue')
        self.ax.plot(self.gpu_usage_data, label='GPU Usage (%)', color='green')
        self.ax.plot(self.cuda_mem_data, label='CUDA Memory (MB)', color='red')
        self.ax.legend(loc='upper right')
        self.ax.set_ylim([0, 100])
        self.ax.set_title("Real-Time System Resource Usage")
        self.canvas.draw()
