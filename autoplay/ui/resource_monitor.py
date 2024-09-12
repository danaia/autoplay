import os
import psutil
import subprocess
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QProgressBar, QLabel
from PyQt6.QtCore import QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class GpuMonitorThread(QThread):
    gpu_data_signal = pyqtSignal(tuple)

    def run(self):
        while True:
            gpu_util, cuda_mem, gpu_wattage, cuda_cores = self.get_gpu_usage()
            self.gpu_data_signal.emit((gpu_util, cuda_mem, gpu_wattage, cuda_cores))
            time.sleep(1)

    def get_gpu_usage(self):
        try:
            result = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,power.draw,utilization.memory',
                                              '--format=csv,noheader,nounits'], text=True)
            gpu_util, mem_used, power_draw, mem_util = map(float, result.strip().split(', '))
            
            total_cores = 16384  # Correct CUDA core count for RTX 4090
            cuda_cores = int((mem_util / 100) * total_cores)
            
            return int(gpu_util), int(mem_used), int(float(power_draw)), cuda_cores
        except Exception as e:
            print(f"Error fetching GPU info: {e}")
            return 0, 0, 0, 0

class CpuMonitorThread(QThread):
    cpu_usage_signal = pyqtSignal(int)

    def run(self):
        while True:
            cpu_util = psutil.cpu_percent(interval=1)
            self.cpu_usage_signal.emit(cpu_util)

class ResourceMonitor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Comprehensive System Resource Monitor")
        self.setGeometry(100, 100, 1200, 800)

        self.setup_ui()
        self.setup_data()
        self.setup_threads()
        self.setup_animations()

    def setup_ui(self):
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        # Progress Bars and Labels
        for name in ['CPU', 'GPU', 'CUDA Memory']:
            setattr(self, f'{name.lower().replace(" ", "_")}_label', QLabel(f"{name} Usage"))
            setattr(self, f'{name.lower().replace(" ", "_")}_progress', QProgressBar())
            left_layout.addWidget(getattr(self, f'{name.lower().replace(" ", "_")}_label'))
            left_layout.addWidget(getattr(self, f'{name.lower().replace(" ", "_")}_progress'))

        self.wattage_label = QLabel("GPU Power Usage: 0 W")
        self.cores_label = QLabel("Estimated CUDA Cores in Use: 0")
        left_layout.addWidget(self.wattage_label)
        left_layout.addWidget(self.cores_label)

        # Matplotlib figures
        self.fig_line, self.ax_line = plt.subplots(figsize=(8, 4))
        self.canvas_line = FigureCanvas(self.fig_line)
        right_layout.addWidget(self.canvas_line)

        self.fig_pie, (self.ax_pie1, self.ax_pie2) = plt.subplots(1, 2, figsize=(8, 4))
        self.canvas_pie = FigureCanvas(self.fig_pie)
        right_layout.addWidget(self.canvas_pie)

        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)
        self.setLayout(main_layout)

    def setup_data(self):
        self.data = {
            'cpu_usage': [0],
            'gpu_usage': [0],
            'cuda_mem': [0],
            'gpu_wattage': [0],
            'cuda_cores': [0]
        }
        self.max_data_points = 60

    def setup_threads(self):
        self.gpu_thread = GpuMonitorThread()
        self.gpu_thread.gpu_data_signal.connect(self.update_gpu_data)
        self.gpu_thread.start()

        self.cpu_thread = CpuMonitorThread()
        self.cpu_thread.cpu_usage_signal.connect(self.update_cpu_usage)
        self.cpu_thread.start()

    def setup_animations(self):
        self.ani_line = FuncAnimation(self.fig_line, self.update_line_plot, interval=1000, cache_frame_data=False)
        self.ani_pie = FuncAnimation(self.fig_pie, self.update_pie_charts, interval=1000, cache_frame_data=False)

    def update_cpu_usage(self, cpu_util):
        self.cpu_progress.setValue(cpu_util)
        self.update_data('cpu_usage', cpu_util)

    def update_gpu_data(self, gpu_data):
        gpu_util, cuda_mem, gpu_wattage, cuda_cores = gpu_data
        self.gpu_progress.setValue(gpu_util)
        self.cuda_memory_progress.setValue(min(cuda_mem, 24576))  # Assuming 24GB VRAM
        self.wattage_label.setText(f"GPU Power Usage: {gpu_wattage} W")
        self.cores_label.setText(f"Estimated CUDA Cores in Use: {cuda_cores}")

        self.update_data('gpu_usage', gpu_util)
        self.update_data('cuda_mem', cuda_mem)
        self.update_data('gpu_wattage', gpu_wattage)
        self.update_data('cuda_cores', cuda_cores)

    def update_data(self, key, value):
        self.data[key].append(max(0, value))  # Ensure non-negative values
        if len(self.data[key]) > self.max_data_points:
            self.data[key].pop(0)

    def update_line_plot(self, frame):
        self.ax_line.clear()
        x = list(range(len(max(self.data.values(), key=len))))  # Use the longest data list for x-axis
        
        for key, color, label in [
            ('cpu_usage', 'blue', 'CPU Usage (%)'),
            ('gpu_usage', 'green', 'GPU Usage (%)'),
            ('cuda_mem', 'red', 'CUDA Memory (MB)'),
            ('gpu_wattage', 'orange', 'GPU Wattage (W)'),
            ('cuda_cores', 'purple', 'CUDA Cores (x100)')
        ]:
            y = self.data[key]
            if key == 'cuda_cores':
                y = [core / 100 for core in y]
            # Pad y with zeros if it's shorter than x
            y = y + [0] * (len(x) - len(y))
            self.ax_line.plot(x, y, color=color, label=label)

        self.ax_line.legend(loc='upper left')
        max_value = max(max(max(data) for data in self.data.values()), 100)
        self.ax_line.set_ylim([0, max_value * 1.1])
        self.ax_line.set_xlim([0, self.max_data_points - 1])
        self.ax_line.set_title("Real-Time System Resource Usage")
        self.ax_line.set_xlabel("Time (s)")
        self.ax_line.set_ylabel("Usage")
        self.ax_line.grid(True)
        
        self.canvas_line.draw()

    def update_pie_charts(self, frame):
        self.ax_pie1.clear()
        self.ax_pie2.clear()

        cpu = max(0, min(100, self.data['cpu_usage'][-1]))
        gpu = max(0, min(100, self.data['gpu_usage'][-1]))
        self.ax_pie1.pie([cpu, 100-cpu, gpu, 100-gpu], 
                         labels=['CPU Used', 'CPU Free', 'GPU Used', 'GPU Free'],
                         colors=['blue', 'lightblue', 'green', 'lightgreen'],
                         autopct='%1.1f%%', startangle=90)
        self.ax_pie1.set_title("CPU and GPU Usage")

        cuda_mem = max(0, min(24576, self.data['cuda_mem'][-1]))  # Assuming 24GB VRAM
        cuda_cores = max(0, min(16384, self.data['cuda_cores'][-1]))  # Corrected for RTX 4090
        total_mem = 24576  # 24GB for RTX 4090
        total_cores = 16384  # Corrected for RTX 4090
        self.ax_pie2.pie([cuda_mem, total_mem-cuda_mem, cuda_cores, total_cores-cuda_cores],
                         labels=['CUDA Mem Used', 'CUDA Mem Free', 'CUDA Cores Used', 'CUDA Cores Free'],
                         colors=['red', 'pink', 'purple', 'lavender'],
                         autopct='%1.1f%%', startangle=90)
        self.ax_pie2.set_title("CUDA Memory and Cores Usage")

        self.canvas_pie.draw()