import sys
import logging
import torch
import time
import gc
import os
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton, 
                             QProgressBar, QLabel, QSlider, QSpinBox, QLineEdit, QListWidget, QScrollArea, 
                             QFrame, QSizePolicy, QDialog, QFileDialog)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QUrl
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
import subprocess
from diffusers import CogVideoXPipeline, CogVideoXDPMScheduler
from diffusers.utils import export_to_video

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

class VideoGenerator(QThread):
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    time_estimate = pyqtSignal(str)
    video_generated = pyqtSignal(str)

    def __init__(self, text, num_inference_steps, guidance_scale, num_frames, project_name, sequence_number, output_dir, num_videos):
        super().__init__()
        self.text = text
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.num_frames = num_frames
        self.project_name = project_name
        self.sequence_number = sequence_number
        self.output_dir = output_dir
        self.num_videos = num_videos

    def run(self):
        try:
            pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16)
            pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
            pipe.enable_model_cpu_offload()
            pipe.enable_sequential_cpu_offload()
            pipe.vae.enable_slicing()
            pipe.vae.enable_tiling()

            start_time = time.time()
            for video_idx in range(self.num_videos):
                video = pipe(
                    prompt=self.text,
                    num_videos_per_prompt=1,
                    num_inference_steps=self.num_inference_steps,
                    num_frames=self.num_frames,
                    use_dynamic_cfg=True,
                    guidance_scale=self.guidance_scale,
                    generator=torch.Generator().manual_seed(42 + video_idx),  # Seed changes per video
                ).frames[0]

                progress = int(((video_idx + 1) / self.num_videos) * 100)
                self.progress.emit(progress)

                elapsed_time = time.time() - start_time
                estimated_total_time = elapsed_time / (video_idx + 1) * self.num_videos
                remaining_time = estimated_total_time - elapsed_time
                self.time_estimate.emit(f"Estimated time remaining: {remaining_time:.2f} seconds")

                # Save each video in the output directory with a unique filename
                output_path = os.path.join(self.output_dir, f"{self.project_name}_{self.sequence_number}_video_{video_idx + 1}.mp4")
                export_to_video(video, output_path, fps=8)

                self.video_generated.emit(output_path)

            self.finished.emit()
        except Exception as e:
            logging.error(f"Error generating video: {e}")
        finally:
            torch.cuda.empty_cache()
            gc.collect()

class VideoPlayer(QDialog):
    def __init__(self, video_path):
        super().__init__()
        self.setWindowTitle("Video Player")
        self.setGeometry(200, 200, 800, 600)

        layout = QVBoxLayout()

        self.video_widget = QVideoWidget()
        layout.addWidget(self.video_widget)

        self.media_player = QMediaPlayer()
        self.media_player.setVideoOutput(self.video_widget)
        self.media_player.setSource(QUrl.fromLocalFile(video_path))

        control_layout = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.play_pause)
        control_layout.addWidget(self.play_button)

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        control_layout.addWidget(self.close_button)

        layout.addLayout(control_layout)

        self.setLayout(layout)

    def play_pause(self):
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause()
            self.play_button.setText("Play")
        else:
            self.media_player.play()
            self.play_button.setText("Pause")

    def closeEvent(self, event):
        self.media_player.stop()
        super().closeEvent(event)

class PromptPanel(QFrame):
    def __init__(self, panel_id):
        super().__init__()
        self.panel_id = panel_id
        self.render_queue = []
        self.current_sequence = 0
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        project_layout = QHBoxLayout()
        project_layout.addWidget(QLabel("Project Name:"))
        self.project_name_input = QLineEdit()
        project_layout.addWidget(self.project_name_input)
        layout.addLayout(project_layout)

        self.text_edit = QTextEdit()
        layout.addWidget(self.text_edit)

        inference_layout = QHBoxLayout()
        inference_layout.addWidget(QLabel("Inference Steps:"))
        self.inference_steps_spinbox = QSpinBox()
        self.inference_steps_spinbox.setRange(1, 100)
        self.inference_steps_spinbox.setValue(50)
        inference_layout.addWidget(self.inference_steps_spinbox)
        layout.addLayout(inference_layout)

        guidance_layout = QHBoxLayout()
        guidance_layout.addWidget(QLabel("Guidance Scale:"))
        self.guidance_slider = QSlider(Qt.Orientation.Horizontal)
        self.guidance_slider.setRange(1, 20)
        self.guidance_slider.setValue(7)
        self.guidance_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.guidance_slider.setTickInterval(1)
        self.guidance_label = QLabel("7.0")
        guidance_layout.addWidget(self.guidance_slider)
        guidance_layout.addWidget(self.guidance_label)
        layout.addLayout(guidance_layout)
        self.guidance_slider.valueChanged.connect(self.update_guidance_label)

        frames_layout = QHBoxLayout()
        frames_layout.addWidget(QLabel("Number of Frames:"))
        self.frames_spinbox = QSpinBox()
        self.frames_spinbox.setRange(1, 100)
        self.frames_spinbox.setValue(49)
        frames_layout.addWidget(self.frames_spinbox)
        layout.addLayout(frames_layout)

        videos_layout = QHBoxLayout()
        videos_layout.addWidget(QLabel("Number of Videos:"))
        self.videos_spinbox = QSpinBox()
        self.videos_spinbox.setRange(1, 10)
        self.videos_spinbox.setValue(1)
        videos_layout.addWidget(self.videos_spinbox)
        layout.addLayout(videos_layout)

        self.queue_button = QPushButton('Add to Queue')
        self.queue_button.clicked.connect(self.add_to_queue)
        layout.addWidget(self.queue_button)

        self.queue_list = QListWidget()
        layout.addWidget(self.queue_list)

        self.setLayout(layout)
        self.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Raised)

    def update_guidance_label(self, value):
        self.guidance_label.setText(f"{value}.0")

    def add_to_queue(self):
        project_name = self.project_name_input.text()
        text = self.text_edit.toPlainText()
        if not project_name or not text:
            logging.warning("Project name and text are required")
            return

        self.current_sequence += 1
        queue_item = {
            'panel_id': self.panel_id,
            'project_name': project_name,
            'text': text,
            'num_inference_steps': self.inference_steps_spinbox.value(),
            'guidance_scale': self.guidance_slider.value(),
            'num_frames': self.frames_spinbox.value(),
            'sequence_number': self.current_sequence,
            'num_videos': self.videos_spinbox.value()
        }
        self.render_queue.append(queue_item)
        self.queue_list.addItem(f"{project_name}_{self.current_sequence}")

class TextToVideoGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.panels = []
        self.render_queue = []
        self.output_dir = ""
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        panel_control_layout = QHBoxLayout()
        panel_control_layout.addWidget(QLabel("Number of Panels:"))
        self.panel_count_spinbox = QSpinBox()
        self.panel_count_spinbox.setRange(1, 10)
        self.panel_count_spinbox.setValue(1)
        self.panel_count_spinbox.valueChanged.connect(self.update_panel_count)
        panel_control_layout.addWidget(self.panel_count_spinbox)
        main_layout.addLayout(panel_control_layout)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        self.panel_layout = QHBoxLayout(scroll_content)
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)

        self.add_panel()

        self.output_dir_button = QPushButton('Select Output Directory')
        self.output_dir_button.clicked.connect(self.select_output_directory)
        main_layout.addWidget(self.output_dir_button)

        self.process_all_button = QPushButton('Process All Queues')
        self.process_all_button.clicked.connect(self.process_all_queues)
        main_layout.addWidget(self.process_all_button)

        self.progress_bar = QProgressBar()
        main_layout.addWidget(self.progress_bar)

        self.time_estimate_label = QLabel("Estimated time remaining: N/A")
        main_layout.addWidget(self.time_estimate_label)

        self.setLayout(main_layout)
        self.setWindowTitle('Multi-Panel Text to Video Generator')
        self.setGeometry(100, 100, 1200, 800)

    def add_panel(self):
        panel = PromptPanel(len(self.panels))
        self.panels.append(panel)
        self.panel_layout.addWidget(panel)

    def select_output_directory(self):
        self.output_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if self.output_dir:
            logging.info(f"Output directory set to: {self.output_dir}")

    def remove_panel(self):
        if self.panels:
            panel = self.panels.pop()
            panel.deleteLater()

    def update_panel_count(self, count):
        current_count = len(self.panels)
        if count > current_count:
            for _ in range(count - current_count):
                self.add_panel()
        elif count < current_count:
            for _ in range(current_count - count):
                self.remove_panel()

    def process_all_queues(self):
        if not self.output_dir:
            logging.warning("Please select an output directory first")
            return

        self.render_queue = []
        for panel in self.panels:
            self.render_queue.extend(panel.render_queue)
            panel.render_queue.clear()
            panel.queue_list.clear()

        if not self.render_queue:
            logging.warning("All queues are empty")
            return

        self.install_dependencies()

    def install_dependencies(self):
        self.installer = DependencyInstaller()
        self.installer.finished.connect(self.on_dependencies_installed)
        self.installer.progress.connect(self.progress_bar.setValue)
        self.installer.start()

    def on_dependencies_installed(self):
        logging.info("Dependencies installed successfully")
        self.start_next_render()

    def start_next_render(self):
        if not self.render_queue:
            logging.info("All renders completed")
            return

        item = self.render_queue.pop(0)
        self.generator = VideoGenerator(
            item['text'],
            item['num_inference_steps'],
            item['guidance_scale'],
            item['num_frames'],
            item['project_name'],
            item['sequence_number'],
            self.output_dir,
            item['num_videos']
        )
        self.generator.finished.connect(self.on_video_generated)
        self.generator.progress.connect(self.progress_bar.setValue)
        self.generator.time_estimate.connect(self.update_time_estimate)
        self.generator.video_generated.connect(self.show_video)
        self.generator.start()

    def on_video_generated(self):
        logging.info("Video generated successfully")
        self.progress_bar.setValue(0)
        self.time_estimate_label.setText("Estimated time remaining: N/A")
        self.start_next_render()  # Move to the next video in the queue

    def update_time_estimate(self, estimate):
        self.time_estimate_label.setText(estimate)

    def show_video(self, video_path):
        player = VideoPlayer(video_path)
        player.finished.connect(self.start_next_render)
        player.show()

def main():
    app = QApplication(sys.argv)
    ex = TextToVideoGUI()
    ex.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
