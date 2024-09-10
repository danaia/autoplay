from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QProgressBar, QLabel, QScrollArea, QSpinBox, QFileDialog
from PyQt6.QtCore import Qt
from ui.prompt_panel import PromptPanel
from core.video_generator import VideoGenerator
from core.dependency_installer import DependencyInstaller
from queue_manager import QueueManager  # Import QueueManager directly if in the same folder as main.py
from ui.resource_monitor import ResourceMonitor  # <-- Import the new Resource Monitor window
import logging

class TextToVideoGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.panels = []
        self.output_dir = ""
        self.queue_manager = QueueManager()  # Initialize the QueueManager
        self.init_ui()
        self.open_resource_monitor()

        # Connect the queue manager's signal to update the UI when the queue changes
        self.queue_manager.queue_updated.connect(self.update_queue_ui)

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

    def open_resource_monitor(self):
        self.resource_monitor = ResourceMonitor()
        self.resource_monitor.show()

    def add_panel(self):
        panel = PromptPanel(len(self.panels))
        self.panels.append(panel)
        self.panel_layout.addWidget(panel)

    def select_output_directory(self):
        self.output_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if self.output_dir:
            logging.info(f"Output directory set to: {self.output_dir}")

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

        # Clear the queue before adding new items
        self.queue_manager.clear_queue()

        # Add all items from the panels to the queue
        for panel in self.panels:
            for item in panel.render_queue:
                self.queue_manager.add_to_queue(item)
            panel.render_queue.clear()
            panel.queue_list.clear()

        if not self.queue_manager.has_items():
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
        # Check if there are items left in the queue
        if not self.queue_manager.has_items():
            logging.info("All renders completed")
            return

        # Get the next item from the queue
        item = self.queue_manager.get_next_item()
        if item is None:
            logging.warning("No items left in the queue")
            return

        # Initialize video generator for the next item
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
        from core.video_player import VideoPlayer  # Delayed import to avoid circular dependency
        player = VideoPlayer(video_path)
        player.finished.connect(self.start_next_render)
        player.show()

    def update_queue_ui(self):
        """Optional: Update the UI when the queue changes."""
        logging.info("Queue updated.")
