import logging
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QProgressBar, QLabel, QScrollArea, QSpinBox, QFileDialog, QTextEdit, QComboBox, QSlider
from PyQt6.QtCore import Qt, QSettings, QThread, pyqtSignal
from ui.video_grid import VideoGrid
from ui.resource_monitor import ResourceMonitor
from core.video_generator import VideoGenerator
from core.dependency_installer import DependencyInstaller
from utils.queue_manager import QueueManager
from openai import OpenAI
from ui.prompt_panel import PromptPanel
import time
import os

# Initialize OpenAI API client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class PromptGenerationWorker(QThread):
    prompts_generated = pyqtSignal(list)
    error_occurred = pyqtSignal(str)

    def __init__(self, global_prompt, num_panels):
        super().__init__()
        self.global_prompt = global_prompt
        self.num_panels = num_panels

    def run(self):
        try:
            prompts = []
            for i in range(self.num_panels):
                response = client.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=[
                        {"role": "system", "content": "You are a creative assistant that generates prompts for text-to-video AI models. keep token low. keep the prompt under 220 on the return prompt."},
                        {"role": "user", "content": f"Based on the following global intention, generate a creative and detailed prompt for this cinematic scene {i+1} of a text-to-video AI model: {self.global_prompt}"}
                    ]
                )
                prompts.append(response.choices[0].message.content.strip())
            self.prompts_generated.emit(prompts)
        except Exception as e:
            self.error_occurred.emit(str(e))
            self.error_occurred.emit(f"Error generating prompts: {str(e)}")
            self.error_occurred.emit(str(e))

class TextToVideoGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.panels = []
        self.output_dir = ""
        self.queue_manager = QueueManager()
        self.settings = QSettings("MicroFilm.AI", "AutoPlay")
        self.video_grid = VideoGrid()  # Initialize VideoGrid here
        self.init_ui()
        self.open_resource_monitor()
        self.load_settings()
        self.video_grid.video_removed.connect(self.on_video_removed)
        self.queue_manager.queue_updated.connect(self.update_queue_ui)


        self.queue_manager.queue_updated.connect(self.update_queue_ui)

    def init_ui(self):
        main_layout = QVBoxLayout()

        # Global prompt and panel count slider side by side
        global_prompt_and_slider_layout = QHBoxLayout()  # New layout to hold both the prompt and slider

        # Global prompt input area
        global_prompt_layout = QVBoxLayout()  # Use QVBoxLayout to keep the prompt label and input vertically aligned
        # self.global_prompt_label = QLabel("Global Prompt:")
        # global_prompt_layout.addWidget(self.global_prompt_label)
        self.global_prompt_input = QTextEdit()
        self.global_prompt_input.setPlaceholderText("Enter your global intention here...")
        self.global_prompt_input.setMaximumHeight(100)
        global_prompt_layout.addWidget(self.global_prompt_input)

        # GPT model selection (can go below the global prompt)
        gpt_model_layout = QHBoxLayout()  # Keep GPT model selection on the same row as the input
        # gpt_model_layout.addWidget(QLabel("GPT Model:"))
        # self.gpt_model_combo = QComboBox()
        # self.gpt_model_combo.addItems(["gpt-4-turbo-preview", "gpt-4", "gpt-4o-mini"])
        # gpt_model_layout.addWidget(self.gpt_model_combo)
        global_prompt_layout.addLayout(gpt_model_layout)

        # Add global prompt layout to the main layout
        global_prompt_and_slider_layout.addLayout(global_prompt_layout)

        # Panel count slider (placed to the right of the global prompt)
        panel_control_layout = QVBoxLayout()  # Use QVBoxLayout to stack the label and slider vertically
        # panel_control_layout.addWidget(QLabel(""))
        self.panel_count_slider = QSlider(Qt.Orientation.Vertical)  # Set slider to vertical orientation
        self.panel_count_slider.setRange(1, 10)
        self.panel_count_slider.setValue(1)
        self.panel_count_slider.setMinimumHeight(150)  # Set minimum height to give vertical space
        self.panel_count_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.panel_count_slider.setTickInterval(1)
        self.panel_count_slider.valueChanged.connect(self.update_panel_count)
        panel_control_layout.addWidget(self.panel_count_slider)
        self.panel_count_label = QLabel("1")
        # panel_control_layout.addWidget(self.panel_count_label)

        # Add the panel control layout (slider) to the global prompt and slider layout
        global_prompt_and_slider_layout.addLayout(panel_control_layout)

        # Add this combined layout to the main layout
        main_layout.addLayout(global_prompt_and_slider_layout)

        # Remaining UI elements (unchanged)
        self.generate_panel_prompts_button = QPushButton('Generate Panel Prompts')
        self.generate_panel_prompts_button.clicked.connect(self.generate_panel_prompts)
        main_layout.addWidget(self.generate_panel_prompts_button)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.panel_layout = QHBoxLayout(self.scroll_content)
        self.scroll_area.setWidget(self.scroll_content)
        main_layout.addWidget(self.scroll_area)

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

        # Add the VideoGrid component
        self.video_grid = VideoGrid()
        main_layout.addWidget(self.video_grid)

        self.setLayout(main_layout)
        self.setWindowTitle('AutoPlay By MicroFilm.AI')
        self.setGeometry(100, 100, 1200, 800)

    def open_resource_monitor(self):
        try:
            self.resource_monitor = ResourceMonitor()  # Assuming ResourceMonitor is a valid widget in your app
            self.resource_monitor.show()
            logging.info("Resource monitor opened successfully.")
        except Exception as e:
            logging.error(f"Failed to open resource monitor: {e}")

    # def generate_panel_prompts(self):
    #     global_prompt = self.global_prompt_input.toPlainText()
    #     if not global_prompt:
    #         logging.warning("Please enter a global prompt first.")
    #         return

    #     gpt_model = self.gpt_model_combo.currentText()
    #     panel_count = len(self.panels)
        
    #     if panel_count == 0:
    #         logging.warning("No panels available to generate prompts for.")
    #         return

    #     self.worker = PromptGenerationWorker(global_prompt, panel_count, gpt_model)
    #     self.worker.prompts_generated.connect(self.handle_prompts_generated)
    #     self.worker.error_occurred.connect(self.handle_error)
    #     self.worker.start()

    def handle_prompts_generated(self, prompts):
        if len(prompts) != len(self.panels):
            logging.warning("Mismatch between the number of generated prompts and panels.")
            return

        # Assign each generated prompt to the corresponding panel
        for i, prompt in enumerate(prompts):
            panel = self.panels[i]
            logging.info(f"Assigning prompt to panel {i+1}: {prompt}")
            panel.text_edit.setPlainText(prompt)

    def add_panel(self):
        panel = PromptPanel(len(self.panels))
        self.panels.append(panel)
        self.panel_layout.addWidget(panel)

    def update_panel_count(self, count):
        current_count = len(self.panels)
        if count > current_count:
            for _ in range(count - current_count):
                self.add_panel()
        elif count < current_count:
            for _ in range(current_count - count):
                panel = self.panels.pop()
                self.panel_layout.removeWidget(panel)
                panel.deleteLater()

    def select_output_directory(self):
        self.output_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if self.output_dir:
            logging.info(f"Output directory set to: {self.output_dir}")

    def process_all_queues(self):
        if not self.output_dir:
            logging.warning("Please select an output directory first")
            return

        self.queue_manager.clear_queue()

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
        if not self.queue_manager.has_items():
            logging.info("All renders completed")
            return

        item = self.queue_manager.get_next_item()
        if item is None:
            logging.warning("No items left in the queue")
            return

        logging.info(f"Starting render for item: {item['project_name']}")
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
        self.generator.finished.connect(self.on_video_generation_finished)
        self.generator.progress.connect(self.progress_bar.setValue)
        self.generator.time_estimate.connect(self.update_time_estimate)
        self.generator.video_generated.connect(self.on_video_generated)
        self.generator.start()

    def generate_panel_prompts(self):
        global_prompt = self.global_prompt_input.toPlainText()
        if not global_prompt:
            logging.warning("Please enter a global prompt first.")
            return

        self.worker = PromptGenerationWorker(global_prompt, len(self.panels))
        self.worker.prompts_generated.connect(self.handle_prompts_generated)
        self.worker.error_occurred.connect(self.handle_error)
        self.worker.start()

    def on_video_generation_finished(self):
        logging.info("Video generation completed")
        self.progress_bar.setValue(0)
        self.time_estimate_label.setText("Estimated time remaining: N/A")
        self.start_next_render()

    def update_time_estimate(self, estimate):
        self.time_estimate_label.setText(estimate)

    def on_video_generated(self, video_path, generation_time):
        logging.info(f"Video generated: {video_path}, Generation time: {generation_time:.2f} seconds")
        if os.path.exists(video_path):
            self.video_grid.add_video(video_path, generation_time)
        else:
            logging.error(f"Generated video file does not exist: {video_path}")

    def on_video_removed(self, video_path):
        logging.info(f"Video removed: {video_path}")


    def show_video(self, video_path):
        self.video_grid.add_video(video_path)
        self.start_next_render()

    def update_queue_ui(self):
        logging.info("Queue updated.")

    def save_settings(self):
        self.settings.setValue("output_dir", self.output_dir)
        self.settings.setValue("panel_count", len(self.panels))
        self.settings.setValue("global_prompt", self.global_prompt_input.toPlainText())
        for i, panel in enumerate(self.panels):
            panel.save_settings(self.settings, i)

    def load_settings(self):
        self.output_dir = self.settings.value("output_dir", "")
        panel_count = int(self.settings.value("panel_count", 1))
        self.update_panel_count(panel_count)
        self.global_prompt_input.setPlainText(self.settings.value("global_prompt", ""))
        # self.gpt_model_combo.setCurrentText(self.settings.value("gpt_model", "gpt-3.5-turbo"))
        for i, panel in enumerate(self.panels):
            panel.load_settings(self.settings, i)

    def closeEvent(self, event):
        self.save_settings()
        event.accept()

    def handle_error(self, error_message):
        logging.error(f"Error generating prompts: {error_message}")
        QMessageBox.critical(self, "Prompt Generation Error", error_message)

    def start_video_generation(self):
        # ... code to set up video generation parameters ...
        self.video_generator = VideoGenerator(text, num_inference_steps, guidance_scale, num_frames, project_name, sequence_number, output_dir, num_videos)
        self.video_generator.finished.connect(self.on_video_generation_finished)
        self.video_generator.video_generated.connect(self.on_video_generated)
        self.video_generator.start()
