import sys
import os
import torch
import torchaudio
import soundfile as sf
import random
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QPushButton,
                             QSlider, QHBoxLayout, QLabel, QLineEdit, QFileDialog, QMessageBox, QProgressBar,
                             QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, QSizePolicy)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
from diffusers import StableAudioPipeline
from queue import Queue

class AudioGeneratorThread(QThread):
    progress_update = pyqtSignal(int, int, int, int)  # prompt_index, waveform_index, step, total_steps
    generation_complete = pyqtSignal(str, int, int)  # output_file, prompt_index, waveform_index
    all_complete = pyqtSignal()

    def __init__(self, prompts, negative_prompt, duration, num_inference_steps, audio_end_in_s, num_waveforms_per_prompt, use_random_seed, seed, project_name):
        super().__init__()
        self.prompts = prompts
        self.negative_prompt = negative_prompt
        self.duration = duration
        self.num_inference_steps = num_inference_steps
        self.audio_end_in_s = audio_end_in_s
        self.num_waveforms_per_prompt = num_waveforms_per_prompt
        self.use_random_seed = use_random_seed
        self.seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        self.project_name = project_name

    def run(self):
        pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16)
        pipe = pipe.to("cuda")

        for prompt_index, prompt in enumerate(self.prompts):
            for waveform_index in range(self.num_waveforms_per_prompt):
                generator = torch.Generator("cuda").manual_seed(self.seed if not self.use_random_seed else random.randint(0, 2**32 - 1))

                def callback(step, timestep, latents):
                    self.progress_update.emit(prompt_index, waveform_index, step, self.num_inference_steps)

                audio_output = pipe(
                    prompt=prompt,
                    negative_prompt=self.negative_prompt,
                    num_inference_steps=self.num_inference_steps,
                    audio_end_in_s=self.audio_end_in_s,
                    num_waveforms_per_prompt=1,  # Generate one waveform at a time
                    generator=generator,
                    callback=callback,
                    callback_steps=1
                ).audios

                audio = audio_output[0]
                output = audio.T.float().cpu().numpy()
                output_file = f"{self.project_name}_{prompt_index+1}_{waveform_index+1}.wav"
                sf.write(output_file, output, pipe.vae.sampling_rate)
                self.generation_complete.emit(output_file, prompt_index, waveform_index)

        self.all_complete.emit()

class AudioPlayerWidget(QWidget):
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.player.setSource(QUrl.fromLocalFile(self.file_path))

        layout = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.pause_button = QPushButton("Pause")
        self.rewind_button = QPushButton("Rewind")

        # Set size policy
        size_policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.play_button.setSizePolicy(size_policy)
        self.pause_button.setSizePolicy(size_policy)
        self.rewind_button.setSizePolicy(size_policy)

        # Set minimum size and styles
        self.play_button.setMinimumHeight(40)
        self.pause_button.setMinimumHeight(40)
        self.rewind_button.setMinimumHeight(40)

        self.play_button.setStyleSheet("font-size: 14px;")
        self.pause_button.setStyleSheet("font-size: 14px;")
        self.rewind_button.setStyleSheet("font-size: 14px;")

        self.play_button.clicked.connect(self.play_audio)
        self.pause_button.clicked.connect(self.pause_audio)
        self.rewind_button.clicked.connect(self.rewind_audio)

        layout.addWidget(self.play_button)
        layout.addWidget(self.pause_button)
        layout.addWidget(self.rewind_button)
        self.setLayout(layout)

    def play_audio(self):
        self.player.play()

    def pause_audio(self):
        self.player.pause()

    def rewind_audio(self):
        self.player.setPosition(0)

class AudioGeneratorApp(QWidget):
    ROW_HEIGHT = 80  # Variable to set the height of table rows

    def __init__(self):
        super().__init__()
        self.initUI()
        self.wav_files = []
        self.generation_thread = None

    def initUI(self):
        layout = QVBoxLayout()

        # Number of prompts
        prompt_layout = QHBoxLayout()
        self.prompt_label = QLabel("Number of prompts:")
        self.prompt_slider = QSlider(Qt.Orientation.Horizontal)
        self.prompt_slider.setMinimum(1)
        self.prompt_slider.setMaximum(20)  # Increased to 20
        self.prompt_slider.setValue(1)
        self.prompt_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.prompt_slider.setTickInterval(1)
        self.prompt_slider.valueChanged.connect(self.update_prompt_inputs)
        prompt_layout.addWidget(self.prompt_label)
        prompt_layout.addWidget(self.prompt_slider)
        layout.addLayout(prompt_layout)

        # Checkbox to copy first prompt to all
        self.copy_prompt_checkbox = QCheckBox("Copy first prompt to all")
        self.copy_prompt_checkbox.setChecked(False)
        self.copy_prompt_checkbox.stateChanged.connect(self.on_copy_prompt_checkbox_changed)
        layout.addWidget(self.copy_prompt_checkbox)

        # Container for prompt input fields
        self.prompt_inputs_layout = QVBoxLayout()
        layout.addLayout(self.prompt_inputs_layout)
        self.prompt_inputs = []  # Keep track of prompt input widgets
        self.update_prompt_inputs()

        # Advanced settings
        settings_layout = QHBoxLayout()

        self.negative_prompt_input = QLineEdit(self)
        self.negative_prompt_input.setPlaceholderText("Negative prompt")
        settings_layout.addWidget(QLabel("Negative prompt:"))
        settings_layout.addWidget(self.negative_prompt_input)

        self.num_inference_steps = QSpinBox(self)
        self.num_inference_steps.setRange(50, 500)
        self.num_inference_steps.setValue(200)
        settings_layout.addWidget(QLabel("Inference steps:"))
        settings_layout.addWidget(self.num_inference_steps)

        self.audio_duration = QDoubleSpinBox(self)
        self.audio_duration.setRange(1, 30)
        self.audio_duration.setValue(10)
        settings_layout.addWidget(QLabel("Duration (s):"))
        settings_layout.addWidget(self.audio_duration)

        self.num_waveforms = QSpinBox(self)
        self.num_waveforms.setRange(1, 4)
        self.num_waveforms.setValue(1)
        settings_layout.addWidget(QLabel("Waveforms:"))
        settings_layout.addWidget(self.num_waveforms)

        layout.addLayout(settings_layout)

        # Project Name Input
        project_name_layout = QHBoxLayout()
        self.project_name_input = QLineEdit(self)
        self.project_name_input.setPlaceholderText("Enter project name")
        project_name_layout.addWidget(QLabel("Project Name:"))
        project_name_layout.addWidget(self.project_name_input)
        layout.addLayout(project_name_layout)

        # Random seed checkbox
        self.random_seed_checkbox = QCheckBox("Use Random Seed")
        self.random_seed_checkbox.setChecked(True)
        layout.addWidget(self.random_seed_checkbox)

        # Generate button
        self.generate_button = QPushButton("Generate Audio")
        self.generate_button.clicked.connect(self.generate_audio_files)
        layout.addWidget(self.generate_button)

        # Table for displaying wav files
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["File", "Progress", "Controls", "Remove"])

        # Set column widths
        self.table.setColumnWidth(0, 250)
        self.table.setColumnWidth(1, 150)
        self.table.setColumnWidth(2, 300)
        self.table.setColumnWidth(3, 100)

        # Set default row height
        self.table.verticalHeader().setDefaultSectionSize(self.ROW_HEIGHT)

        layout.addWidget(self.table)

        self.setLayout(layout)
        self.setWindowTitle("Audio Generator")

    def update_prompt_inputs(self):
        while self.prompt_inputs_layout.count():
            widget = self.prompt_inputs_layout.takeAt(0).widget()
            if widget:
                widget.deleteLater()
        self.prompt_inputs = []

        for i in range(self.prompt_slider.value()):
            prompt_input = QLineEdit(self)
            prompt_input.setPlaceholderText(f"Prompt {i + 1}")
            self.prompt_inputs_layout.addWidget(prompt_input)
            self.prompt_inputs.append(prompt_input)

        # Update the copy prompt functionality
        self.on_copy_prompt_checkbox_changed(self.copy_prompt_checkbox.checkState())

    def on_copy_prompt_checkbox_changed(self, state):
        if self.copy_prompt_checkbox.isChecked():
            if self.prompt_inputs:
                self.prompt_inputs[0].textChanged.connect(self.copy_first_prompt)
                self.copy_first_prompt(self.prompt_inputs[0].text())
        else:
            if self.prompt_inputs:
                try:
                    self.prompt_inputs[0].textChanged.disconnect(self.copy_first_prompt)
                except TypeError:
                    pass  # Was not connected

    def copy_first_prompt(self, text):
        for prompt_input in self.prompt_inputs[1:]:
            prompt_input.setText(text)

    def generate_audio_files(self):
        prompts = [prompt_input.text() for prompt_input in self.prompt_inputs]

        if not all(prompts):
            QMessageBox.warning(self, "Input Error", "Please fill in all prompt fields.")
            return

        negative_prompt = self.negative_prompt_input.text()
        num_inference_steps = self.num_inference_steps.value()
        audio_end_in_s = self.audio_duration.value()
        num_waveforms_per_prompt = self.num_waveforms.value()
        use_random_seed = self.random_seed_checkbox.isChecked()
        project_name = self.project_name_input.text().strip()

        if not project_name:
            QMessageBox.warning(self, "Input Error", "Please enter a project name.")
            return

        seed = random.randint(0, 2**32 - 1) if use_random_seed else None

        # Clear existing rows in the table
        self.table.setRowCount(0)

        # Prepare table rows and mapping
        self.row_mapping = {}  # (prompt_index, waveform_index) -> row_index
        row_index = 0
        for prompt_index in range(len(prompts)):
            for waveform_index in range(num_waveforms_per_prompt):
                self.table.insertRow(row_index)
                self.table.setRowHeight(row_index, self.ROW_HEIGHT)  # Set the row height
                self.table.setItem(row_index, 0, QTableWidgetItem(f"Generating..."))

                progress_bar = QProgressBar()
                self.table.setCellWidget(row_index, 1, progress_bar)

                # Empty controls cell for now
                self.table.setCellWidget(row_index, 2, QWidget())

                # Remove button placeholder
                self.table.setCellWidget(row_index, 3, QWidget())

                self.row_mapping[(prompt_index, waveform_index)] = row_index
                row_index += 1

        self.generate_button.setEnabled(False)

        self.generation_thread = AudioGeneratorThread(prompts, negative_prompt, audio_end_in_s, num_inference_steps,
                                                      audio_end_in_s, num_waveforms_per_prompt, use_random_seed, seed, project_name)
        self.generation_thread.progress_update.connect(self.update_progress)
        self.generation_thread.generation_complete.connect(self.on_generation_complete)
        self.generation_thread.all_complete.connect(self.on_all_complete)
        self.generation_thread.start()

    def update_progress(self, prompt_index, waveform_index, step, total_steps):
        row = self.row_mapping.get((prompt_index, waveform_index))
        if row is not None:
            progress_bar = self.table.cellWidget(row, 1)
            if progress_bar:
                progress_bar.setValue(int((step / total_steps) * 100))

    def on_generation_complete(self, output_file, prompt_index, waveform_index):
        row = self.row_mapping.get((prompt_index, waveform_index))
        if row is not None:
            self.table.setItem(row, 0, QTableWidgetItem(output_file))

            # Set progress bar to 100%
            progress_bar = self.table.cellWidget(row, 1)
            if progress_bar:
                progress_bar.setValue(100)

            # Add controls
            controls_widget = AudioPlayerWidget(output_file)
            self.table.setCellWidget(row, 2, controls_widget)

            # Add remove button
            remove_button = QPushButton("Remove")
            remove_button.setMinimumHeight(40)
            remove_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            remove_button.setStyleSheet("font-size: 14px;")
            remove_button.clicked.connect(lambda _, r=row: self.remove_audio(r))
            self.table.setCellWidget(row, 3, remove_button)

    def on_all_complete(self):
        self.generate_button.setEnabled(True)
        QMessageBox.information(self, "Generation Complete", "All audio files have been generated.")

    def remove_audio(self, row):
        file_item = self.table.item(row, 0)
        if file_item:
            file_path = file_item.text()
            if os.path.exists(file_path):
                os.remove(file_path)
            self.table.removeRow(row)

    def closeEvent(self, event):
        # Stop the generation thread if it's running
        if self.generation_thread and self.generation_thread.isRunning():
            self.generation_thread.terminate()
            self.generation_thread.wait()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = AudioGeneratorApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
