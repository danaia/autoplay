import sys
import os
import torch
import torchaudio
import soundfile as sf
import random
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QPushButton,
                             QSlider, QHBoxLayout, QLabel, QLineEdit, QFileDialog, QMessageBox, QProgressBar,
                             QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
from diffusers import StableAudioPipeline
from queue import Queue

class AudioGeneratorThread(QThread):
    progress_update = pyqtSignal(int, int)
    generation_complete = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, prompt, negative_prompt, duration, num_inference_steps, audio_end_in_s, num_waveforms_per_prompt, output_file, use_random_seed, seed):
        super().__init__()
        self.pipe = pipe
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.duration = duration
        self.num_inference_steps = num_inference_steps
        self.audio_end_in_s = audio_end_in_s
        self.num_waveforms_per_prompt = num_waveforms_per_prompt
        self.output_file = output_file
        self.use_random_seed = use_random_seed
        self.seed = seed if seed is not None else random.randint(0, 2**32 - 1)

    def run(self):
        try:
            # Set random seed
            generator = torch.Generator("cuda").manual_seed(self.seed)
            
            def callback(step, timestep, latents):
                self.progress_update.emit(step, self.num_inference_steps)

            # Generating audio using the pipeline
            audio = self.pipe(
                prompt=self.prompt,
                negative_prompt=self.negative_prompt,
                num_inference_steps=self.num_inference_steps,
                audio_end_in_s=self.audio_end_in_s,
                num_waveforms_per_prompt=self.num_waveforms_per_prompt,
                generator=generator,
                callback=callback,
                callback_steps=1
            ).audios

            # Save audio file
            output = audio[0].T.float().cpu().numpy()
            sf.write(self.output_file, output, self.pipe.vae.sampling_rate)

            self.generation_complete.emit(self.output_file)
        except Exception as e:
            self.error_occurred.emit(f"Error during generation: {str(e)}")


class AudioGeneratorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

        self.file_path = ""
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.wav_files = []
        self.generation_threads = []
        self.prompt_queue = Queue()

        # Load the pipeline once during initialization
        self.pipe = self.initialize_pipeline()

    def initialize_pipeline(self):
        try:
            # Initialize the StableAudioPipeline
            pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16)
            pipe = pipe.to("cuda")
            return pipe
        except Exception as e:
            QMessageBox.critical(self, "Pipeline Error", f"Failed to load the pipeline: {str(e)}")
            sys.exit(1)

    def initUI(self):
        layout = QVBoxLayout()

        # Number of prompts
        prompt_layout = QHBoxLayout()
        self.prompt_label = QLabel("Number of prompts:")
        self.prompt_slider = QSlider(Qt.Orientation.Horizontal)
        self.prompt_slider.setMinimum(1)
        self.prompt_slider.setMaximum(5)
        self.prompt_slider.setValue(1)
        self.prompt_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.prompt_slider.setTickInterval(1)
        self.prompt_slider.valueChanged.connect(self.update_prompt_inputs)
        prompt_layout.addWidget(self.prompt_label)
        prompt_layout.addWidget(self.prompt_slider)
        layout.addLayout(prompt_layout)

        # Container for prompt input fields
        self.prompt_inputs_layout = QVBoxLayout()
        layout.addLayout(self.prompt_inputs_layout)
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
        layout.addWidget(self.table)

        self.setLayout(layout)
        self.setWindowTitle("Audio Generator")

    def update_prompt_inputs(self):
        while self.prompt_inputs_layout.count():
            widget = self.prompt_inputs_layout.takeAt(0).widget()
            if widget:
                widget.deleteLater()

        for i in range(self.prompt_slider.value()):
            prompt_input = QLineEdit(self)
            prompt_input.setPlaceholderText(f"Prompt {i + 1}")
            self.prompt_inputs_layout.addWidget(prompt_input)

    def generate_audio_files(self):
        prompts = [self.prompt_inputs_layout.itemAt(i).widget().text() for i in range(self.prompt_inputs_layout.count())]

        if not all(prompts):
            QMessageBox.warning(self, "Input Error", "Please fill in all prompt fields.")
            return

        negative_prompt = self.negative_prompt_input.text()
        num_inference_steps = self.num_inference_steps.value()
        audio_end_in_s = self.audio_duration.value()
        num_waveforms_per_prompt = self.num_waveforms.value()
        use_random_seed = self.random_seed_checkbox.isChecked()

        seed = random.randint(0, 2**32 - 1) if use_random_seed else None

        for i, prompt in enumerate(prompts):
            output_file = f"generated_audio_{i}_{random.randint(1000, 9999)}.wav"
            self.prompt_queue.put((prompt, negative_prompt, num_inference_steps, audio_end_in_s, num_waveforms_per_prompt, output_file, use_random_seed, seed))

        if self.prompt_queue.qsize() == 1:
            self.process_next_prompt()

    def process_next_prompt(self):
        if not self.prompt_queue.empty():
            prompt, negative_prompt, num_inference_steps, audio_end_in_s, num_waveforms_per_prompt, output_file, use_random_seed, seed = self.prompt_queue.get()

            thread = AudioGeneratorThread(self.pipe, prompt, negative_prompt, audio_end_in_s, num_inference_steps,
                                          audio_end_in_s, num_waveforms_per_prompt, output_file, use_random_seed, seed)
            thread.progress_update.connect(self.update_progress)
            thread.generation_complete.connect(self.on_generation_complete)
            thread.error_occurred.connect(self.on_error_occurred)
            thread.start()
            self.generation_threads.append(thread)

    def update_progress(self, step, total):
        # Update progress here
        pass

    def on_generation_complete(self, output_file):
        # Handle audio generation completion here
        QMessageBox.information(self, "Generation Complete", f"Audio saved to {output_file}")

    def on_error_occurred(self, error_message):
        QMessageBox.critical(self, "Error", error_message)


def main():
    app = QApplication(sys.argv)
    window = AudioGeneratorApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
