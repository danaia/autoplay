import logging
from PyQt6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QTextEdit, QSlider, QSpinBox, QPushButton, QListWidget
from PyQt6.QtCore import Qt


class PromptPanel(QFrame):
    def __init__(self, panel_id):
        super().__init__()
        self.panel_id = panel_id
        self.render_queue = []
        self.current_sequence = 0
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Project name input
        project_layout = QHBoxLayout()
        project_layout.addWidget(QLabel("Project Name:"))
        self.project_name_input = QLineEdit()
        project_layout.addWidget(self.project_name_input)
        layout.addLayout(project_layout)

        # Text prompt input
        self.text_edit = QTextEdit()
        layout.addWidget(self.text_edit)

        # Inference steps slider
        inference_layout = QHBoxLayout()
        inference_layout.addWidget(QLabel("Inference Steps:"))
        self.inference_steps_spinbox = QSpinBox()
        self.inference_steps_spinbox.setRange(1, 100)
        self.inference_steps_spinbox.setValue(50)
        inference_layout.addWidget(self.inference_steps_spinbox)
        layout.addLayout(inference_layout)

        # Guidance scale slider
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

        # Number of frames input
        frames_layout = QHBoxLayout()
        frames_layout.addWidget(QLabel("Number of Frames:"))
        self.frames_spinbox = QSpinBox()
        self.frames_spinbox.setRange(1, 100)
        self.frames_spinbox.setValue(49)
        frames_layout.addWidget(self.frames_spinbox)
        layout.addLayout(frames_layout)

        # Number of videos input
        videos_layout = QHBoxLayout()
        videos_layout.addWidget(QLabel("Number of Videos:"))
        self.videos_spinbox = QSpinBox()
        self.videos_spinbox.setRange(1, 10)
        self.videos_spinbox.setValue(1)
        videos_layout.addWidget(self.videos_spinbox)
        layout.addLayout(videos_layout)

        # Add to queue button
        self.queue_button = QPushButton('Add to Queue')
        self.queue_button.clicked.connect(self.add_to_queue)
        layout.addWidget(self.queue_button)

        # Queue list to display added projects
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
