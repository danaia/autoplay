import os
from PyQt6.QtWidgets import QWidget, QGridLayout, QPushButton, QLabel, QVBoxLayout, QScrollArea, QMessageBox
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage
import cv2
from ui.video_player import VideoPlayer  # Import VideoPlayer from the ui package

class VideoGrid(QWidget):
    video_removed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.video_paths = []

    def init_ui(self):
        self.layout = QVBoxLayout(self)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.grid_layout = QGridLayout(self.scroll_content)
        self.scroll_area.setWidget(self.scroll_content)
        
        self.layout.addWidget(self.scroll_area)

    def add_video(self, video_path):
        if video_path not in self.video_paths:
            self.video_paths.append(video_path)
            self.update_grid()

    def update_grid(self):
        # Clear existing widgets
        for i in reversed(range(self.grid_layout.count())): 
            self.grid_layout.itemAt(i).widget().setParent(None)

        # Add video thumbnails to the grid
        for i, video_path in enumerate(self.video_paths):
            thumbnail = self.create_video_thumbnail(video_path)
            row, col = divmod(i, 4)  # 4 columns in the grid
            self.grid_layout.addWidget(thumbnail, row, col)

    def create_video_thumbnail(self, video_path):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Create a thumbnail from the first frame of the video
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            thumbnail_label = QLabel()
            thumbnail_label.setPixmap(pixmap.scaled(200, 150, Qt.AspectRatioMode.KeepAspectRatio))
            layout.addWidget(thumbnail_label)
        cap.release()

        # Add a play button
        play_button = QPushButton("Play")
        play_button.clicked.connect(lambda: self.play_video(video_path))
        layout.addWidget(play_button)

        # Add a remove button
        remove_button = QPushButton("Remove")
        remove_button.clicked.connect(lambda: self.remove_video(video_path))
        layout.addWidget(remove_button)

        return widget

    def play_video(self, video_path):
        self.video_player = VideoPlayer(video_path)
        self.video_player.show()

    def remove_video(self, video_path):
        reply = QMessageBox.question(self, 'Remove Video', 
                                     f"Are you sure you want to remove {os.path.basename(video_path)}?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, 
                                     QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.Yes:
            try:
                os.remove(video_path)
                self.video_paths.remove(video_path)
                self.update_grid()
                self.video_removed.emit(video_path)
            except Exception as e:
                QMessageBox.warning(self, 'Error', f"Failed to remove video: {str(e)}")