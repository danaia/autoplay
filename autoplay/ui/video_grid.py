import os
import logging
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, 
                             QScrollArea, QMessageBox, QTableWidget, 
                             QTableWidgetItem, QHeaderView, QPushButton)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage
import cv2
from ui.video_player import VideoPlayer

class VideoGrid(QWidget):
    video_removed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.video_info = []  # List of tuples (video_path, generation_time)

    def init_ui(self):
        self.layout = QVBoxLayout(self)
        
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(['Thumbnail', 'Video Name', 'Generation Time', 'Actions'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        
        # Set row height
        self.table.verticalHeader().setDefaultSectionSize(150)
        
        self.layout.addWidget(self.table)

    def add_video(self, video_path, generation_time=None):
        logging.info(f"Attempting to add video: {video_path}")
        if not os.path.exists(video_path):
            logging.error(f"Video file does not exist: {video_path}")
            return

        if video_path not in [info[0] for info in self.video_info]:
            self.video_info.append((video_path, generation_time))
            logging.info(f"Video added to info list: {video_path}")
            self.update_table()
        else:
            logging.info(f"Video already in grid: {video_path}")

    def update_table(self):
        logging.info(f"Updating table with {len(self.video_info)} videos")
        self.table.setRowCount(len(self.video_info))
        for row, (video_path, generation_time) in enumerate(self.video_info):
            self.set_table_row(row, video_path, generation_time)
        self.table.resizeColumnsToContents()
        self.table.resizeRowsToContents()
        logging.info("Table update completed")

    def set_table_row(self, row, video_path, generation_time):
        logging.info(f"Setting row {row} for video: {video_path}")
        
        # Thumbnail
        thumbnail = self.create_video_thumbnail(video_path)
        self.table.setCellWidget(row, 0, thumbnail)

        # Video Name
        name_item = QTableWidgetItem(os.path.basename(video_path))
        name_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table.setItem(row, 1, name_item)

        # Generation Time
        time_item = QTableWidgetItem(self.format_time(generation_time))
        time_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table.setItem(row, 2, time_item)

        # Remove Button
        remove_button = QPushButton("Remove")
        remove_button.clicked.connect(lambda: self.remove_video(video_path))
        self.table.setCellWidget(row, 3, remove_button)

    def create_video_thumbnail(self, video_path):
        thumbnail_label = QLabel()
        thumbnail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            thumbnail_label.setPixmap(pixmap.scaled(200, 150, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            logging.info(f"Thumbnail created for: {video_path}")
        else:
            logging.error(f"Failed to create thumbnail for: {video_path}")
        cap.release()
        thumbnail_label.mousePressEvent = lambda event: self.play_video(video_path)
        return thumbnail_label



    def format_time(self, seconds):
        if seconds is None:
            return "Unknown"
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds:.2f}s"

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
                self.video_info = [info for info in self.video_info if info[0] != video_path]
                self.update_table()
                self.video_removed.emit(video_path)
            except Exception as e:
                QMessageBox.warning(self, 'Error', f"Failed to remove video: {str(e)}")