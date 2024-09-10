from PyQt6.QtCore import QObject, pyqtSignal

class QueueManager(QObject):
    """Manages the queue of video generation tasks."""
    queue_updated = pyqtSignal()  # Signal to notify when the queue changes

    def __init__(self):
        super().__init__()
        self.queue = []

    def add_to_queue(self, item):
        """Adds a new item to the queue."""
        self.queue.append(item)
        self.queue_updated.emit()  # Emit signal to notify queue has changed

    def get_next_item(self):
        """Returns the next item in the queue."""
        if self.queue:
            return self.queue.pop(0)
        return None

    def has_items(self):
        """Returns True if there are items left in the queue."""
        return len(self.queue) > 0

    def clear_queue(self):
        """Clears the queue."""
        self.queue.clear()
        self.queue_updated.emit()
