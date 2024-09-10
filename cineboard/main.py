import sys
from PyQt6.QtWidgets import QApplication
from ui.main_window import TextToVideoGUI

def main():
    app = QApplication(sys.argv)
    window = TextToVideoGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
