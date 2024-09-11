# AutoPlay: Text to Video Generator

## Overview

AutoPlay is a PyQt6-based application that generates videos from text prompts using advanced AI models. It features a multi-panel interface for managing multiple video generation tasks, a video grid for displaying and managing generated videos, and a resource monitor for tracking system performance.

## Features

- Multi-panel interface for managing multiple video generation tasks
- AI-powered video generation from text prompts
- Video grid for displaying and managing generated videos
- Resource monitor for tracking system performance
- Queue management for processing multiple video generation tasks
- Automatic dependency installation

## Prerequisites

- Python 3.8 or higher
- PyQt6
- OpenCV
- Other dependencies (will be installed automatically)

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/danaia/autoplay.git
   cd autoplay
   ```

2. Create a virtual environment (optional but recommended):

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the main application:

   ```
   python main.py
   ```

2. Use the interface to:
   - Add text prompts for video generation
   - Select output directory for generated videos
   - Process video generation tasks
   - View and manage generated videos in the video grid

## Project Structure

```
autoplay/
├── core/
│   ├── ai_interface.py
│   ├── dependency_installer.py
│   └── video_generator.py
├── services/
│   └── pipeline_service.py
├── ui/
│   ├── main_window.py
│   ├── prompt_panel.py
│   ├── resource_monitor.py
│   ├── video_grid.py
│   └── video_player.py
├── utils/
│   └── logger.py
├── vids/
├── main.py
├── queue_manager.py
├── text-to-video.py
├── .gitignore
├── install.txt
└── requirements.txt
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) for the GUI framework
- [OpenCV](https://opencv.org/) for video processing
- [CogVideoX](https://github.com/THUDM/CogVideo) for the AI video generation model
