import os
import torch
import time
import gc
from PyQt6.QtCore import QThread, pyqtSignal
from diffusers import CogVideoXPipeline, CogVideoXDPMScheduler
from diffusers.utils import export_to_video
import logging

import os
import torch
import time
import gc
import subprocess  # For running FFmpeg commands
from PyQt6.QtCore import QThread, pyqtSignal
from diffusers import CogVideoXPipeline, CogVideoXDPMScheduler
from diffusers.utils import export_to_video
import logging

class VideoGenerator(QThread):
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    time_estimate = pyqtSignal(str)
    video_generated = pyqtSignal(str, float)  # Now emits video path and generation time

    def __init__(
        self,
        text,
        num_inference_steps,
        guidance_scale,
        num_frames,
        project_name,
        sequence_number,
        output_dir,
        num_videos,
    ):
        super().__init__()
        self.text = text
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.num_frames = num_frames
        self.project_name = project_name
        self.sequence_number = sequence_number
        self.output_dir = output_dir
        self.num_videos = num_videos
        self.generation_start_time = None

    def run(self):
        try:
            # Initialize the pipeline with bfloat16 precision
            pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16)

            # Configure the scheduler
            pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

            # Enable memory optimization techniques
            pipe.enable_model_cpu_offload()
            pipe.enable_sequential_cpu_offload()
            pipe.vae.enable_slicing()
            pipe.vae.enable_tiling()

            start_time = time.time()
            for video_idx in range(self.num_videos):
                self.generation_start_time = time.time()  # Start time for each individual video

                # Generate video frames
                result = pipe(
                    prompt=self.text,
                    num_videos_per_prompt=1,
                    num_inference_steps=self.num_inference_steps,
                    num_frames=self.num_frames,
                    height=480,
                    width=720,
                    use_dynamic_cfg=True,
                    guidance_scale=self.guidance_scale,
                    generator=torch.Generator().manual_seed(42 + video_idx),
                )
                video_frames = result.frames[0]  # Get the video frames

                # Update progress
                progress = int(((video_idx + 1) / self.num_videos) * 100)
                self.progress.emit(progress)

                # Calculate and emit time estimate
                elapsed_time = time.time() - start_time
                estimated_total_time = elapsed_time / (video_idx + 1) * self.num_videos
                remaining_time = estimated_total_time - elapsed_time
                self.time_estimate.emit(f"Estimated time remaining: {remaining_time:.2f} seconds")

                # Save the video frames to a temporary file
                temp_output_path = os.path.join(
                    self.output_dir,
                    f"temp_{self.project_name}_{self.sequence_number}_video_{video_idx + 1}.mp4",
                )
                export_to_video(video_frames, temp_output_path, fps=8)

                # Re-encode the video using FFmpeg
                output_path = os.path.join(
                    self.output_dir,
                    f"{self.project_name}_{self.sequence_number}_video_{video_idx + 1}.mp4",
                )
                ffmpeg_command = [
                    'ffmpeg',
                    '-y',  # Overwrite output files without asking
                    '-i', temp_output_path,
                    '-c:v', 'libx264',  # Video codec: H.264
                    '-pix_fmt', 'yuv420p',  # Pixel format
                    '-r', '24',  # Frame rate
                    output_path,
                ]
                # Run the FFmpeg command
                subprocess.run(ffmpeg_command, check=True)

                # Remove the temporary file
                os.remove(temp_output_path)

                generation_time = time.time() - self.generation_start_time
                self.video_generated.emit(output_path, generation_time)

                # Clear memory after each video generation
                torch.cuda.empty_cache()
                gc.collect()

            self.finished.emit()

        except Exception as e:
            logging.error(f"Error generating video: {e}")
        finally:
            torch.cuda.empty_cache()
            gc.collect()
