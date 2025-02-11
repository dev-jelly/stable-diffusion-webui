# Author: Jelly Choi
# This code is based on https://github.com/Filarius/stable-diffusion-webui/blob/master/scripts/vid2vid.py

import os
import subprocess
import sys

import random

import modules.scripts as scripts
import gradio as gr

from modules.processing import Processed, process_images
from PIL import Image

from modules.shared import state
from modules import processing

from subprocess import Popen, PIPE, TimeoutExpired
import numpy as np
from datetime import datetime
import time


class Script(scripts.Script):
    def title(self):
        return '[Jelly] Video to video With Audio 5'

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        input_path = gr.Textbox(label="Input file path", lines=1)

        # output_path = gr.Textbox(label="Output file path", lines=1)
        crf = gr.Slider(
            label="CRF (quality, less is better, x264 param)",
            minimum=1,
            maximum=40,
            step=1,
            value=24,
        )
        fps = gr.Slider(
            label="FPS",
            minimum=1,
            maximum=60,
            step=1,
            value=24,
        )

        with gr.Row():
            seed_walk = gr.Slider(
                minimum=-20, maximum=20, step=1, label="Seed step size", value=0
            )
            seed_max_distance = gr.Slider(
                minimum=1, maximum=200, step=1, label="Seed max distance", value=10
            )

        with gr.Row():
            start_time = gr.Textbox(label="Start time", value="00:00:00", lines=1)
            end_time = gr.Textbox(label="End time", value="00:00:00", lines=1)

        with gr.Row():
            use_timestamp = gr.Checkbox(label="Use Timestamp (Filename)", value=True)
            original_resolution = gr.Checkbox(label="Keep Original Resolution", value=True)

        return [
            input_path,
            crf,
            fps,
            seed_walk,
            seed_max_distance,
            start_time,
            end_time,
            use_timestamp,
            original_resolution
        ]

    def run(
            self,
            p,
            input_path,
            crf,
            fps,
            seed_walk,
            seed_max_distance,
            start_time,
            end_time,
            use_timestamp,
            original_resolution
    ):
        processing.fix_seed(p)

        input_path = input_path.replace('"', "")

        # p.subseed_strength == 0
        initial_seed = p.seed
        initial_info = None

        p.do_not_save_grid = True
        p.do_not_save_samples = True
        p.batch_count = 1

        start_time = start_time.strip()
        end_time = end_time.strip()

        if start_time == "":
            start_time = "00:00:00"
        if end_time == "00:00:00":
            end_time = ""

        time_interval = (
            f"-ss {start_time}" + f" -to {end_time}" if len(end_time) else ""
        )
        import modules

        path = modules.paths.script_path
        save_dir = "outputs/img2img-video-with-audio"
        ffmpeg.install(path, save_dir)

        input_file = os.path.normpath(input_path.strip())

        if original_resolution:
            ffmpeg(f"ffmpeg/ffprobe.exe -show_streams -select_streams v {input_file}")
            width, height = get_width_height(input_file)
            p.width = width
            p.height = height

        decoder = ffmpeg(
            " ".join(
                [
                    "ffmpeg/ffmpeg -y -loglevel panic",
                    f'{time_interval} -i "{input_file}"',
                    f"-s:v {p.width}x{p.height} -r {fps}",
                    "-f image2pipe -pix_fmt rgb24",
                    "-vcodec rawvideo -",
                ]
            ),
            use_stdout=True,
        )
        decoder.start()
        output_file = input_file.split("\\")[-1]
        encoder = ffmpeg(
            " ".join(
                [
                    "ffmpeg/ffmpeg -y -loglevel panic",
                    "-f rawvideo -pix_fmt rgb24",
                    f"-s:v {p.width}x{p.height} -r {fps}",
                    "-i - -c:v libx264 -preset fast -movflags faststart",
                    f'-crf {crf} "{save_dir}/{output_file}"',
                ]
            ),
            use_stdin=True,
        )
        encoder.start()

        batch = []
        seed = initial_seed

        frame = 1
        if len(end_time) > 0:
            seconds = ffmpeg.seconds(end_time) - ffmpeg.seconds(start_time)
            loops = seconds * int(fps)
            state.job_count = loops
        else:
            loops = None

        pull_count = p.width * p.height * 3
        raw_image = decoder.readout(pull_count)

        while raw_image is not None and len(raw_image) > 0:
            if state.interrupted or state.skipped:
                decoder._process.kill()
                encoder._process.kill()
                return Processed(p, [], p.seed, initial_info)

            image_PIL = Image.fromarray(
                np.uint8(raw_image).reshape((p.height, p.width, 3)), mode="RGB"
            )
            batch.append(image_PIL)

            if len(batch) == p.batch_size:
                if seed_walk != 0:
                    seed_step = (
                        random.randint(0, seed_walk)
                        if seed_walk >= 0
                        else random.randint(seed_walk, 0)
                    )
                    if (
                            seed_step != 0
                            or abs(seed + seed_step - initial_seed) <= seed_max_distance
                    ):
                        seed = seed + seed_step

                p.seed = [seed for _ in batch]
                p.init_images = batch
                batch = []

                state.job = f"{frame}/{int(loops)}|{seed}/{seed_step}"
                proc = process_images(p)
                if initial_info is None:
                    initial_info = proc.info

                for output in proc.images:
                    encoder.write(np.asarray(output))

            raw_image = decoder.readout(pull_count)
            frame += 1

        try:
            encoder._process.communicate(timeout=2)
        except TimeoutExpired:
            encoder._process.kill()
            encoder._process.communicate()

        try:
            decoder._process.communicate(timeout=2)
        except TimeoutExpired:
            encoder._process.kill()
            decoder._process.communicate()

        if use_timestamp:
            dt = datetime.now()
            time_stamp = dt.strftime("%Y%m%d_%H%M%S")
        else:
            time_stamp = "[Audio]"

        command = " ".join(
            [
                f'ffmpeg/ffmpeg -y -ss {start_time} -to {end_time} -i "{input_file}"',
                f'-i "{save_dir}/{output_file}" -movflags faststart',
                f'-map 0:a? -map 1:v "{save_dir}/{time_stamp}-{output_file}"',
            ]
        )
        subprocess.run(command, shell=True)

        return Processed(p, [], p.seed, initial_info)


def get_width_height(filename):
    result = subprocess.run(f"ffmpeg/ffprobe.exe -show_streams -select_streams v {filename}", text=True,
                            capture_output=True)

    width_line: str = [line for line in result.stdout.splitlines() if "width" in line][0]
    width: int = int(width_line.split("=")[1])
    w = width // 64

    height_line: str = [line for line in result.stdout.splitlines() if "height" in line][0]
    height: int = int(height_line.split("=")[1])
    h = height // 64
    if w == 0 or h == 0:
        return 64, 64

    while w * 64 > 2048 or h * 64 > 2048:
        w -= 1
        h -= 1

    return w * 64, h * 64


class ffmpeg:
    def __init__(
            self,
            cmdln,
            use_stdin=False,
            use_stdout=False,
            use_stderr=False,
            print_to_console=True,
    ):
        self._process = None
        self._cmdln = cmdln
        self._stdin = None

        if use_stdin:
            self._stdin = PIPE

        self._stdout = None
        self._stderr = None

        if print_to_console:
            self._stderr = sys.stdout
            self._stdout = sys.stdout

        if use_stdout:
            self._stdout = PIPE

        if use_stderr:
            self._stderr = PIPE

        self._process = None

    def start(self):
        self._process = Popen(
            self._cmdln, stdin=self._stdin, stdout=self._stdout, stderr=self._stderr
        )

    def readout(self, cnt=None):
        if cnt is None:
            buf = self._process.stdout.read()
        else:
            buf = self._process.stdout.read(cnt)
        arr = np.frombuffer(buf, dtype=np.uint8)

        return arr

    def readerr(self, cnt):
        buf = self._process.stderr.read(cnt)
        return np.frombuffer(buf, dtype=np.uint8)

    def write(self, arr):
        bytes = arr.tobytes()
        self._process.stdin.write(bytes)

    def write_eof(self):
        if self._stdin != None:
            self._process.stdin.close()

    def is_running(self):
        return self._process.poll() is None

    @staticmethod
    def install(path, save_dir):
        from basicsr.utils.download_util import load_file_from_url
        from zipfile import ZipFile

        ffmpeg_url = "https://github.com/GyanD/codexffmpeg/releases/download/5.1.1/ffmpeg-5.1.1-full_build.zip"
        ffmpeg_dir = os.path.join(path, "ffmpeg")

        ckpt_path = load_file_from_url(url=ffmpeg_url, model_dir=ffmpeg_dir)

        if not os.path.exists(os.path.abspath(os.path.join(ffmpeg_dir, "ffmpeg.exe"))):
            with ZipFile(ckpt_path, "r") as zipObj:
                listOfFileNames = zipObj.namelist()
                for fileName in listOfFileNames:
                    if "/bin/" in fileName:
                        zipObj.extract(fileName, ffmpeg_dir)
            os.rename(
                os.path.join(ffmpeg_dir, listOfFileNames[0][:-1], "bin", "ffmpeg.exe"),
                os.path.join(ffmpeg_dir, "ffmpeg.exe"),
            )
            os.rename(
                os.path.join(ffmpeg_dir, listOfFileNames[0][:-1], "bin", "ffplay.exe"),
                os.path.join(ffmpeg_dir, "ffplay.exe"),
            )
            os.rename(
                os.path.join(ffmpeg_dir, listOfFileNames[0][:-1], "bin", "ffprobe.exe"),
                os.path.join(ffmpeg_dir, "ffprobe.exe"),
            )

            os.rmdir(os.path.join(ffmpeg_dir, listOfFileNames[0][:-1], "bin"))
            os.rmdir(os.path.join(ffmpeg_dir, listOfFileNames[0][:-1]))
        os.makedirs(save_dir, exist_ok=True)
        return

    @staticmethod
    def seconds(input="00:00:00"):
        [hours, minutes, seconds] = [int(pair) for pair in input.split(":")]
        return hours * 3600 + minutes * 60 + seconds
