# Author: Jelly Choi
# This code is based on https://github.com/Filarius/stable-diffusion-webui/blob/master/scripts/vid2vid.py

import os
import subprocess
import sys

import random
from typing import TypedDict

import modules.scripts as scripts
import gradio as gr

from modules.processing import Processed, process_images
from PIL import Image

from modules.shared import state
from modules import processing, paths

from subprocess import Popen, PIPE, TimeoutExpired
import numpy as np
from datetime import datetime
import time


class Video2VideoParam:
    def __init__(self):
        self.input_path = ''
        self.crf = 24
        self.fps = 24
        self.seed_walk = 1
        self.seed_max_distance = 1
        self.start_time = ''
        self.end_time = ''
        self.use_timestamp = False
        self.original_resolution = False
        self.is_grayscale = False
        self.is_high_contrast = False
        self.initial_seed = 0
        self.initial_info = None
        self.time_interval = '0'
        self.save_dir = ''
        self.filter = 'none'
        self.input_file = ''
        self.output_file = ''
        self.interpolate = False
        self.interpolate_strength = 0.5
        self.blend_frame = 0.5


def init_seed(p, v2v: Video2VideoParam):
    processing.fix_seed(p)
    # p.subseed_strength == 0
    v2v.input_path = v2v.input_path.replace('"', "")
    v2v.initial_seed = p.seed
    v2v.initial_info = None


def init_i2i_params(p):
    p.do_not_save_grid = True
    p.do_not_save_samples = True
    p.batch_count = 1


def init_time_interval(v2v: Video2VideoParam):
    v2v.start_time = v2v.start_time.strip()
    v2v.end_time = v2v.end_time.strip()
    if v2v.start_time == "":
        v2v.start_time = "00:00:00"
    if v2v.end_time == "00:00:00":
        v2v.end_time = ""

    v2v.time_interval = (
        f"-ss {v2v.start_time}" + f" -to {v2v.end_time}" if len(v2v.end_time) else ""
    )


def init_resolution(v2v: Video2VideoParam):
    if v2v.original_resolution:
        ffmpeg(f"ffmpeg/ffprobe.exe -show_streams -select_streams v {v2v.input_file}")
        width, height = get_width_height(v2v.input_file)
        v2v.width = width
        v2v.height = height


def init_paths(v2v: Video2VideoParam):
    path = paths.script_path
    v2v.save_dir = "outputs/vid2vid-jelly"
    ffmpeg.install(path, v2v.save_dir)

    v2v.input_path = v2v.input_path.replace('"', "")
    v2v.input_file = os.path.normpath(v2v.input_path.strip())


def init_filters(v2v):
    filters = []
    if v2v.is_high_contrast:
        filters.append("curves=preset=strong_contrast")
    if v2v.is_grayscale:
        filters.append("format=gray")
    v2v.filter = '-vf "' + ",".join(filters) + '"' if v2v.is_high_contrast or v2v.is_grayscale else ""


def init_params(p, v2v: Video2VideoParam):
    init_paths(v2v)
    init_seed(p, v2v)
    init_time_interval(v2v)
    init_resolution(v2v)
    init_filters(v2v)


class Script(scripts.Script):
    def title(self):
        return '[Jelly] Video to video'

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
                minimum=-20, maximum=20, step=1, label="Seed step size", value=1
            )
            seed_max_distance = gr.Slider(
                minimum=1, maximum=200, step=1, label="Seed max distance", value=1
            )

        with gr.Row():
            start_time = gr.Textbox(label="Start time", value="00:00:00", lines=1)
            end_time = gr.Textbox(label="End time", value="00:00:00", lines=1)

        with gr.Row():
            use_timestamp = gr.Checkbox(label="Use Timestamp (Filename)", value=True)
            original_resolution = gr.Checkbox(label="Keep Original Resolution", value=True)
            is_grayscale = gr.Checkbox(label="Grayscale Input", value=False)
            is_high_contrast = gr.Checkbox(label="High Contrast Input", value=False)
            interpolate = gr.Checkbox(label="Interpolate", value=False, interactive=True)

        with gr.Row(visible=False) as interpolate_row:
            interpolate_strength = gr.Slider(label="Interpolate Denoise Strength", minimum=0.01, maximum=1, step=0.01,
                                             value=0.5)
            blend_frame = gr.Slider(label="Blend Before Frame", minimum=0.01, maximum=1, step=0.01, value=0.5)

        interpolate.change(fn=lambda value: interpolate_row.update(value), inputs=interpolate, outputs=interpolate_row)

        return [
            input_path,
            crf,
            fps,
            seed_walk,
            seed_max_distance,
            start_time,
            end_time,
            use_timestamp,
            original_resolution,
            is_grayscale,
            is_high_contrast,
            interpolate,
            interpolate_strength,
            blend_frame
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
            original_resolution,
            is_grayscale,
            is_high_contrast,
            interpolate,
            interpolate_strength,
            blend_frame
    ):
        v2v = Video2VideoParam()
        v2v.input_path = input_path
        v2v.crf = crf
        v2v.fps = fps
        v2v.seed_walk = seed_walk
        v2v.seed_max_distance = seed_max_distance
        v2v.start_time = start_time
        v2v.end_time = end_time
        v2v.use_timestamp = use_timestamp
        v2v.original_resolution = original_resolution
        v2v.is_grayscale = is_grayscale
        v2v.is_high_contrast = is_high_contrast
        v2v.interpolate_strength = interpolate_strength
        v2v.blend_frame = blend_frame

        init_params(p, v2v)

        decoder = ffmpeg(self.decode_command(p, v2v), use_stdout=True)
        decoder.start()

        v2v.output_file = v2v.input_file.split("\\")[-1]

        encoder = ffmpeg(self.encode_command(p, v2v), use_stdin=True)
        encoder.start()

        batch = []
        seed = v2v.initial_seed

        frame = 1
        if len(v2v.end_time) > 0:
            seconds = ffmpeg.seconds(v2v.end_time) - ffmpeg.seconds(v2v.start_time)
            loops = seconds * int(v2v.fps)
            state.job_count = loops
        else:
            loops = None

        pull_count = p.width * p.height * 3
        raw_image = decoder.readout(pull_count)

        while raw_image is not None and len(raw_image) > 0:
            if state.interrupted or state.skipped:
                decoder.process.kill()
                encoder.process.kill()
                return Processed(p, [], p.seed, v2v.initial_info)

            image_pil = Image.fromarray(
                np.uint8(raw_image).reshape((p.height, p.width, 3)), mode="RGB"
            )
            if v2v.interpolate:
                if len(batch) > 0:
                    p.denoising_strength = v2v.interpolate_strength
                    image_pil = Image.blend(image_pil, batch[-1], v2v.blend_frame)

            batch.append(image_pil)

            if len(batch) == p.batch_size:
                if v2v.seed_walk != 0:
                    seed_step = (
                        random.randint(0, v2v.seed_walk)
                        if v2v.seed_walk >= 0
                        else random.randint(v2v.seed_walk, 0)
                    )
                    if (
                            seed_step != 0
                            or abs(seed + seed_step - v2v.initial_seed) <= v2v.seed_max_distance
                    ):
                        seed = seed + seed_step
                else:
                    seed_step = 1
                p.seed = [seed for _ in batch]
                p.init_images = batch
                batch = []

                state.job = f"{frame}/{int(loops)}|{seed}/{seed_step}"
                proc = process_images(p)
                if v2v.initial_info is None:
                    v2v.initial_info = proc.info

                for output in proc.images:
                    encoder.write(np.asarray(output))

            raw_image = decoder.readout(pull_count)
            frame += 1


        decoder.safe_exit()
        encoder.safe_exit()

        if v2v.use_timestamp:
            dt = datetime.now()
            time_stamp = dt.strftime("%Y%m%d_%H%M%S")
        else:
            time_stamp = "[Audio]"

        command = " ".join(
            [
                f'ffmpeg/ffmpeg.exe -y {v2v.time_interval} -i "{v2v.input_file}"',
                f'-i "{v2v.save_dir}/{v2v.output_file}" -movflags faststart',
                f'-map 0:a? -map 1:v "{v2v.save_dir}/{time_stamp}-{v2v.output_file}"',
            ]
        )
        audio_mix = ffmpeg(command)
        audio_mix.start()
        audio_mix.safe_exit()

        return Processed(p, [], p.seed, v2v.initial_info)

    def encode_command(self, p, v2v):
        return " ".join(
            [
                "ffmpeg/ffmpeg -y -loglevel panic",
                "-f rawvideo -pix_fmt rgb24",
                f"-s:v {p.width}x{p.height} -r {v2v.fps}",
                "-i - -c:v libx264 -preset fast -movflags faststart",
                f'-crf {v2v.crf} "{v2v.save_dir}/{v2v.output_file}"',
            ]
        )

    def decode_command(self, p, v2v):
        return " ".join(
            [
                "ffmpeg/ffmpeg -y -loglevel panic",
                f'{v2v.time_interval} -i "{v2v.input_file}"',
                f"-s:v {p.width}x{p.height} -r {v2v.fps} {v2v.filter}",
                "-f image2pipe -pix_fmt rgb24",
                "-vcodec rawvideo -",
            ]
        )


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
        self.process = None
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

        self.process = None

    def start(self):
        self.process = Popen(
            self._cmdln, stdin=self._stdin, stdout=self._stdout, stderr=self._stderr
        )

    def readout(self, cnt=None):
        if cnt is None:
            buf = self.process.stdout.read()
        else:
            buf = self.process.stdout.read(cnt)
        arr = np.frombuffer(buf, dtype=np.uint8)

        return arr

    def readerr(self, cnt):
        buf = self.process.stderr.read(cnt)
        return np.frombuffer(buf, dtype=np.uint8)

    def write(self, arr):
        bytes = arr.tobytes()
        self.process.stdin.write(bytes)

    def write_eof(self):
        if self._stdin != None:
            self.process.stdin.close()

    def is_running(self):
        return self.process.poll() is None

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

    def safe_exit(self):
        try:
            self.process.communicate(timeout=15)
        except TimeoutExpired:
            self.process.kill()
            self.process.communicate()

    @staticmethod
    def seconds(input="00:00:00"):
        [hours, minutes, seconds] = [int(pair) for pair in input.split(":")]
        return hours * 3600 + minutes * 60 + seconds
