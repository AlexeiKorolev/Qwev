"""Deterministic synthetic scenes, not benchmark accuracy ground truth.

The minimal PNG writer keeps fixtures and --dry-run usable without installing
PyTorch or Pillow. These are original diagrams, not BabyAI/GUI benchmark data.
"""

import hashlib
import json
import struct
import zlib
from pathlib import Path


FONT = {
    "A": "01110 10001 10001 11111 10001 10001 10001", "B": "11110 10001 10001 11110 10001 10001 11110",
    "C": "01111 10000 10000 10000 10000 10000 01111", "D": "11110 10001 10001 10001 10001 10001 11110",
    "E": "11111 10000 10000 11110 10000 10000 11111", "F": "11111 10000 10000 11110 10000 10000 10000",
    "G": "01111 10000 10000 10111 10001 10001 01111", "H": "10001 10001 10001 11111 10001 10001 10001",
    "I": "11111 00100 00100 00100 00100 00100 11111", "J": "00111 00010 00010 00010 10010 10010 01100",
    "K": "10001 10010 10100 11000 10100 10010 10001", "L": "10000 10000 10000 10000 10000 10000 11111",
    "M": "10001 11011 10101 10101 10001 10001 10001", "N": "10001 11001 10101 10011 10001 10001 10001",
    "O": "01110 10001 10001 10001 10001 10001 01110", "P": "11110 10001 10001 11110 10000 10000 10000",
    "Q": "01110 10001 10001 10001 10101 10010 01101", "R": "11110 10001 10001 11110 10100 10010 10001",
    "S": "01111 10000 10000 01110 00001 00001 11110", "T": "11111 00100 00100 00100 00100 00100 00100",
    "U": "10001 10001 10001 10001 10001 10001 01110", "V": "10001 10001 10001 10001 10001 01010 00100",
    "W": "10001 10001 10001 10101 10101 10101 01010", "X": "10001 10001 01010 00100 01010 10001 10001",
    "Y": "10001 10001 01010 00100 00100 00100 00100", "Z": "11111 00001 00010 00100 01000 10000 11111",
    "0": "01110 10001 10011 10101 11001 10001 01110", "1": "00100 01100 00100 00100 00100 00100 01110",
    "2": "01110 10001 00001 00010 00100 01000 11111", "3": "11110 00001 00001 01110 00001 00001 11110",
    "4": "00010 00110 01010 10010 11111 00010 00010", "5": "11111 10000 10000 11110 00001 00001 11110",
    "6": "01110 10000 10000 11110 10001 10001 01110", "7": "11111 00001 00010 00100 01000 01000 01000",
    "8": "01110 10001 10001 01110 10001 10001 01110", "9": "01110 10001 10001 01111 00001 00001 01110",
    ":": "00000 00100 00100 00000 00100 00100 00000", "-": "00000 00000 00000 11111 00000 00000 00000",
    ".": "00000 00000 00000 00000 00000 00100 00100", "/": "00001 00001 00010 00100 01000 10000 10000",
}


class Canvas:
    def __init__(self, width, height, color):
        self.width, self.height = width, height
        self.pixels = bytearray(bytes(color) * width * height)

    def rect(self, x, y, width, height, color):
        left, right = max(0, x), min(self.width, x + width)
        if right <= left:
            return
        row = bytes(color) * (right - left)
        for yy in range(max(0, y), min(self.height, y + height)):
            start = (yy * self.width + left) * 3
            self.pixels[start:start + len(row)] = row

    def circle(self, x, y, radius, color):
        for dy in range(-radius, radius + 1):
            half = int((radius * radius - dy * dy) ** 0.5)
            self.rect(x - half, y + dy, 2 * half + 1, 1, color)

    def text(self, x, y, text, color=(35, 45, 65), scale=3):
        for char in text.upper():
            for row, pattern in enumerate(FONT.get(char, "").split()):
                for col, bit in enumerate(pattern):
                    if bit == "1":
                        self.rect(x + col * scale, y + row * scale, scale, scale, color)
            x += 6 * scale

    def save(self, path):
        def chunk(kind, data):
            return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
        rows = b"".join(b"\x00" + self.pixels[y*self.width*3:(y+1)*self.width*3] for y in range(self.height))
        payload = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0))
        payload += chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")
        path.write_bytes(payload)


def generate_fixtures(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    grid = Canvas(1024, 1024, (245, 247, 250))
    grid.text(44, 25, "QWEV SYNTHETIC GRID", scale=4)
    grid.text(44, 72, "GOAL: PICK UP THE YELLOW KEY", scale=3)
    for row in range(8):
        for col in range(8):
            color = (64, 73, 86) if row in {0, 7} or col in {0, 7} else (228, 233, 240)
            grid.rect(104 + col*102, 125 + row*102, 99, 99, color)
    # Agent at (2,4), facing right. Key at (4,4); forward is valid.
    for dx in range(56):
        half = (56-dx)//2
        grid.rect(327+dx, 586-half, 1, 2*half+1, (42, 105, 205))
    grid.circle(548, 582, 19, (233, 184, 28))
    grid.circle(548, 582, 9, (228, 233, 240))
    grid.rect(558, 576, 35, 12, (233, 184, 28))
    grid.rect(581, 584, 9, 13, (233, 184, 28))
    grid.circle(656, 378, 25, (204, 55, 68))
    grid.rect(720, 641, 65, 70, (52, 152, 97))
    grid.circle(771, 676, 5, (248, 235, 139))
    grid.text(44, 960, "INVENTORY: EMPTY", scale=4)
    grid_path = output / "grid.png"
    grid.save(grid_path)
    gui = Canvas(1280, 800, (242, 244, 248))
    gui.rect(0, 0, 1280, 64, (30, 43, 64))
    gui.text(28, 20, "QWEV SETTINGS - SYNTHETIC UI", (255, 255, 255), 3)
    gui.rect(0, 64, 260, 736, (228, 234, 241))
    for i, name in enumerate(["GENERAL", "DISPLAY", "AUDIO", "CONTROLS"]):
        if i == 1:
            gui.rect(16, 164, 228, 57, (205, 221, 247))
        gui.text(34, 110 + i*72, name, scale=3)
    gui.text(310, 104, "DISPLAY SETTINGS", scale=5)
    for i, (label, value) in enumerate([("RESOLUTION", "1920 X 1080"), ("QUALITY", "HIGH"), ("FULLSCREEN", "ON")]):
        y = 215 + 125*i
        gui.text(320, y, label, scale=3)
        gui.rect(770, y-18, 390, 65, (255, 255, 255))
        gui.text(793, y, value, scale=3)
    gui.rect(910, 660, 252, 76, (40, 104, 204))
    gui.text(956, 685, "APPLY", (255, 255, 255), 4)
    gui.text(330, 688, "UNSAVED CHANGES", scale=3)
    gui_path = output / "gui.png"
    gui.save(gui_path)
    manifest = {
        "description": "Original synthetic latency fixtures; not real BabyAI or GUI benchmark observations.",
        "images": [{"path": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in [grid_path, gui_path]],
        "manual_checks": [
            {"image": "grid.png", "question": "What color is the key?", "answer": "yellow"},
            {"image": "grid.png", "question": "Which way is the blue agent facing?", "answer": "right"},
            {"image": "gui.png", "question": "What is the resolution setting?", "answer": "1920 x 1080"},
            {"image": "gui.png", "question": "Which button saves the changes?", "answer": "Apply"},
        ],
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return [grid_path.resolve(), gui_path.resolve(), manifest_path.resolve()]
