import os
import struct
from pathlib import Path

import numpy as np
from PIL import Image

import torch


class PNGReader():
    def __init__(self, filepath):
        self.filepath = filepath
        self.eof = False

    def read_one_frame(self, src_format="rgb"):
        if self.eof:
            return None

        png_path = self.filepath
        if not os.path.exists(png_path):
            self.eof = True
            return None

        rgb = Image.open(png_path).convert('RGB')
        rgb = np.asarray(rgb).astype('float32').transpose(2, 0, 1)
        rgb = rgb / 255.
        return rgb


def get_padding_size(height, width, p=64):
    new_h = (height + p - 1) // p * p
    new_w = (width + p - 1) // p * p
    # padding_left = (new_w - width) // 2
    padding_left = 0
    padding_right = new_w - width - padding_left
    # padding_top = (new_h - height) // 2
    padding_top = 0
    padding_bottom = new_h - height - padding_top
    return padding_left, padding_right, padding_top, padding_bottom


def get_downsampled_shape(height, width, p):
    new_h = (height + p - 1) // p * p
    new_w = (width + p - 1) // p * p
    return int(new_h / p + 0.5), int(new_w / p + 0.5)


def get_rounded_q(q_scale):
    q_scale = np.clip(q_scale, 0.01, 655.)
    q_index = int(np.round(q_scale * 100))
    q_scale = q_index / 100
    return q_scale, q_index


def consume_prefix_in_state_dict_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    for key in keys:
        if key.startswith(prefix):
            newkey = key[len(prefix):]
            state_dict[newkey] = state_dict.pop(key)

    # also strip the prefix in metadata if any.
    if "_metadata" in state_dict:
        metadata = state_dict["_metadata"]
        for key in list(metadata.keys()):
            if len(key) == 0:
                continue
            newkey = key[len(prefix):]
            metadata[newkey] = metadata.pop(key)


def get_state_dict(ckpt_path):
    def get_one_state_dict(path):
        ckpt = torch.load(path, map_location=torch.device('cpu'))
        if "state_dict" in ckpt:
            ckpt = ckpt['state_dict']
        if "net" in ckpt:
            ckpt = ckpt["net"]
        consume_prefix_in_state_dict_if_present(ckpt, prefix="module.")
        return ckpt

    if isinstance(ckpt_path, list):
        state_dict = [get_one_state_dict(path) for path in ckpt_path]
    else:
        state_dict = get_one_state_dict(ckpt_path)
    return state_dict


def filesize(filepath: str) -> int:
    if not Path(filepath).is_file():
        raise ValueError(f'Invalid file "{filepath}".')
    return Path(filepath).stat().st_size


def write_uints(fd, values, fmt=">{:d}I"):
    fd.write(struct.pack(fmt.format(len(values)), *values))


def write_uchars(fd, values, fmt=">{:d}B"):
    fd.write(struct.pack(fmt.format(len(values)), *values))


def read_uints(fd, n, fmt=">{:d}I"):
    sz = struct.calcsize("I")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def read_uchars(fd, n, fmt=">{:d}B"):
    sz = struct.calcsize("B")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def write_bytes(fd, values, fmt=">{:d}s"):
    if len(values) == 0:
        return
    fd.write(struct.pack(fmt.format(len(values)), values))


def read_bytes(fd, n, fmt=">{:d}s"):
    sz = struct.calcsize("s")
    return struct.unpack(fmt.format(n), fd.read(n * sz))[0]


def write_ushorts(fd, values, fmt=">{:d}H"):
    fd.write(struct.pack(fmt.format(len(values)), *values))


def read_ushorts(fd, n, fmt=">{:d}H"):
    sz = struct.calcsize("H")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def encode_i(height, width, q_index, bit_stream, output):
    with Path(output).open("wb") as f:
        stream_length = len(bit_stream)

        write_uints(f, (height, width))
        write_ushorts(f, (q_index,))
        write_uints(f, (stream_length,))
        write_bytes(f, bit_stream)


def decode_i(inputpath):
    with Path(inputpath).open("rb") as f:
        header = read_uints(f, 2)
        height = header[0]
        width = header[1]
        q_index = read_ushorts(f, 1)[0]
        stream_length = read_uints(f, 1)[0]
        bit_stream = read_bytes(f, stream_length)

    return height, width, q_index, bit_stream


if __name__ == "__main__":

    print("end")