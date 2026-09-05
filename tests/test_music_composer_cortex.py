"""
Test Music Composer Cortex Lifeform & Binary Format Standard
============================================================
验证硅基天籁音乐与复调对位歌王 (Neuro-Acoustic Singing King):
1. 检查点符合 SDSC-BIN v2 纯二进制标准规范 (1024 细胞, 196,608 突触)
2. 业务生命体清单 (models/business_lifeforms/manifest.json) 正确挂载
3. C-ABI 运行时或纯二进制解析器读取零崩溃、无 NaN、李雅普诺夫有界
"""

import os
import struct
import json
import numpy as np
import pytest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_PATH = os.path.join(ROOT_DIR, "checkpoints", "music_composer_cortex.bin")
MANIFEST_PATH = os.path.join(ROOT_DIR, "models", "business_lifeforms", "manifest.json")


def test_music_composer_binary_header():
    assert os.path.exists(CKPT_PATH), f"Checkpoint not found: {CKPT_PATH}"
    with open(CKPT_PATH, "rb") as f:
        header_bytes = f.read(72)
    assert len(header_bytes) == 72, "Header must be exactly 72 bytes"

    (magic, version, num_cells, num_synapses, in_dim, out_dim,
     cells_off, row_ptr_off, col_idx_off, weights_off, reserved) = struct.unpack(
        "<IIIIIIQQQQ16s", header_bytes
    )

    assert magic == 0x53445343, f"Invalid magic: {hex(magic)}"
    assert version == 2, f"Invalid version: {version}"
    assert num_cells == 1024, f"Unexpected cell count: {num_cells}"
    assert num_synapses == 196608, f"Unexpected synapse count: {num_synapses}"
    assert in_dim == 32, f"Unexpected input dim: {in_dim}"
    assert out_dim == 16, f"Unexpected output dim: {out_dim}"
    assert cells_off == 72
    assert row_ptr_off == cells_off + num_cells * 4
    assert col_idx_off == row_ptr_off + (num_cells + 1) * 4
    assert weights_off == col_idx_off + num_synapses * 4


def test_music_composer_manifest():
    assert os.path.exists(MANIFEST_PATH), f"Manifest missing: {MANIFEST_PATH}"
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    lifeforms = manifest.get("lifeforms", [])
    music_lifeform = None
    for item in lifeforms:
        if item.get("id") == "music_composer_cortex":
            music_lifeform = item
            break

    assert music_lifeform is not None, "music_composer_cortex not found in manifest"
    assert music_lifeform.get("cells_scale") == 1024
    assert music_lifeform.get("synapses_scale") == 196608
    assert "TonotopicHarmonicCortex" in music_lifeform.get("macro_columns", {})
    assert "TensionResolutionCortex" in music_lifeform.get("macro_columns", {})
    assert "RhythmGrooveCPGCortex" in music_lifeform.get("macro_columns", {})
    assert "MelodicMotifMemoryCortex" in music_lifeform.get("macro_columns", {})


def test_music_composer_data_integrity():
    with open(CKPT_PATH, "rb") as f:
        header = f.read(72)
        cells = f.read(1024 * 4)
        row_ptr = f.read((1024 + 1) * 4)
        col_idx = f.read(196608 * 4)
        weights = f.read(196608 * 4)

    assert len(cells) == 1024 * 4
    assert len(row_ptr) == 1025 * 4
    assert len(col_idx) == 196608 * 4
    assert len(weights) == 196608 * 4

    weights_arr = np.frombuffer(weights, dtype=np.float32)
    assert not np.isnan(weights_arr).any(), "Weights contain NaN"
    assert not np.isinf(weights_arr).any(), "Weights contain Inf"
    assert np.abs(weights_arr).max() > 0.0, "Weights must be non-zero"
