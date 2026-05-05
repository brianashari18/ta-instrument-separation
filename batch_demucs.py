#!/usr/bin/env python3
"""
batch_demucs.py — Batch Music Source Separation menggunakan Demucs (htdemucs_6s)

Menghasilkan beberapa versi per lagu:
  - Stem individual : vocals, guitar, piano, bass, drums, other
  - Campuran kustom : kombinasi 2+ stem (dikonfigurasi di STEM_MIXES)

Model: htdemucs_6s (6-stem)
  Stem yang tersedia: vocals, guitar, piano, bass, drums, other

Usage:
    python batch_demucs.py
    python batch_demucs.py --input_dir ./input --output_dir ./output
    python batch_demucs.py --device cpu --stems vocals guitar piano
"""

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ─── Logging Configuration ───────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Constants & Configuration ───────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".mp3", ".wav"}

# Model 6-stem: vocals, guitar, piano, bass, drums, other
DEMUCS_MODEL = "htdemucs_6s"

# Stem individual yang ingin dikumpulkan
# Ubah list ini sesuai kebutuhan. Opsi: vocals, guitar, piano, bass, drums, other
INDIVIDUAL_STEMS: List[str] = ["vocals", "guitar", "piano", "bass"]

# Campuran kustom: {nama_output: [stem1, stem2, ...]}
# Stem-stem ini akan dijumlah (mix) menjadi satu file audio
STEM_MIXES: Dict[str, List[str]] = {
    "guitar_piano":       ["guitar", "piano"],
    "guitar_piano_bass":  ["guitar", "piano", "bass"],
    "no_vocals":          ["guitar", "piano", "bass", "drums", "other"],
}


# ─── Audio Utilities ─────────────────────────────────────────────────────────

try:
    import soundfile as sf
    _HAS_SOUNDFILE = True
except ImportError:
    _HAS_SOUNDFILE = False
    logger.warning(
        "soundfile tidak terinstall. Mix stem tidak akan bisa dilakukan. "
        "Install dengan: pip install soundfile"
    )


def load_wav(path: Path) -> Tuple[np.ndarray, int]:
    """
    Membaca file WAV dan mengembalikan (audio_array, sample_rate).

    Args:
        path: Path ke file WAV.

    Returns:
        Tuple (numpy array float32, sample rate).

    Raises:
        ImportError: Jika soundfile tidak terinstall.
    """
    if not _HAS_SOUNDFILE:
        raise ImportError("soundfile diperlukan untuk operasi mix stem.")
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return data, sr


def save_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    """
    Menyimpan audio array ke file WAV.

    Args:
        path:        Tujuan penyimpanan file.
        audio:       Numpy array shape (samples, channels).
        sample_rate: Sample rate audio.
    """
    if not _HAS_SOUNDFILE:
        raise ImportError("soundfile diperlukan untuk menyimpan file WAV.")
    sf.write(str(path), audio, sample_rate)
    logger.debug("Disimpan: %s", path.name)


def mix_stems(stem_paths: List[Path]) -> Tuple[np.ndarray, int]:
    """
    Menjumlahkan beberapa stem WAV menjadi satu audio (mixing).

    Semua stem diasumsikan memiliki sample rate dan panjang yang sama
    (output dari Demucs selalu terpenuhi).

    Args:
        stem_paths: List Path ke file WAV yang akan di-mix.

    Returns:
        Tuple (mixed audio array, sample rate).

    Raises:
        ValueError: Jika list kosong atau panjang audio tidak sama.
    """
    if not stem_paths:
        raise ValueError("Tidak ada stem yang diberikan untuk di-mix.")

    mixed: Optional[np.ndarray] = None
    sr: int = 0

    for p in stem_paths:
        audio, sample_rate = load_wav(p)
        if mixed is None:
            mixed = audio.copy()
            sr = sample_rate
        else:
            if audio.shape != mixed.shape:
                raise ValueError(
                    f"Shape audio berbeda: {p.name} ({audio.shape}) "
                    f"vs stem sebelumnya ({mixed.shape})"
                )
            mixed += audio

    # Clip agar tidak ada clipping artefak
    mixed = np.clip(mixed, -1.0, 1.0)
    return mixed, sr


# ─── Core Pipeline Functions ──────────────────────────────────────────────────


def discover_audio_files(input_dir: Path) -> List[Path]:
    """
    Mencari semua file audio (.mp3 / .wav) di dalam folder input secara rekursif.

    Args:
        input_dir: Path ke folder yang berisi file audio.

    Returns:
        List berisi Path ke setiap file audio, terurut alfabetis.

    Raises:
        FileNotFoundError: Jika folder input tidak ditemukan.
        ValueError: Jika tidak ada file audio yang ditemukan.
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Folder input tidak ditemukan: {input_dir}")

    audio_files = []
    for ext in SUPPORTED_EXTENSIONS:
        audio_files.extend(input_dir.rglob(f"*{ext}"))
        audio_files.extend(input_dir.rglob(f"*{ext.upper()}"))
    
    audio_files = sorted(list(set(audio_files)))

    if not audio_files:
        raise ValueError(
            f"Tidak ada file audio ({', '.join(SUPPORTED_EXTENSIONS)}) "
            f"di dalam folder (atau subfolder): {input_dir}"
        )

    logger.info("Ditemukan %d file audio di '%s' (termasuk subfolder)", len(audio_files), input_dir)
    return audio_files


def run_demucs(
    audio_path: Path,
    demucs_temp_dir: Path,
    device: str = "cpu",
) -> Path:
    """
    Menjalankan Demucs CLI (htdemucs_6s) untuk satu file audio.

    Menghasilkan 6 stem: vocals, guitar, piano, bass, drums, other.

    Args:
        audio_path:      Path ke file audio sumber.
        demucs_temp_dir: Folder tempat Demucs menyimpan output sementara.
        device:          Device untuk inferensi ('cpu' atau 'cuda').

    Returns:
        Path ke folder stem hasil Demucs untuk track ini.

    Raises:
        subprocess.CalledProcessError: Jika proses Demucs gagal.
        FileNotFoundError: Jika folder hasil Demucs tidak ditemukan.
    """
    cmd = [
        sys.executable, "-m", "demucs",
        "-n", DEMUCS_MODEL,
        "-d", device,
        "-o", str(demucs_temp_dir),
        str(audio_path),
    ]

    logger.info("Menjalankan Demucs: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # Demucs menyimpan hasil di: <temp>/<model>/<track_name>/
    result_dir = demucs_temp_dir / DEMUCS_MODEL / audio_path.stem

    if not result_dir.exists():
        raise FileNotFoundError(
            f"Folder hasil Demucs tidak ditemukan: {result_dir}"
        )

    return result_dir


def collect_individual_stems(
    demucs_result_dir: Path,
    output_dir: Path,
    original_stem_name: str,
    stems_to_collect: List[str],
    artist_name: str = "Unknown",
) -> Dict[str, Path]:
    """
    Menyalin stem individual dari hasil Demucs ke subfolder output.

    Struktur output:
        output_dir/
          <stem>/
            <artist_name>/
              <nama>_<stem>.wav

    Args:
        demucs_result_dir:  Folder hasil Demucs (berisi *.wav per stem).
        output_dir:         Root folder tujuan output.
        original_stem_name: Nama file asli tanpa ekstensi.
        stems_to_collect:   List nama stem yang ingin dikumpulkan.
        artist_name:        Nama artist untuk subfolder.
    """
    collected: Dict[str, Path] = {}

    for stem in stems_to_collect:
        src = demucs_result_dir / f"{stem}.wav"

        if not src.exists():
            logger.warning("Stem '%s' tidak ditemukan: %s", stem, src)
            continue

        dest_dir = output_dir / stem / artist_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{original_stem_name}_{stem}.wav"

        shutil.copy2(str(src), str(dest))
        collected[stem] = src  # simpan path sumber untuk mixing
        logger.info("  [%s] → %s", stem, dest.relative_to(output_dir))

    return collected


def create_stem_mixes(
    demucs_result_dir: Path,
    output_dir: Path,
    original_stem_name: str,
    mixes: Dict[str, List[str]],
    artist_name: str = "Unknown",
) -> None:
    """
    Membuat file campuran stem dan menyimpannya ke subfolder output.

    Args:
        demucs_result_dir:  Folder hasil Demucs.
        output_dir:         Root folder output.
        original_stem_name: Nama file asli tanpa ekstensi.
        mixes:              Dict {nama_mix: [stem1, stem2, ...]}.
        artist_name:        Nama artist untuk subfolder.
    """
    if not _HAS_SOUNDFILE:
        logger.warning("Melewati pembuatan mix — soundfile tidak terinstall.")
        return

    for mix_name, stem_list in mixes.items():
        stem_paths = []
        missing = []

        for stem in stem_list:
            p = demucs_result_dir / f"{stem}.wav"
            if p.exists():
                stem_paths.append(p)
            else:
                missing.append(stem)

        if missing:
            logger.warning(
                "Mix '%s': stem tidak tersedia: %s — dilewati.",
                mix_name, missing,
            )
            continue

        try:
            mixed_audio, sr = mix_stems(stem_paths)
            mix_dir = output_dir / mix_name / artist_name
            mix_dir.mkdir(parents=True, exist_ok=True)
            dest = mix_dir / f"{original_stem_name}_{mix_name}.wav"
            save_wav(dest, mixed_audio, sr)
            logger.info("  [mix:%s] → %s", mix_name, dest.relative_to(output_dir))

        except Exception as e:
            logger.error("Gagal membuat mix '%s': %s", mix_name, e)


def cleanup_demucs_output(demucs_temp_dir: Path, track_name: str) -> None:
    """
    Menghapus folder sementara yang dibuat Demucs untuk satu track.

    Args:
        demucs_temp_dir: Folder base output Demucs (argumen -o).
        track_name:      Nama track (stem dari nama file asli).
    """
    track_dir = demucs_temp_dir / DEMUCS_MODEL / track_name

    if track_dir.exists():
        shutil.rmtree(track_dir)
        logger.debug("Dihapus (temp): %s", track_dir)

    model_dir = demucs_temp_dir / DEMUCS_MODEL
    if model_dir.exists() and not any(model_dir.iterdir()):
        model_dir.rmdir()


# ─── Main Pipeline ────────────────────────────────────────────────────────────


def process_batch(
    input_dir: Path,
    output_dir: Path,
    device: str = "cpu",
    individual_stems: List[str] = INDIVIDUAL_STEMS,
    stem_mixes: Dict[str, List[str]] = STEM_MIXES,
) -> None:
    """
    Pipeline utama batch separation.

    Per lagu:
      1. Jalankan Demucs 6-stem (htdemucs_6s)
      2. Salin stem individual ke subfolder output
      3. Buat file mix dari kombinasi stem
      4. Bersihkan folder temp

    Args:
        input_dir:        Folder berisi file audio sumber.
        output_dir:       Root folder tujuan output.
        device:           Device inferensi ('cpu' atau 'cuda').
        individual_stems: Stem individual yang dikumpulkan.
        stem_mixes:       Dict campuran kustom.
    """
    audio_files = discover_audio_files(input_dir)

    demucs_temp_dir = Path("_demucs_temp")
    demucs_temp_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    fail_count = 0
    failed_files: List[str] = []

    logger.info("=" * 65)
    logger.info("MEMULAI BATCH SEPARATION — %d lagu", len(audio_files))
    logger.info("Model         : %s", DEMUCS_MODEL)
    logger.info("Device        : %s", device)
    logger.info("Stem individual: %s", individual_stems)
    logger.info("Mix kustom    : %s", list(stem_mixes.keys()))
    logger.info("=" * 65)

    for idx, audio_path in enumerate(audio_files, start=1):
        logger.info(
            "─── [%d/%d] Memproses: %s ───",
            idx, len(audio_files), audio_path.name,
        )

        try:
            # Tentukan nama artist berdasarkan struktur folder atau nama file
            relative_path = audio_path.relative_to(input_dir)
            if len(relative_path.parts) > 1:
                # Jika di dalam subfolder, gunakan nama subfolder pertama sebagai artist
                artist_name = relative_path.parts[0]
            else:
                # Jika di root, coba ambil dari nama file: "artist-title"
                if "-" in audio_path.stem:
                    artist_name = audio_path.stem.split("-")[0].strip()
                else:
                    artist_name = "Unknown"

            # 1. Jalankan Demucs
            result_dir = run_demucs(audio_path, demucs_temp_dir, device)

            # 2. Kumpulkan stem individual
            collect_individual_stems(
                demucs_result_dir=result_dir,
                output_dir=output_dir,
                original_stem_name=audio_path.stem,
                stems_to_collect=individual_stems,
                artist_name=artist_name,
            )

            # 3. Buat campuran kustom
            create_stem_mixes(
                demucs_result_dir=result_dir,
                output_dir=output_dir,
                original_stem_name=audio_path.stem,
                mixes=stem_mixes,
                artist_name=artist_name,
            )

            # 4. Bersihkan folder temp
            cleanup_demucs_output(demucs_temp_dir, audio_path.stem)

            success_count += 1

        except FileNotFoundError as e:
            logger.error("File error pada '%s': %s", audio_path.name, e)
            fail_count += 1
            failed_files.append(audio_path.name)

        except subprocess.CalledProcessError as e:
            logger.error(
                "Demucs gagal untuk '%s' (return code: %d)",
                audio_path.name, e.returncode,
            )
            fail_count += 1
            failed_files.append(audio_path.name)

        except Exception as e:
            logger.error(
                "Error tidak terduga pada '%s': %s", audio_path.name, e,
            )
            fail_count += 1
            failed_files.append(audio_path.name)

    # — Bersihkan folder temp utama
    if demucs_temp_dir.exists():
        shutil.rmtree(demucs_temp_dir, ignore_errors=True)

    # ─── Summary ──────────────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("BATCH SEPARATION SELESAI")
    logger.info("  ✅ Berhasil : %d / %d", success_count, len(audio_files))
    logger.info("  ❌ Gagal    : %d / %d", fail_count, len(audio_files))

    if failed_files:
        logger.warning("File yang gagal diproses:")
        for name in failed_files:
            logger.warning("  • %s", name)

    logger.info("Output tersimpan di: %s", output_dir.resolve())
    logger.info("=" * 65)


# ─── Entry Point ──────────────────────────────────────────────────────────────

   
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments. """
    parser = argparse.ArgumentParser(
        description=(
            "Batch vocal/instrument separation menggunakan Demucs htdemucs_6s.\n"
            "Menghasilkan stem individual + campuran kustom per lagu."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("input_audio"),
        help="Folder berisi file audio sumber (default: input_audio/)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output_stems"),
        help="Root folder tujuan output (default: output_stems/)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device untuk inferensi (default: cpu)",
    )
    parser.add_argument(
        "--stems",
        type=str,
        nargs="+",
        default=INDIVIDUAL_STEMS,
        choices=["vocals", "guitar", "piano", "bass", "drums", "other"],
        help=(
            "Stem individual yang dikumpulkan "
            "(default: %(default)s)"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_batch(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        device=args.device,
        individual_stems=args.stems,
        stem_mixes=STEM_MIXES,
    )
