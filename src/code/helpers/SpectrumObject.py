"""Lightweight container and loaders for MALDI-TOF spectra."""

import base64
import binascii
import numpy as np
import pandas as pd
import os
import shutil
import subprocess
import tempfile

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
LOCAL_R_LIBS = os.path.join(WORKSPACE_ROOT, ".r_libs")
BRUKER_BASE64_FILES = {"acqu", "acqus", "fid", "1r", "proc", "procs", "sptype"}


def _decode_base64_payload(raw_bytes):
    compact = b"".join(raw_bytes.split())
    if not compact or len(compact) % 4:
        return None
    try:
        compact.decode("ascii")
        return base64.b64decode(compact, validate=True)
    except (UnicodeDecodeError, binascii.Error, ValueError):
        return None


def _decode_bruker_text_bytes(raw_bytes):
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = None
    if text and "##$" in text:
        return text

    decoded = _decode_base64_payload(raw_bytes)
    if decoded is None:
        return text
    try:
        decoded_text = decoded.decode("utf-8")
    except UnicodeDecodeError:
        return text
    if "##$" in decoded_text:
        return decoded_text
    return text


def _read_bruker_text_lines(path):
    with open(path, "rb") as handle:
        raw_bytes = handle.read()
    decoded_text = _decode_bruker_text_bytes(raw_bytes)
    if decoded_text is None:
        return []
    return [line.rstrip() for line in decoded_text.splitlines()]


def _read_bruker_intensity(path, byte_order, expected_count=None):
    dtype = np.dtype({0: "<i4", 1: ">i4"}[byte_order])
    with open(path, "rb") as handle:
        raw_bytes = handle.read()

    expected_size = expected_count * dtype.itemsize if expected_count is not None else None
    if expected_size is not None and len(raw_bytes) == expected_size:
        payload = raw_bytes
    else:
        decoded = _decode_base64_payload(raw_bytes)
        if decoded is not None and (
            expected_size is None or len(decoded) == expected_size
        ):
            payload = decoded
        else:
            payload = raw_bytes

    usable_size = len(payload) - (len(payload) % dtype.itemsize)
    return np.frombuffer(payload[:usable_size], dtype=dtype).copy()


def _prepare_bruker_dir_for_r(path):
    source_dir = os.fspath(path)
    temp_dir = tempfile.mkdtemp(prefix="bruker_r_")
    shutil.copytree(source_dir, temp_dir, dirs_exist_ok=True)
    for root, _, files in os.walk(temp_dir):
        for name in files:
            if name.lower() not in BRUKER_BASE64_FILES:
                continue
            file_path = os.path.join(root, name)
            with open(file_path, "rb") as handle:
                raw_bytes = handle.read()
            decoded = _decode_base64_payload(raw_bytes)
            if decoded is not None:
                with open(file_path, "wb") as handle:
                    handle.write(decoded)
    return temp_dir

def _find_case_insensitive_file(folder, name):
    direct = os.path.join(folder, name)
    if os.path.isfile(direct):
        return direct
    lowered = name.lower()
    for entry in os.listdir(folder):
        candidate = os.path.join(folder, entry)
        if os.path.isfile(candidate) and entry.lower() == lowered:
            return candidate
    return None


def _find_bruker_acquisition_file(folder):
    return _find_case_insensitive_file(folder, "acqu") or _find_case_insensitive_file(folder, "acqus")


def _find_case_insensitive_file_in_ancestors(folder, name, max_levels=4):
    current = folder
    for _ in range(max_levels + 1):
        found = _find_case_insensitive_file(current, name)
        if found:
            return found
        parent = os.path.dirname(current)
        if not parent or parent == current:
            break
        current = parent
    return None


def _find_bruker_acquisition_file_in_ancestors(folder, max_levels=4):
    current = folder
    for _ in range(max_levels + 1):
        found = _find_bruker_acquisition_file(current)
        if found:
            return found
        parent = os.path.dirname(current)
        if not parent or parent == current:
            break
        current = parent
    return None


def _load_bruker_with_maldiquant(path):
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise RuntimeError("Rscript is not available for MALDIquant fallback.")

    with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False) as handle:
        output_path = handle.name
    prepared_dir = _prepare_bruker_dir_for_r(path)
    try:
        script = """
args <- commandArgs(trailingOnly=TRUE)
input_path <- args[[1]]
output_path <- args[[2]]
local_lib <- args[[3]]
if (nzchar(local_lib) && dir.exists(local_lib)) {
  .libPaths(c(local_lib, .libPaths()))
}
if (requireNamespace("readBrukerFlexData", quietly=TRUE)) {
  fid_path <- file.path(input_path, "fid")
  if (!file.exists(fid_path)) {
    alt <- list.files(input_path, pattern="^fid$", ignore.case=TRUE, full.names=TRUE)
    if (length(alt) > 0) {
      fid_path <- alt[[1]]
    }
  }
  spect <- readBrukerFlexData::readBrukerFlexFile(fid_path, removeMetaData=TRUE, verbose=FALSE)
  utils::write.table(
    data.frame(
      mz = spect$spectrum$mass,
      intensity = spect$spectrum$intensity
    ),
    file = output_path,
    sep = "\\t",
    row.names = FALSE,
    col.names = FALSE,
    quote = FALSE
  )
} else {
  if (!requireNamespace("MALDIquant", quietly=TRUE) ||
      !requireNamespace("MALDIquantForeign", quietly=TRUE)) {
    stop("readBrukerFlexData or MALDIquant with MALDIquantForeign must be installed.")
  }
  spectra <- MALDIquantForeign::importBrukerFlex(input_path, verbose=FALSE)
  if (length(spectra) < 1) {
    stop("No spectra could be imported by MALDIquantForeign.")
  }
  spect <- spectra[[1]]
  utils::write.table(
    data.frame(
      mz = MALDIquant::mass(spect),
      intensity = MALDIquant::intensity(spect)
    ),
    file = output_path,
    sep = "\\t",
    row.names = FALSE,
    col.names = FALSE,
    quote = FALSE
  )
}
"""
        result = subprocess.run(
            [rscript, "-e", script, prepared_dir, output_path, LOCAL_R_LIBS],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            stderr = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(stderr or "Unknown MALDIquant fallback error.")

        data = pd.read_table(output_path, sep="\t", header=None).values
        if data.size == 0:
            raise RuntimeError("MALDIquant fallback returned an empty spectrum.")
        return data[:, 0].astype(float), data[:, 1].astype(float)
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)
        if os.path.exists(prepared_dir):
            shutil.rmtree(prepared_dir, ignore_errors=True)


class SpectrumObject:
    """Store paired m/z and intensity arrays for one spectrum."""

    def __init__(self, mz=None, intensity=None):
        """Create a spectrum and normalize unsigned arrays to signed integer types."""
        self.mz = mz
        self.intensity = intensity
        if self.intensity is not None:
            if np.issubdtype(self.intensity.dtype, np.unsignedinteger):
                self.intensity = self.intensity.astype(int)
        if self.mz is not None:
            if np.issubdtype(self.mz.dtype, np.unsignedinteger):
                self.mz = self.mz.astype(int)

    def __getitem__(self, index):
        """Return a sliced SpectrumObject."""
        return SpectrumObject(mz=self.mz[index], intensity=self.intensity[index])

    def __len__(self):
        """Return the number of m/z points."""
        if self.mz is not None:
            return self.mz.shape[0]
        return 0

    def plot(self, as_peaks=False, **kwargs):
        """Plot the spectrum either as a line or as peak sticks."""
        import matplotlib.pyplot as plt

        if as_peaks:
            mz_plot = np.stack([self.mz - 1, self.mz, self.mz + 1]).T.reshape(-1)
            int_plot = np.stack(
                [
                    np.zeros_like(self.intensity),
                    self.intensity,
                    np.zeros_like(self.intensity),
                ]
            ).T.reshape(-1)
        else:
            mz_plot, int_plot = self.mz, self.intensity
        plt.plot(mz_plot, int_plot, **kwargs)

    def __repr__(self):
        """Return a compact representation of m/z and intensity arrays."""
        string_ = np.array2string(
            np.stack([self.mz, self.intensity]), precision=5, threshold=10, edgeitems=2
        )
        mz_string, int_string = string_.split("\n")
        mz_string = mz_string[1:]
        int_string = int_string[1:-1]
        return "SpectrumObject([\n\tmz  = %s,\n\tint = %s\n])" % (mz_string, int_string)

    @staticmethod
    def tof2mass(ML1, ML2, ML3, TOF):
        """Convert time-of-flight values to masses using Bruker calibration constants."""
        A = ML3
        B = np.sqrt(1e12 / ML1)
        C = ML2 - TOF
        if A == 0:
            return (C * C) / (B * B)
        else:
            return ((-B + np.sqrt((B * B) - (4 * A * C))) / (2 * A)) ** 2

    @classmethod
    def from_bruker(cls, acqu_file, fid_file):
        """Load a Bruker acquisition from `acqu` metadata and `fid` intensities."""
        lines = _read_bruker_text_lines(acqu_file)
        TD = None
        DELAY = None
        DW = None
        ML1 = None
        ML2 = None
        ML3 = None
        BYTORDA = 0
        for l in lines:
            if l.startswith("##$TD"):
                TD = int(l.split("= ")[1])
            if l.startswith("##$DELAY"):
                DELAY = int(l.split("= ")[1])
            if l.startswith("##$DW"):
                DW = float(l.split("= ")[1])
            if l.startswith("##$ML1"):
                ML1 = float(l.split("= ")[1])
            if l.startswith("##$ML2"):
                ML2 = float(l.split("= ")[1])
            if l.startswith("##$ML3"):
                ML3 = float(l.split("= ")[1])
            if l.startswith("##$BYTORDA"):
                BYTORDA = int(l.split("= ")[1])
            if l.startswith("##$NTBCal"):
                NTBCal = l.split("= ")[1]

        intensity = _read_bruker_intensity(fid_file, BYTORDA, expected_count=TD)
        if TD is None:
            TD = len(intensity)
        elif len(intensity) < TD:
            TD = len(intensity)
        missing = [
            name
            for name, value in {
                "DELAY": DELAY,
                "DW": DW,
                "ML1": ML1,
                "ML2": ML2,
                "ML3": ML3,
            }.items()
            if value is None
        ]
        if missing:
            try:
                mz, intensity = _load_bruker_with_maldiquant(os.path.dirname(fid_file))
                intensity[intensity < 0] = 0
                return cls(mz=mz, intensity=intensity)
            except Exception as exc:
                raise ValueError(
                    f"Missing required Bruker acquisition field(s) in {acqu_file}: {', '.join(missing)}. "
                    f"MALDIquant fallback failed: {exc}"
                ) from exc
        intensity = intensity[:TD]
        TOF = DELAY + np.arange(TD) * DW
        mass = cls.tof2mass(ML1, ML2, ML3, TOF)
        intensity[intensity < 0] = 0
        return cls(mz=mass, intensity=intensity)

    @classmethod
    def from_tsv(cls, file, sep=" "):
        """Load a two-column text spectrum containing m/z and intensity values."""
        s = pd.read_table(file, sep=sep, index_col=None, comment="#", header=None).values
        if not str(s[0, 0]).replace(".", "", 1).isdigit():
            s = s[1:]
        mz = s[:, 0].astype(float)
        intensity = s[:, 1].astype(float)
        return cls(mz=mz, intensity=intensity)

    @classmethod
    def from_mzxml(cls, file_path: str):
        """Read mzXML/mzML using pyteomics (no index needed)."""
        if file_path.lower().endswith(".mzml"):
            import pyteomics.mzml as mz_reader
            reader_cls = mz_reader.MzML
        else:
            import pyteomics.mzxml as mz_reader
            reader_cls = mz_reader.MzXML

        try:
            with reader_cls(file_path) as reader:
                for spectrum in reader:
                    ms_level = spectrum.get("msLevel", spectrum.get("ms level"))
                    if ms_level in (None, 1, "1"):
                        mz = np.array(spectrum['m/z array'], dtype=np.float32)
                        intensity = np.array(spectrum['intensity array'], dtype=np.float32)
                        if len(mz) == 0:
                            continue
                        return cls(mz=mz, intensity=intensity)
            raise ValueError("No valid MS1 spectrum found")
        except Exception as e:
            raise IOError(f"Error reading {file_path}: {e}")

    @classmethod
    def load(cls, path: str):
        """Universal loader for Bruker folders, TSV files, and mzXML/mzML."""
        if path.lower().endswith(('.mzxml', '.mzml')):
            return cls.from_mzxml(path)

        if os.path.isdir(path):
            acqu_file = _find_bruker_acquisition_file_in_ancestors(path)
            fid_file = _find_case_insensitive_file(path, 'fid')
            if acqu_file and fid_file:
                return cls.from_bruker(acqu_file, fid_file)
        elif os.path.isfile(path) and os.path.basename(path).lower() in {'acqu', 'acqus', 'fid'}:
            folder = os.path.dirname(path)
            acqu_file = _find_bruker_acquisition_file_in_ancestors(folder)
            fid_file = _find_case_insensitive_file(folder, 'fid')
            if acqu_file and fid_file:
                return cls.from_bruker(acqu_file, fid_file)

        return cls.from_tsv(path)
