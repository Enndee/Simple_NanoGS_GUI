from __future__ import annotations

import csv
import contextlib
import itertools
import json
import queue
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass, replace
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from simplification import simplify
from utils.params import CostParams, RunParams
from utils.sog_utils import convert_sog_to_ply


SUPPORTED_EXTENSIONS = {".ply", ".sog", ".spz"}
SETTINGS_PATH = Path(__file__).with_name("nanogs_gui_settings.json")
DEFAULT_GSBOX_PATH = Path(__file__).with_name("gsbox.exe")

DEFAULT_SETTINGS = {
    "source_mode": "file",
    "source_path": "",
    "output_dir": "",
    "recurse": False,
    "output_suffix": "_simplified",
    "ratio": "0.3",
    "k": "8",
    "opacity_threshold": "0.18",
    "lam_geo": "1.0",
    "lam_sh": "0.5",
    "device": "auto",
    "block_edges": "0",
    "quality_test_mode": False,
    "gsbox_path": str(DEFAULT_GSBOX_PATH),
    "spz_converter": "",
}

DEFAULT_PROFILE_SETTINGS = {
    "output_suffix": DEFAULT_SETTINGS["output_suffix"],
    "ratio": DEFAULT_SETTINGS["ratio"],
    "k": DEFAULT_SETTINGS["k"],
    "opacity_threshold": DEFAULT_SETTINGS["opacity_threshold"],
    "lam_geo": DEFAULT_SETTINGS["lam_geo"],
    "lam_sh": DEFAULT_SETTINGS["lam_sh"],
    "device": DEFAULT_SETTINGS["device"],
    "block_edges": DEFAULT_SETTINGS["block_edges"],
    "quality_test_mode": DEFAULT_SETTINGS["quality_test_mode"],
    "gsbox_path": DEFAULT_SETTINGS["gsbox_path"],
    "spz_converter": DEFAULT_SETTINGS["spz_converter"],
}

OPTIMAL_PARAMETER_SETTINGS = {
    "ratio": DEFAULT_SETTINGS["ratio"],
    "k": DEFAULT_SETTINGS["k"],
    "opacity_threshold": DEFAULT_SETTINGS["opacity_threshold"],
    "lam_geo": DEFAULT_SETTINGS["lam_geo"],
    "lam_sh": DEFAULT_SETTINGS["lam_sh"],
    "device": DEFAULT_SETTINGS["device"],
    "block_edges": DEFAULT_SETTINGS["block_edges"],
    "quality_test_mode": DEFAULT_SETTINGS["quality_test_mode"],
}


@dataclass
class JobConfig:
    source_mode: str
    source_path: Path
    output_dir: Path | None
    recurse: bool
    output_suffix: str
    ratio: float
    k: int
    opacity_threshold: float
    lam_geo: float
    lam_sh: float
    device: str
    block_edges: int
    quality_test_mode: bool
    gsbox_path: str
    spz_converter: str


@dataclass(frozen=True)
class QualityVariant:
    index: int
    ratio: float
    k: int
    opacity_threshold: float
    lam_sh: float

    @property
    def slug(self) -> str:
        return (
            f"q{self.index:02d}_r{format_float_token(self.ratio)}"
            f"_k{self.k}_op{format_float_token(self.opacity_threshold)}"
            f"_sh{format_float_token(self.lam_sh)}"
        )


class QueueWriter:
    def __init__(self, log_queue: queue.Queue[tuple[str, object]]) -> None:
        self.log_queue = log_queue

    def write(self, text: str) -> int:
        if text:
            self.log_queue.put(("log", text))
        return len(text)

    def flush(self) -> None:
        return None


def quote_path(value: Path) -> str:
    return subprocess.list2cmdline([str(value)])


def format_float_token(value: float) -> str:
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def ordered_unique_float(values: list[float]) -> list[float]:
    seen: set[float] = set()
    unique: list[float] = []
    for value in values:
        rounded = round(value, 3)
        if rounded in seen:
            continue
        seen.add(rounded)
        unique.append(rounded)
    return unique


def build_ratio_grid(base: float) -> list[float]:
    values = [base - 0.10, base - 0.05, base, base + 0.05, base + 0.10]
    clamped = [min(0.95, max(0.10, value)) for value in values]
    return ordered_unique_float(clamped)


def build_opacity_grid(base: float) -> list[float]:
    values = [base - 0.02, base + 0.02]
    clamped = [min(0.95, max(0.0, value)) for value in values]
    return ordered_unique_float(clamped)


def build_lam_sh_grid(base: float) -> list[float]:
    values = [base - 0.5, base + 0.5]
    clamped = [max(0.0, value) for value in values]
    return ordered_unique_float(clamped)


def build_k_grid(base: int) -> list[int]:
    values = [max(1, base - 4), max(1, base + 4)]
    unique: list[int] = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return unique


def generate_quality_variants(config: JobConfig) -> list[QualityVariant]:
    variants: list[QualityVariant] = []
    combinations = itertools.product(
        build_ratio_grid(config.ratio),
        build_k_grid(config.k),
        build_opacity_grid(config.opacity_threshold),
        build_lam_sh_grid(config.lam_sh),
    )
    for index, (ratio, k, opacity_threshold, lam_sh) in enumerate(combinations, start=1):
        variants.append(
            QualityVariant(
                index=index,
                ratio=ratio,
                k=k,
                opacity_threshold=opacity_threshold,
                lam_sh=lam_sh,
            )
        )
    return variants


def resolve_gsbox_path(config: JobConfig) -> Path:
    configured = config.gsbox_path.strip()
    if configured:
        candidate = Path(configured)
        if not candidate.is_absolute():
            candidate = Path(__file__).resolve().parent / candidate
        if candidate.exists():
            return candidate

        found = shutil.which(configured)
        if found:
            return Path(found)

    default_candidate = DEFAULT_GSBOX_PATH
    if default_candidate.exists():
        return default_candidate

    found = shutil.which("gsbox.exe") or shutil.which("gsbox")
    if found:
        return Path(found)

    raise RuntimeError("gsbox.exe was not found. Set the GSBox executable path in the GUI.")


def run_gsbox(
    config: JobConfig,
    action: str,
    input_path: Path,
    output_path: Path,
    log_queue: queue.Queue[tuple[str, object]],
) -> None:
    gsbox_path = resolve_gsbox_path(config)
    command = [str(gsbox_path), action, "-i", str(input_path), "-o", str(output_path)]
    log_queue.put(("log", f"Running GSBox: {' '.join(command)}\n"))
    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.stdout:
        log_queue.put(("log", result.stdout))
    if result.stderr:
        log_queue.put(("log", result.stderr))
    if result.returncode != 0:
        raise RuntimeError(f"GSBox exited with code {result.returncode} while running {action}.")
    if not output_path.exists():
        raise RuntimeError(f"GSBox did not create the expected output file: {output_path}")


def collect_files(source_dir: Path, recurse: bool) -> list[Path]:
    pattern = "**/*" if recurse else "*"
    files = [
        path for path in source_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files)


def build_output_path(input_path: Path, config: JobConfig, source_root: Path | None) -> Path:
    base_dir = resolve_output_dir(input_path, config, source_root)
    suffix = config.output_suffix.strip()
    file_suffix = suffix if suffix else "_simplified"
    return base_dir / f"{input_path.stem}{file_suffix}{input_path.suffix.lower()}"


def resolve_output_dir(input_path: Path, config: JobConfig, source_root: Path | None) -> Path:
    if config.output_dir is None:
        base_dir = input_path.parent
    elif config.source_mode == "folder" and source_root is not None:
        try:
            relative_parent = input_path.parent.relative_to(source_root)
        except ValueError:
            relative_parent = Path()
        base_dir = config.output_dir / relative_parent
    else:
        base_dir = config.output_dir

    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def build_quality_test_dir(input_path: Path, config: JobConfig, source_root: Path | None) -> Path:
    base_dir = resolve_output_dir(input_path, config, source_root)
    suffix = config.output_suffix.strip()
    file_suffix = suffix if suffix else "_simplified"
    quality_dir = base_dir / f"{input_path.stem}{file_suffix}_quality_test"
    quality_dir.mkdir(parents=True, exist_ok=True)
    return quality_dir


def build_quality_output_path(input_path: Path, quality_dir: Path, variant: QualityVariant) -> Path:
    return quality_dir / f"{input_path.stem}_{variant.slug}{input_path.suffix.lower()}"


def build_processing_output_path(input_path: Path, final_output_path: Path, temp_dir: Path) -> Path:
    if input_path.suffix.lower() == ".ply":
        return final_output_path
    return temp_dir / f"{final_output_path.stem}.ply"


def resolve_input_file(
    input_path: Path,
    temp_dir: Path,
    config: JobConfig,
    log_queue: queue.Queue[tuple[str, object]],
) -> Path:
    ext = input_path.suffix.lower()
    if ext == ".ply":
        return input_path

    converted_path = temp_dir / f"{input_path.stem}.ply"
    if ext == ".sog":
        log_queue.put(("log", f"Converting {input_path.name} from .sog to temporary .ply\n"))
        header = convert_sog_to_ply(input_path, converted_path)
        log_queue.put(("log", f"SOG version {header.version}, splats {header.count}, SH degree {header.sh_degree}\n"))
        return converted_path

    if ext == ".spz":
        try:
            log_queue.put(("log", f"Converting {input_path.name} from .spz to temporary .ply with GSBox\n"))
            run_gsbox(config, "z2p", input_path, converted_path, log_queue)
            return converted_path
        except RuntimeError:
            converter = config.spz_converter.strip()
            if not converter:
                raise

    converter = config.spz_converter.strip()
    if not converter:
        raise RuntimeError(
            f"{input_path.name} is {ext} and NanoGS only reads .ply. "
            "Set the .spz converter command in the GUI to convert it before processing."
        )

    command = converter.format(input=quote_path(input_path), output=quote_path(converted_path))
    log_queue.put(("log", f"Converting {input_path.name} with: {command}\n"))
    result = subprocess.run(
        command,
        shell=True,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.stdout:
        log_queue.put(("log", result.stdout))
    if result.stderr:
        log_queue.put(("log", result.stderr))
    if result.returncode != 0:
        raise RuntimeError(f"Converter exited with code {result.returncode} for {input_path.name}.")
    if not converted_path.exists():
        raise RuntimeError(f"Converter did not create the expected PLY: {converted_path}")
    return converted_path


def export_output_file(
    input_path: Path,
    processed_ply_path: Path,
    final_output_path: Path,
    config: JobConfig,
    log_queue: queue.Queue[tuple[str, object]],
) -> None:
    ext = input_path.suffix.lower()
    if ext == ".ply":
        return
    if ext == ".sog":
        log_queue.put(("log", f"Exporting simplified model back to .sog with GSBox\n"))
        run_gsbox(config, "p2g", processed_ply_path, final_output_path, log_queue)
        return
    if ext == ".spz":
        log_queue.put(("log", f"Exporting simplified model back to .spz with GSBox\n"))
        run_gsbox(config, "p2z", processed_ply_path, final_output_path, log_queue)
        return
    raise RuntimeError(f"Unsupported export format: {ext}")


def write_quality_manifest(
    manifest_path: Path,
    input_path: Path,
    config: JobConfig,
    variants: list[QualityVariant],
    output_paths: list[Path],
) -> None:
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "input_file",
            "variant",
            "output_file",
            "ratio",
            "k",
            "opacity_threshold",
            "lam_geo",
            "lam_sh",
            "device",
            "block_edges",
        ])
        for variant, output_path in zip(variants, output_paths):
            writer.writerow([
                str(input_path),
                variant.slug,
                str(output_path),
                f"{variant.ratio:.3f}",
                variant.k,
                f"{variant.opacity_threshold:.3f}",
                f"{config.lam_geo:.3f}",
                f"{variant.lam_sh:.3f}",
                config.device,
                config.block_edges,
            ])


class NanoGSGui:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("NanoGS Batch GUI")
        self.root.geometry("980x760")
        self.root.minsize(880, 680)

        self.log_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.stop_requested = threading.Event()

        self.source_mode = tk.StringVar(value=DEFAULT_SETTINGS["source_mode"])
        self.source_path = tk.StringVar(value=DEFAULT_SETTINGS["source_path"])
        self.output_dir = tk.StringVar(value=DEFAULT_SETTINGS["output_dir"])
        self.recurse = tk.BooleanVar(value=DEFAULT_SETTINGS["recurse"])
        self.output_suffix = tk.StringVar(value=DEFAULT_SETTINGS["output_suffix"])
        self.ratio = tk.StringVar(value=DEFAULT_SETTINGS["ratio"])
        self.k = tk.StringVar(value=DEFAULT_SETTINGS["k"])
        self.opacity_threshold = tk.StringVar(value=DEFAULT_SETTINGS["opacity_threshold"])
        self.lam_geo = tk.StringVar(value=DEFAULT_SETTINGS["lam_geo"])
        self.lam_sh = tk.StringVar(value=DEFAULT_SETTINGS["lam_sh"])
        self.device = tk.StringVar(value=DEFAULT_SETTINGS["device"])
        self.block_edges = tk.StringVar(value=DEFAULT_SETTINGS["block_edges"])
        self.quality_test_mode = tk.BooleanVar(value=DEFAULT_SETTINGS["quality_test_mode"])
        self.gsbox_path = tk.StringVar(value=DEFAULT_SETTINGS["gsbox_path"])
        self.spz_converter = tk.StringVar(value=DEFAULT_SETTINGS["spz_converter"])
        self.profile_name = tk.StringVar()
        self.status_text = tk.StringVar(value="Idle")
        self.progress_text = tk.StringVar(value="0 / 0")
        self.saved_profiles: dict[str, dict[str, object]] = {}

        self._build_layout()
        self._load_settings()
        self._toggle_source_mode()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._drain_log_queue)

    def _collect_settings(self) -> dict[str, object]:
        return {
            "source_mode": self.source_mode.get(),
            "source_path": self.source_path.get(),
            "output_dir": self.output_dir.get(),
            "recurse": bool(self.recurse.get()),
            "output_suffix": self.output_suffix.get(),
            "ratio": self.ratio.get(),
            "k": self.k.get(),
            "opacity_threshold": self.opacity_threshold.get(),
            "lam_geo": self.lam_geo.get(),
            "lam_sh": self.lam_sh.get(),
            "device": self.device.get(),
            "block_edges": self.block_edges.get(),
            "quality_test_mode": bool(self.quality_test_mode.get()),
            "gsbox_path": self.gsbox_path.get(),
            "spz_converter": self.spz_converter.get(),
        }

    def _collect_profile_settings(self) -> dict[str, object]:
        return {
            "output_suffix": self.output_suffix.get(),
            "ratio": self.ratio.get(),
            "k": self.k.get(),
            "opacity_threshold": self.opacity_threshold.get(),
            "lam_geo": self.lam_geo.get(),
            "lam_sh": self.lam_sh.get(),
            "device": self.device.get(),
            "block_edges": self.block_edges.get(),
            "quality_test_mode": bool(self.quality_test_mode.get()),
            "gsbox_path": self.gsbox_path.get(),
            "spz_converter": self.spz_converter.get(),
        }

    def _apply_profile_settings(self, settings: dict[str, object]) -> None:
        merged = dict(DEFAULT_PROFILE_SETTINGS)
        merged.update(settings)

        self.output_suffix.set(str(merged["output_suffix"]))
        self.ratio.set(str(merged["ratio"]))
        self.k.set(str(merged["k"]))
        self.opacity_threshold.set(str(merged["opacity_threshold"]))
        self.lam_geo.set(str(merged["lam_geo"]))
        self.lam_sh.set(str(merged["lam_sh"]))

        device = str(merged["device"])
        if device not in {"auto", "cpu", "gpu"}:
            device = DEFAULT_SETTINGS["device"]
        self.device.set(device)

        self.block_edges.set(str(merged["block_edges"]))
        self.quality_test_mode.set(bool(merged["quality_test_mode"]))
        self.gsbox_path.set(str(merged["gsbox_path"]))
        self.spz_converter.set(str(merged["spz_converter"]))

    def _normalize_profiles(self, profiles: object) -> dict[str, dict[str, object]]:
        if not isinstance(profiles, dict):
            return {}

        normalized: dict[str, dict[str, object]] = {}
        for name, settings in profiles.items():
            if not isinstance(name, str) or not name.strip():
                continue
            if not isinstance(settings, dict):
                continue
            merged = dict(DEFAULT_PROFILE_SETTINGS)
            merged.update(settings)
            normalized[name.strip()] = merged
        return normalized

    def _refresh_profile_names(self) -> None:
        profile_names = sorted(self.saved_profiles)
        if hasattr(self, "profile_combo"):
            self.profile_combo.configure(values=profile_names)
        current_name = self.profile_name.get().strip()
        if current_name and current_name in self.saved_profiles:
            self.profile_name.set(current_name)
        elif profile_names:
            self.profile_name.set(profile_names[0])
        else:
            self.profile_name.set("")

    def _apply_settings(self, settings: dict[str, object]) -> None:
        self.source_mode.set(str(settings.get("source_mode", DEFAULT_SETTINGS["source_mode"])))
        self.source_path.set(str(settings.get("source_path", DEFAULT_SETTINGS["source_path"])))
        self.output_dir.set(str(settings.get("output_dir", DEFAULT_SETTINGS["output_dir"])))
        self.recurse.set(bool(settings.get("recurse", DEFAULT_SETTINGS["recurse"])))
        self.output_suffix.set(str(settings.get("output_suffix", DEFAULT_SETTINGS["output_suffix"])))
        self.ratio.set(str(settings.get("ratio", DEFAULT_SETTINGS["ratio"])))
        self.k.set(str(settings.get("k", DEFAULT_SETTINGS["k"])))
        self.opacity_threshold.set(str(settings.get("opacity_threshold", DEFAULT_SETTINGS["opacity_threshold"])))
        self.lam_geo.set(str(settings.get("lam_geo", DEFAULT_SETTINGS["lam_geo"])))
        self.lam_sh.set(str(settings.get("lam_sh", DEFAULT_SETTINGS["lam_sh"])))

        device = str(settings.get("device", DEFAULT_SETTINGS["device"]))
        if device not in {"auto", "cpu", "gpu"}:
            device = DEFAULT_SETTINGS["device"]
        self.device.set(device)

        self.block_edges.set(str(settings.get("block_edges", DEFAULT_SETTINGS["block_edges"])))
        self.quality_test_mode.set(bool(settings.get("quality_test_mode", DEFAULT_SETTINGS["quality_test_mode"])))
        self.gsbox_path.set(str(settings.get("gsbox_path", DEFAULT_SETTINGS["gsbox_path"])))
        self.spz_converter.set(str(settings.get("spz_converter", DEFAULT_SETTINGS["spz_converter"])))

    def _load_optimal_defaults(self) -> None:
        self._apply_profile_settings(OPTIMAL_PARAMETER_SETTINGS)
        self._append_log("Loaded optimal defaults.\n")

    def _save_profile(self) -> None:
        profile_name = self.profile_name.get().strip()
        if not profile_name:
            messagebox.showerror("NanoGS", "Enter a profile name first.")
            return

        self.saved_profiles[profile_name] = self._collect_profile_settings()
        self._refresh_profile_names()
        self.profile_name.set(profile_name)
        self._save_settings()
        self._append_log(f"Saved profile '{profile_name}'.\n")

    def _load_profile(self) -> None:
        profile_name = self.profile_name.get().strip()
        if not profile_name:
            messagebox.showerror("NanoGS", "Select or enter a profile name first.")
            return
        profile = self.saved_profiles.get(profile_name)
        if profile is None:
            messagebox.showerror("NanoGS", f"Profile '{profile_name}' was not found.")
            return

        self._apply_profile_settings(profile)
        self._append_log(f"Loaded profile '{profile_name}'.\n")

    def _delete_profile(self) -> None:
        profile_name = self.profile_name.get().strip()
        if not profile_name:
            messagebox.showerror("NanoGS", "Select or enter a profile name first.")
            return
        if profile_name not in self.saved_profiles:
            messagebox.showerror("NanoGS", f"Profile '{profile_name}' was not found.")
            return

        del self.saved_profiles[profile_name]
        self._refresh_profile_names()
        self._save_settings()
        self._append_log(f"Deleted profile '{profile_name}'.\n")

    def _load_settings(self) -> None:
        settings = dict(DEFAULT_SETTINGS)
        profiles: dict[str, dict[str, object]] = {}
        selected_profile = ""
        if SETTINGS_PATH.exists():
            try:
                loaded = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    profiles = self._normalize_profiles(loaded.get("profiles", {}))
                    selected_profile = str(loaded.get("selected_profile", "")).strip()
                    settings.update({
                        key: value for key, value in loaded.items()
                        if key not in {"profiles", "selected_profile"}
                    })
            except Exception:
                self._append_log(f"Warning: could not read {SETTINGS_PATH.name}. Using defaults.\n")
        self.saved_profiles = profiles
        self._apply_settings(settings)
        self._refresh_profile_names()
        if selected_profile and selected_profile in self.saved_profiles:
            self.profile_name.set(selected_profile)

    def _save_settings(self) -> None:
        settings = self._collect_settings()
        settings["profiles"] = self.saved_profiles
        settings["selected_profile"] = self.profile_name.get().strip()
        SETTINGS_PATH.write_text(json.dumps(settings, indent=2), encoding="utf-8")

    def _on_close(self) -> None:
        try:
            self._save_settings()
        except Exception as exc:
            messagebox.showerror("NanoGS", f"Could not save GUI settings: {exc}")
            return
        self.root.destroy()

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        container = ttk.Frame(self.root, padding=12)
        container.grid(sticky="nsew")
        container.columnconfigure(0, weight=1)
        container.rowconfigure(4, weight=1)

        source_frame = ttk.LabelFrame(container, text="Input")
        source_frame.grid(row=0, column=0, sticky="ew")
        source_frame.columnconfigure(1, weight=1)

        ttk.Radiobutton(
            source_frame,
            text="Single file",
            variable=self.source_mode,
            value="file",
            command=self._toggle_source_mode,
        ).grid(row=0, column=0, padx=8, pady=8, sticky="w")
        ttk.Radiobutton(
            source_frame,
            text="Folder",
            variable=self.source_mode,
            value="folder",
            command=self._toggle_source_mode,
        ).grid(row=0, column=1, padx=8, pady=8, sticky="w")

        ttk.Label(source_frame, text="Path").grid(row=1, column=0, padx=8, pady=6, sticky="w")
        ttk.Entry(source_frame, textvariable=self.source_path).grid(row=1, column=1, padx=8, pady=6, sticky="ew")
        self.source_browse_button = ttk.Button(source_frame, text="Browse", command=self._browse_source)
        self.source_browse_button.grid(row=1, column=2, padx=8, pady=6, sticky="ew")

        self.recurse_check = ttk.Checkbutton(
            source_frame,
            text="Include subfolders",
            variable=self.recurse,
        )
        self.recurse_check.grid(row=2, column=1, padx=8, pady=(0, 8), sticky="w")

        output_frame = ttk.LabelFrame(container, text="Output")
        output_frame.grid(row=1, column=0, pady=(12, 0), sticky="ew")
        output_frame.columnconfigure(1, weight=1)

        ttk.Label(output_frame, text="Output folder").grid(row=0, column=0, padx=8, pady=6, sticky="w")
        ttk.Entry(output_frame, textvariable=self.output_dir).grid(row=0, column=1, padx=8, pady=6, sticky="ew")
        ttk.Button(output_frame, text="Browse", command=self._browse_output_dir).grid(row=0, column=2, padx=8, pady=6, sticky="ew")
        ttk.Label(output_frame, text="Leave empty to write next to each source file.").grid(row=1, column=1, padx=8, pady=(0, 6), sticky="w")

        ttk.Label(output_frame, text="Output suffix").grid(row=2, column=0, padx=8, pady=6, sticky="w")
        ttk.Entry(output_frame, textvariable=self.output_suffix).grid(row=2, column=1, padx=8, pady=6, sticky="ew")
        ttk.Label(output_frame, text="Example: _simplified or _r25").grid(row=2, column=2, padx=8, pady=6, sticky="w")

        params_frame = ttk.LabelFrame(container, text="Parameters")
        params_frame.grid(row=2, column=0, pady=(12, 0), sticky="ew")
        for column in range(4):
            params_frame.columnconfigure(column, weight=1)

        self._add_labeled_entry(params_frame, 0, 0, "Keep ratio", self.ratio)
        self._add_labeled_entry(params_frame, 0, 2, "KNN k", self.k)
        self._add_labeled_entry(params_frame, 1, 0, "Opacity threshold", self.opacity_threshold)
        self._add_labeled_entry(params_frame, 1, 2, "Lambda geo", self.lam_geo)
        self._add_labeled_entry(params_frame, 2, 0, "Lambda SH", self.lam_sh)
        ttk.Label(params_frame, text="Device").grid(row=2, column=2, padx=8, pady=6, sticky="w")
        ttk.Combobox(
            params_frame,
            textvariable=self.device,
            values=("auto", "cpu", "gpu"),
            state="readonly",
        ).grid(row=2, column=3, padx=8, pady=6, sticky="ew")
        self._add_labeled_entry(params_frame, 3, 0, "Block edges", self.block_edges)
        ttk.Label(params_frame, text="0 = auto-tuned").grid(row=3, column=2, padx=8, pady=6, sticky="w")
        ttk.Checkbutton(
            params_frame,
            text="Qualitaetstestmodus (~40 Varianten)",
            variable=self.quality_test_mode,
        ).grid(row=4, column=0, columnspan=2, padx=8, pady=6, sticky="w")
        ttk.Label(
            params_frame,
            text="Erzeugt eine Vergleichsmatrix um die aktuellen Werte fuer Ratio, k, Opacity und Lambda SH.",
        ).grid(row=4, column=2, columnspan=2, padx=8, pady=6, sticky="w")
        ttk.Button(
            params_frame,
            text="Load optimal defaults",
            command=self._load_optimal_defaults,
        ).grid(row=5, column=0, columnspan=2, padx=8, pady=6, sticky="w")
        ttk.Label(
            params_frame,
            text="Laedt Ratio 0.3, k 8, Opacity 0.18, Lambda SH 0.5, Device auto.",
        ).grid(row=5, column=2, columnspan=2, padx=8, pady=6, sticky="w")

        profile_frame = ttk.LabelFrame(container, text="Profile Manager")
        profile_frame.grid(row=3, column=0, pady=(12, 0), sticky="ew")
        profile_frame.columnconfigure(1, weight=1)

        ttk.Label(profile_frame, text="Profile name").grid(row=0, column=0, padx=8, pady=6, sticky="w")
        self.profile_combo = ttk.Combobox(profile_frame, textvariable=self.profile_name)
        self.profile_combo.grid(row=0, column=1, padx=8, pady=6, sticky="ew")
        ttk.Button(profile_frame, text="Save current", command=self._save_profile).grid(row=0, column=2, padx=8, pady=6, sticky="ew")
        ttk.Button(profile_frame, text="Load", command=self._load_profile).grid(row=0, column=3, padx=8, pady=6, sticky="ew")
        ttk.Button(profile_frame, text="Delete", command=self._delete_profile).grid(row=0, column=4, padx=8, pady=6, sticky="ew")
        ttk.Label(
            profile_frame,
            text="Profiles speichern Parameter- und Exporteinstellungen, aber nicht Eingabedatei oder Ausgabeordner.",
        ).grid(row=1, column=0, columnspan=5, padx=8, pady=(0, 6), sticky="w")

        converter_frame = ttk.LabelFrame(container, text="Format handling")
        converter_frame.grid(row=4, column=0, pady=(12, 0), sticky="ew")
        converter_frame.columnconfigure(1, weight=1)

        ttk.Label(
            converter_frame,
            text=".sog and .spz are converted through temporary .ply files, then exported back to the original format with GSBox.",
        ).grid(row=0, column=0, columnspan=3, padx=8, pady=(8, 4), sticky="w")
        ttk.Label(converter_frame, text="GSBox executable").grid(row=1, column=0, padx=8, pady=6, sticky="w")
        ttk.Entry(converter_frame, textvariable=self.gsbox_path).grid(row=1, column=1, padx=8, pady=6, sticky="ew")
        ttk.Label(converter_frame, text="Default: gsbox.exe next to the GUI").grid(row=1, column=2, padx=8, pady=6, sticky="w")
        ttk.Label(converter_frame, text="Fallback SPZ converter").grid(row=2, column=0, padx=8, pady=6, sticky="w")
        ttk.Entry(converter_frame, textvariable=self.spz_converter).grid(row=2, column=1, padx=8, pady=6, sticky="ew")
        ttk.Label(converter_frame, text="Used only if GSBox is unavailable. Example: converter.exe {input} {output}").grid(row=2, column=2, padx=8, pady=6, sticky="w")

        run_frame = ttk.Frame(container)
        run_frame.grid(row=5, column=0, pady=(12, 0), sticky="nsew")
        run_frame.columnconfigure(0, weight=1)
        run_frame.rowconfigure(2, weight=1)

        controls = ttk.Frame(run_frame)
        controls.grid(row=0, column=0, sticky="ew")
        controls.columnconfigure(3, weight=1)

        self.run_button = ttk.Button(controls, text="Run", command=self._start_run)
        self.run_button.grid(row=0, column=0, padx=(0, 8), sticky="w")
        self.cancel_button = ttk.Button(controls, text="Cancel after current file", command=self._request_stop, state="disabled")
        self.cancel_button.grid(row=0, column=1, padx=(0, 8), sticky="w")
        ttk.Label(controls, textvariable=self.status_text).grid(row=0, column=2, sticky="w")
        ttk.Label(controls, textvariable=self.progress_text).grid(row=0, column=3, sticky="e")

        self.progress = ttk.Progressbar(run_frame, mode="determinate")
        self.progress.grid(row=1, column=0, pady=(8, 8), sticky="ew")

        self.log_output = ScrolledText(run_frame, wrap="word", height=20)
        self.log_output.grid(row=2, column=0, sticky="nsew")
        self.log_output.configure(state="disabled")

    def _add_labeled_entry(
        self,
        parent: ttk.LabelFrame,
        row: int,
        column: int,
        label: str,
        variable: tk.StringVar,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=column, padx=8, pady=6, sticky="w")
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=column + 1, padx=8, pady=6, sticky="ew")

    def _toggle_source_mode(self) -> None:
        is_folder = self.source_mode.get() == "folder"
        state = "normal" if is_folder else "disabled"
        self.recurse_check.configure(state=state)

    def _browse_source(self) -> None:
        if self.source_mode.get() == "folder":
            selected = filedialog.askdirectory(title="Select a folder with splat files")
        else:
            selected = filedialog.askopenfilename(
                title="Select a splat file",
                filetypes=[("Splat files", "*.ply *.sog *.spz"), ("All files", "*.*")],
            )
        if selected:
            self.source_path.set(selected)

    def _browse_output_dir(self) -> None:
        selected = filedialog.askdirectory(title="Select an output folder")
        if selected:
            self.output_dir.set(selected)

    def _request_stop(self) -> None:
        self.stop_requested.set()
        self.status_text.set("Stopping after current file")

    def _start_run(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            messagebox.showinfo("NanoGS", "A run is already in progress.")
            return

        try:
            config = self._read_config()
        except ValueError as exc:
            messagebox.showerror("Invalid input", str(exc))
            return

        self._save_settings()

        self.stop_requested.clear()
        self._set_running(True)
        self._append_log("Starting batch run.\n")
        self.worker = threading.Thread(target=self._run_jobs, args=(config,), daemon=True)
        self.worker.start()

    def _read_config(self) -> JobConfig:
        source_value = self.source_path.get().strip()
        if not source_value:
            raise ValueError("Select a file or folder first.")

        source_path = Path(source_value)
        source_mode = self.source_mode.get()
        if source_mode == "file" and not source_path.is_file():
            raise ValueError("The selected input file does not exist.")
        if source_mode == "folder" and not source_path.is_dir():
            raise ValueError("The selected input folder does not exist.")
        if source_mode == "file" and source_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError("Supported file types are .ply, .sog, and .spz.")
        if self.quality_test_mode.get() and source_mode != "file":
            raise ValueError("Quality test mode currently supports only a single input file.")

        ratio = float(self.ratio.get())
        if not (0.0 < ratio < 1.0):
            raise ValueError("Keep ratio must be between 0 and 1.")

        k_value = int(self.k.get())
        if k_value < 1:
            raise ValueError("KNN k must be at least 1.")

        opacity_threshold = float(self.opacity_threshold.get())
        lam_geo = float(self.lam_geo.get())
        lam_sh = float(self.lam_sh.get())
        block_edges = int(self.block_edges.get())
        if block_edges < 0:
            raise ValueError("Block edges must be 0 or a positive integer.")

        output_dir_value = self.output_dir.get().strip()
        output_dir = Path(output_dir_value) if output_dir_value else None

        return JobConfig(
            source_mode=source_mode,
            source_path=source_path,
            output_dir=output_dir,
            recurse=self.recurse.get(),
            output_suffix=self.output_suffix.get(),
            ratio=ratio,
            k=k_value,
            opacity_threshold=opacity_threshold,
            lam_geo=lam_geo,
            lam_sh=lam_sh,
            device=self.device.get(),
            block_edges=block_edges,
            quality_test_mode=self.quality_test_mode.get(),
            gsbox_path=self.gsbox_path.get(),
            spz_converter=self.spz_converter.get(),
        )

    def _run_jobs(self, config: JobConfig) -> None:
        try:
            if config.source_mode == "file":
                files = [config.source_path]
                source_root = None
            else:
                files = collect_files(config.source_path, config.recurse)
                source_root = config.source_path

            if not files:
                raise RuntimeError("No .ply, .sog, or .spz files were found in the selected folder.")

            if config.quality_test_mode:
                variants = generate_quality_variants(config)
                self.log_queue.put(("reset_progress", len(variants)))
                self.log_queue.put(("log", f"Quality test mode active: {len(variants)} variants will be generated.\n"))
            else:
                variants = []
                self.log_queue.put(("reset_progress", len(files)))

            with tempfile.TemporaryDirectory(prefix="nanogs_gui_") as temp_dir_name:
                temp_dir = Path(temp_dir_name)
                queue_writer = QueueWriter(self.log_queue)
                progress_value = 0
                for index, input_path in enumerate(files, start=1):
                    if self.stop_requested.is_set():
                        self.log_queue.put(("log", "Cancellation requested. Stopping batch.\n"))
                        break

                    self.log_queue.put(("status", f"Processing {input_path.name}"))
                    self.log_queue.put(("log", f"[{index}/{len(files)}] {input_path}\n"))
                    resolved_input = resolve_input_file(input_path, temp_dir, config, self.log_queue)
                    if config.quality_test_mode:
                        quality_dir = build_quality_test_dir(input_path, config, source_root)
                        output_paths: list[Path] = []
                        for variant in variants:
                            if self.stop_requested.is_set():
                                self.log_queue.put(("log", "Cancellation requested. Stopping quality test matrix.\n"))
                                break

                            variant_config = replace(
                                config,
                                ratio=variant.ratio,
                                k=variant.k,
                                opacity_threshold=variant.opacity_threshold,
                                lam_sh=variant.lam_sh,
                                quality_test_mode=False,
                            )
                            rp = RunParams(
                                ratio=variant_config.ratio,
                                k=variant_config.k,
                                opacity_threshold=variant_config.opacity_threshold,
                            )
                            cp = CostParams(
                                lam_geo=variant_config.lam_geo,
                                lam_sh=variant_config.lam_sh,
                                device=variant_config.device,
                                block_edges=variant_config.block_edges,
                            )
                            output_path = build_quality_output_path(input_path, quality_dir, variant)
                            processing_output_path = build_processing_output_path(input_path, output_path, temp_dir)
                            self.log_queue.put(("status", f"Variant {variant.index}/{len(variants)}"))
                            self.log_queue.put(("log", f"  Variant {variant.slug}: ratio={variant.ratio:.3f}, k={variant.k}, opacity={variant.opacity_threshold:.3f}, lam_sh={variant.lam_sh:.3f}\n"))

                            with contextlib.redirect_stdout(queue_writer), contextlib.redirect_stderr(queue_writer):
                                simplify(str(resolved_input), str(processing_output_path), rp, cp)

                            export_output_file(input_path, processing_output_path, output_path, variant_config, self.log_queue)
                            output_paths.append(output_path)
                            progress_value += 1
                            self.log_queue.put(("log", f"Wrote {output_path}\n\n"))
                            self.log_queue.put(("progress", progress_value))

                        manifest_path = quality_dir / "quality_test_matrix.csv"
                        write_quality_manifest(manifest_path, input_path, config, variants[: len(output_paths)], output_paths)
                        self.log_queue.put(("log", f"Wrote manifest {manifest_path}\n\n"))
                    else:
                        rp = RunParams(
                            ratio=config.ratio,
                            k=config.k,
                            opacity_threshold=config.opacity_threshold,
                        )
                        cp = CostParams(
                            lam_geo=config.lam_geo,
                            lam_sh=config.lam_sh,
                            device=config.device,
                            block_edges=config.block_edges,
                        )
                        output_path = build_output_path(input_path, config, source_root)
                        processing_output_path = build_processing_output_path(input_path, output_path, temp_dir)

                        with contextlib.redirect_stdout(queue_writer), contextlib.redirect_stderr(queue_writer):
                            simplify(str(resolved_input), str(processing_output_path), rp, cp)

                        export_output_file(input_path, processing_output_path, output_path, config, self.log_queue)

                        self.log_queue.put(("log", f"Wrote {output_path}\n\n"))
                        self.log_queue.put(("progress", index))

            self.log_queue.put(("done", None))
        except Exception as exc:
            self.log_queue.put(("error", str(exc)))

    def _set_running(self, running: bool) -> None:
        self.run_button.configure(state="disabled" if running else "normal")
        self.cancel_button.configure(state="normal" if running else "disabled")
        if running:
            self.status_text.set("Running")
        else:
            self.status_text.set("Idle")

    def _append_log(self, text: str) -> None:
        self.log_output.configure(state="normal")
        self.log_output.insert("end", text)
        self.log_output.see("end")
        self.log_output.configure(state="disabled")

    def _drain_log_queue(self) -> None:
        while True:
            try:
                kind, payload = self.log_queue.get_nowait()
            except queue.Empty:
                break

            if kind == "log":
                self._append_log(str(payload))
            elif kind == "status":
                self.status_text.set(str(payload))
            elif kind == "reset_progress":
                total = int(payload)
                self.progress.configure(maximum=max(total, 1), value=0)
                self.progress_text.set(f"0 / {total}")
            elif kind == "progress":
                value = int(payload)
                self.progress.configure(value=value)
                total = int(float(self.progress.cget("maximum")))
                self.progress_text.set(f"{value} / {total}")
            elif kind == "done":
                self._set_running(False)
                self.status_text.set("Finished")
                if self.stop_requested.is_set():
                    self.status_text.set("Stopped")
            elif kind == "error":
                self._set_running(False)
                self.status_text.set("Error")
                self._append_log(f"Error: {payload}\n")
                messagebox.showerror("NanoGS", str(payload))

        self.root.after(100, self._drain_log_queue)


def main() -> None:
    root = tk.Tk()
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    app = NanoGSGui(root)
    app.root.mainloop()


if __name__ == "__main__":
    main()