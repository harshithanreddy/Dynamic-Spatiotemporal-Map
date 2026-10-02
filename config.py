"""
Central config. Edit paths and region parameters here.
Nothing else in the codebase should hardcode a path.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

ROOT = Path(__file__).parent
RAW = ROOT / "data"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)


@dataclass
class RegionConfig:
    name: str
    raw_dir: Path
    landslide_path: Union[Path, List[Path]]
    scarp_path: Optional[Path]
    deformation_gpkg: Optional[Path]
    deformation_csv: Optional[Path]
    deformation_tif_ew: Optional[Path]
    deformation_tif_ud: Optional[Path]
    dem: Path
    fault_csv: Path
    earthquake_catalog: Path
    knet_dir: Path
    kik_dir: Path
    bbox: Dict[str, float]
    deformation_value_col: str
    deformation_value_col_secondary: Optional[str] = None
    fault_stub_mode: bool = False
    default_n_blocks_per_side: int = 6


def _find_file(folder: Path, pattern: str, allow_multiple: bool = False) -> Union[Path, List[Path]]:
    """
    Finds file(s) matching `pattern` in `folder`.
    If allow_multiple is True, returns a list of matching Paths.
    If allow_multiple is False, returns a single matching Path (or raises ValueError if multiple).
    """
    if not folder.exists():
        raise FileNotFoundError(f"Folder '{folder}' does not exist")
    matches = sorted(list(folder.glob(pattern)))
    if len(matches) == 0:
        raise FileNotFoundError(f"No file matching '{pattern}' found in {folder}")
    if len(matches) > 1 and not allow_multiple:
        raise ValueError(f"Multiple files matching '{pattern}' found in {folder}: {matches}")
    return matches if allow_multiple else matches[0]


def _build_noto_config() -> RegionConfig:
    raw_dir = RAW / "noto"
    return RegionConfig(
        name="noto",
        raw_dir=raw_dir,
        landslide_path=_find_file(raw_dir / "landslide", "20240101noto_land*.gpkg"),
        scarp_path=_find_file(raw_dir / "scarp", "20240101noto_scar*"),
        deformation_gpkg=_find_file(raw_dir / "deformation", "20240101noto_piv*.gpkg"),
        deformation_csv=_find_file(raw_dir / "deformation", "20240101noto_piv*.csv"),
        deformation_tif_ew=None,
        deformation_tif_ud=None,
        dem=raw_dir / "dem" / "output_AW3D30.tif",
        fault_csv=raw_dir / "faults" / "notokaigan_fault.csv",
        earthquake_catalog=raw_dir / "earthquakes" / "query.csv",
        knet_dir=raw_dir / "knet" / "csv",
        kik_dir=raw_dir / "kik" / "csv",
        bbox={
            "min_lon": 136.65,
            "max_lon": 137.35,
            "min_lat": 36.85,
            "max_lat": 37.55,
        },
        deformation_value_col="Mag(m)_local",
        deformation_value_col_secondary="Mag(m)_wide",
        fault_stub_mode=True,
        default_n_blocks_per_side=3,
    )


def _build_hokkaido_config() -> RegionConfig:
    raw_dir = RAW / "hokkaido"
    if not raw_dir.exists() and (RAW / "Hokkaido").exists():
        raw_dir = RAW / "Hokkaido"

    dem_path = raw_dir / "dem" / "output_AW3D30.tif"
    if not dem_path.exists() and (RAW / "noto" / "dem" / "output_AW3D30.tif").exists():
        dem_path = RAW / "noto" / "dem" / "output_AW3D30.tif"

    # TODO: The fault_csv (gem_active_faults.geojson) contains active fault traces across a broad area.
    # It needs bbox-filtering / isolation to the relevant Ishikari lowland eastern-margin fault segment
    # near Atsuma/Mukawa/Abira for Hokkaido coseismic feature extraction.
    return RegionConfig(
        name="hokkaido",
        raw_dir=raw_dir,
        landslide_path=_find_file(raw_dir / "landslide", "landslide_*.geojson", allow_multiple=True),
        scarp_path=None,
        deformation_gpkg=None,
        deformation_csv=None,
        deformation_tif_ew=raw_dir / "deformation" / "hokkaido_qEW.tif",
        deformation_tif_ud=raw_dir / "deformation" / "hokkaido_qUD.tif",
        dem=raw_dir / "dem" / "hokkaido_dem.tif",
        fault_csv=raw_dir / "active_fault" / "gem_active_faults.geojson",
        earthquake_catalog=raw_dir / "earthquake_catalog" / "hokkaido_jma_unified_catalog_2018.csv",
        knet_dir=raw_dir / "strong_motion" / "knet",
        kik_dir=raw_dir / "strong_motion" / "kik",
        bbox={
            # Hokkaido epicenter: (42.690 N, 142.007 E) near Atsuma/Iburi with ~0.6 deg padding
            "min_lon": 141.40,
            "max_lon": 142.60,
            "min_lat": 42.09,
            "max_lat": 43.29,
        },
        deformation_value_col="qEW",
        deformation_value_col_secondary="qUD",
        fault_stub_mode=False,
        default_n_blocks_per_side=8,
    )


_REGION_FACTORIES = {
    "noto": _build_noto_config,
    "hokkaido": _build_hokkaido_config,
}


class _RegionsDict(dict):
    """Lazy dictionary mapping region names to RegionConfig instances."""

    def __getitem__(self, key: str) -> RegionConfig:
        key_lower = key.lower()
        if key_lower in _REGION_FACTORIES:
            return get_region(key_lower)
        valid_options = list(_REGION_FACTORIES.keys())
        raise KeyError(f"Unrecognized region '{key}'. Valid options are: {valid_options}")

    def __contains__(self, key: object) -> bool:
        if isinstance(key, str):
            return key.lower() in _REGION_FACTORIES
        return False

    def keys(self):
        return _REGION_FACTORIES.keys()

    def __iter__(self):
        return iter(_REGION_FACTORIES.keys())


REGIONS: Dict[str, RegionConfig] = _RegionsDict()
_REGION_CACHE: Dict[str, RegionConfig] = {}


def get_region(name: str) -> RegionConfig:
    """
    Returns the RegionConfig for the specified region name (e.g. 'noto' or 'hokkaido').
    Raises a ValueError with valid options if the name is unrecognized.
    """
    key = name.lower()
    if key not in _REGION_FACTORIES:
        valid_options = list(_REGION_FACTORIES.keys())
        raise ValueError(f"Unrecognized region '{name}'. Valid options are: {valid_options}")
    if key not in _REGION_CACHE:
        _REGION_CACHE[key] = _REGION_FACTORIES[key]()
    return _REGION_CACHE[key]


# --- Non-region-specific module globals ---
GEOLOGY_POLY = RAW / "geology" / "seamlessV2_poly.shp"
if not GEOLOGY_POLY.exists() and (RAW / "noto" / "geology" / "seamlessV2_poly.shp").exists():
    GEOLOGY_POLY = RAW / "noto" / "geology" / "seamlessV2_poly.shp"

GEOLOGY_LEGEND = RAW / "geology" / "legend.tsv"
if not GEOLOGY_LEGEND.exists() and (RAW / "noto" / "geology" / "legend.tsv").exists():
    GEOLOGY_LEGEND = RAW / "noto" / "geology" / "legend.tsv"

GEOLOGY_CLASS_COL = "symbol"
GEOLOGY_LITHOLOGY_LEVEL = "lithology_en"

H3_RESOLUTION = 8  # ~0.74 km^2 per cell
METRIC_CRS = "EPSG:6675"
WGS84 = "EPSG:4326"

LABEL_COVERAGE_THRESHOLD = 0.05
RANDOM_SEED = 42


# --- Deprecated module-level aliases (pointing to "noto" region for backward compatibility) ---
# DEPRECATED: Direct module-level import of region-specific constants is deprecated.
# Migrate callers to accept a --region argument and call get_region(name).
_DEPRECATED_NOTO_MAPPING = {
    "LANDSLIDE_POLY": "landslide_path",
    "SCARP_LINE": "scarp_path",
    "DEFORMATION_GPKG": "deformation_gpkg",
    "DEFORMATION_CSV": "deformation_csv",
    "DEM": "dem",
    "FAULT_CSV": "fault_csv",
    "EARTHQUAKE_CATALOG": "earthquake_catalog",
    "KNET_DIR": "knet_dir",
    "KIK_DIR": "kik_dir",
    "NOTO_BBOX": "bbox",
    "DEFORMATION_VALUE_COL": "deformation_value_col",
    "DEFORMATION_VALUE_COL_SECONDARY": "deformation_value_col_secondary",
    "FAULT_STUB_MODE": "fault_stub_mode",
}


def __getattr__(name: str):
    if name in _DEPRECATED_NOTO_MAPPING:
        field_name = _DEPRECATED_NOTO_MAPPING[name]
        return getattr(get_region("noto"), field_name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
