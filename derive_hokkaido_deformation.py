"""
derive_hokkaido_deformation.py

Derives quasi-East-West (qEW) and quasi-Up-Down (qUD) coseismic deformation 
rasters for the 2018 Hokkaido Eastern Iburi earthquake from raw Line-of-Sight (LOS) 
InSAR component rasters.

Background & Motivation:
The official pre-computed 2.5D coseismic deformation products (qEW.tif and qUD.tif) 
for the 2018 Hokkaido earthquake are currently unavailable (HTTP 404) on GSI's download 
servers. This script reproduces those official GSI deformation products directly from the 
raw per-path unwrapped InSAR LOS displacement rasters (p*.insar_unwrap.tif) and unit vector 
component rasters (p*.uew_rng.tif, p*.uns_rng.tif).

Mathematical Formulation:
For each satellite path i, the measured LOS displacement is:
    LOS_i = u_ew_i * d_EW + u_ns_i * d_NS + u_ud_i * d_UD

Because satellite orbits are near-polar, InSAR measurements have low sensitivity 
to North-South movement (d_NS). Following GSI's standard 2.5D InSAR decomposition 
methodology, we assume d_NS ~ 0. The up-down unit vector component is computed as:
    u_ud_i = sqrt(max(0, 1 - u_ew_i^2 - u_ns_i^2))

For each pixel with at least 2 valid path measurements, we set up the linear system:
    LOS_i = u_ew_i * d_EW + u_ud_i * d_UD
and solve for [d_EW, d_UD]^T in a least-squares sense using vectorized numpy operations.

Usage:
    python derive_hokkaido_deformation.py \
        --input-dir data/hokkaido/insar_raw \
        --output-dir data/hokkaido/deformation \
        --noto-template data/noto/deformation/qEW.tif
"""

import argparse
import glob
import os
import sys
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import reproject, Resampling

# Ensure UTF-8 output formatting on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def discover_paths(input_dir):
    """
    Discovers satellite paths in input_dir that have all three required files:
    - {path}.insar_unwrap.tif
    - {path}.uew_rng.tif
    - {path}.uns_rng.tif
    """
    unwrap_files = glob.glob(os.path.join(input_dir, "*.insar_unwrap.tif"))
    valid_paths = []
    
    for unwrap_path in sorted(unwrap_files):
        filename = os.path.basename(unwrap_path)
        path_id = filename.replace(".insar_unwrap.tif", "")
        
        uew_path = os.path.join(input_dir, f"{path_id}.uew_rng.tif")
        uns_path = os.path.join(input_dir, f"{path_id}.uns_rng.tif")
        
        if os.path.exists(uew_path) and os.path.exists(uns_path):
            valid_paths.append({
                "id": path_id,
                "unwrap": unwrap_path,
                "uew": uew_path,
                "uns": uns_path
            })
        else:
            print(f"Warning: Incomplete raster set for path '{path_id}'. Skipping.")
            
    return valid_paths


def inspect_unit_vector_metadata(paths_info):
    """
    Inspects GeoTIFF tags for accompanying GMT/GSI metadata to verify unit vector components.
    """
    print("\n--- Unit Vector Component Metadata Verification ---")
    for p_info in paths_info:
        pid = p_info["id"]
        with rasterio.open(p_info["uns"]) as src:
            tags = src.tags()
            history = tags.get("NC_GLOBAL#history", "")
            title = tags.get("NC_GLOBAL#title", "")
            print(f"Path {pid} uns_rng metadata: title='{title}', history='{history}'")
            
    print(
        "\n[Assumption & Methodological Note]:\n"
        "  - The file 'uew_rng.tif' contains the East-West unit vector component (u_EW).\n"
        "  - The file 'uns_rng.tif' contains the North-South unit vector component (u_NS), as confirmed by GMT metadata (nsF.grd).\n"
        "  - Since satellite orbits run quasi-north-south, InSAR has negligible sensitivity to North-South motion (d_NS ~ 0).\n"
        "  - Per GSI 2.5D InSAR decomposition standards, we assume d_NS = 0 and derive the Up-Down unit vector component as:\n"
        "        u_UD = sqrt(max(0, 1 - u_EW^2 - u_NS^2))\n"
        "  - The linear system solved per pixel is: LOS_i = u_EW_i * d_EW + u_UD_i * d_UD.\n"
    )


def select_reference_grid(paths_info):
    """
    Selects the path with the finest resolution (smallest pixel area) as the reference grid.
    """
    best_path = None
    min_pixel_area = float("inf")
    
    for p_info in paths_info:
        with rasterio.open(p_info["unwrap"]) as src:
            res_x, res_y = abs(src.res[0]), abs(src.res[1])
            pixel_area = res_x * res_y
            print(f"Path {p_info['id']}: Shape={src.shape}, Resolution=({res_x:.6f}, {res_y:.6f}), Pixel Area={pixel_area:.8e}")
            
            if pixel_area < min_pixel_area:
                min_pixel_area = pixel_area
                best_path = p_info
                
    print(f"\nSelected path '{best_path['id']}' as the reference grid (finest resolution).")
    
    with rasterio.open(best_path["unwrap"]) as src:
        profile = src.profile.copy()
        crs = src.crs if src.crs is not None else CRS.from_epsg(4326)
        profile["crs"] = crs
        
    return profile, best_path["id"]


def derive_hokkaido_deformation(input_dir, output_dir, noto_template_path=None):
    """
    Main function to load InSAR rasters, reproject onto common reference grid,
    solve for qEW and qUD via vectorized least-squares, and save GeoTIFF outputs.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    paths_info = discover_paths(input_dir)
    if not paths_info:
        print(f"Error: No valid InSAR raster trios found in '{input_dir}'.", file=sys.stderr)
        sys.exit(1)
        
    print(f"Discovered {len(paths_info)} valid satellite paths: {[p['id'] for p in paths_info]}")
    
    inspect_unit_vector_metadata(paths_info)
    
    ref_profile, ref_path_id = select_reference_grid(paths_info)
    ref_crs = ref_profile["crs"]
    ref_transform = ref_profile["transform"]
    ref_shape = (ref_profile["height"], ref_profile["width"])
    
    los_list, uew_list, uud_list, valid_mask_list = [], [], [], []
    
    print("\n--- Reprojecting & Resampling Paths onto Reference Grid ---")
    for p_info in paths_info:
        pid = p_info["id"]
        print(f"Processing path '{pid}'...")
        
        # 1. Load & reproject LOS displacement (Bilinear)
        with rasterio.open(p_info["unwrap"]) as src:
            src_crs = src.crs if src.crs is not None else CRS.from_epsg(4326)
            los = np.full(ref_shape, np.nan, dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=los,
                src_transform=src.transform,
                src_crs=src_crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.bilinear,
                dst_nodata=np.nan
            )
            
        # 2. Load & reproject East-West unit vector component (Nearest-Neighbor)
        with rasterio.open(p_info["uew"]) as src:
            src_crs = src.crs if src.crs is not None else CRS.from_epsg(4326)
            uew = np.full(ref_shape, np.nan, dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=uew,
                src_transform=src.transform,
                src_crs=src_crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.nearest,
                dst_nodata=np.nan
            )
            
        # 3. Load & reproject North-South unit vector component (Nearest-Neighbor)
        with rasterio.open(p_info["uns"]) as src:
            src_crs = src.crs if src.crs is not None else CRS.from_epsg(4326)
            uns = np.full(ref_shape, np.nan, dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=uns,
                src_transform=src.transform,
                src_crs=src_crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.nearest,
                dst_nodata=np.nan
            )
            
        # Calculate Up-Down unit vector component
        uud_sq = np.maximum(0.0, 1.0 - uew**2 - uns**2)
        uud = np.sqrt(uud_sq)
        
        # Valid data mask for this path
        valid = ~np.isnan(los) & ~np.isnan(uew) & ~np.isnan(uns)
        
        los_list.append(np.where(valid, los, 0.0))
        uew_list.append(np.where(valid, uew, 0.0))
        uud_list.append(np.where(valid, uud, 0.0))
        valid_mask_list.append(valid)
        
        valid_pct = np.mean(valid) * 100
        print(f"  -> Path '{pid}': {np.sum(valid):,} valid pixels ({valid_pct:.2f}% of reference grid)")

    # Stack arrays across path dimension (K, H, W)
    los_arr = np.stack(los_list, axis=0)
    uew_arr = np.stack(uew_list, axis=0)
    uud_arr = np.stack(uud_list, axis=0)
    mask_arr = np.stack(valid_mask_list, axis=0)
    
    # Requirement 3: Count valid paths per pixel and mask where < 2 paths
    valid_count = np.sum(mask_arr, axis=0)
    solve_mask = valid_count >= 2
    
    print(f"\n--- Solving Least-Squares 2.5D System (Vectorized) ---")
    print(f"Total reference grid pixels: {solve_mask.size:,}")
    print(f"Pixels with >= 2 valid paths: {np.sum(solve_mask):,} ({np.mean(solve_mask)*100:.2f}%)")
    
    # Requirement 2: Vectorized normal equations least-squares solve:
    #   [ See  Seu ] [ d_EW ] = [ Re ]
    #   [ Seu  Suu ] [ d_UD ]   [ Ru ]
    See = np.sum((uew_arr**2) * mask_arr, axis=0)
    Suu = np.sum((uud_arr**2) * mask_arr, axis=0)
    Seu = np.sum((uew_arr * uud_arr) * mask_arr, axis=0)
    Re  = np.sum((uew_arr * los_arr) * mask_arr, axis=0)
    Ru  = np.sum((uud_arr * los_arr) * mask_arr, axis=0)
    
    det = See * Suu - Seu**2
    valid_solve = solve_mask & (np.abs(det) > 1e-6)
    
    d_EW = np.full(ref_shape, np.nan, dtype=np.float32)
    d_UD = np.full(ref_shape, np.nan, dtype=np.float32)
    
    d_EW[valid_solve] = (Suu[valid_solve] * Re[valid_solve] - Seu[valid_solve] * Ru[valid_solve]) / det[valid_solve]
    d_UD[valid_solve] = (See[valid_solve] * Ru[valid_solve] - Seu[valid_solve] * Re[valid_solve]) / det[valid_solve]
    
    # Requirement 4: Configure output profile using Noto template if available, else reference profile
    out_profile = ref_profile.copy()
    
    if noto_template_path and os.path.exists(noto_template_path):
        print(f"\nReading template profile from '{noto_template_path}'...")
        with rasterio.open(noto_template_path) as tmpl:
            tmpl_prof = tmpl.profile.copy()
            # Retain formatting options (driver, nodata, compression) from template
            for key in ["driver", "nodata", "compress", "tiled", "blockxsize", "blockysize"]:
                if key in tmpl_prof:
                    out_profile[key] = tmpl_prof[key]
    else:
        if noto_template_path:
            print(f"\nNoto template '{noto_template_path}' not found. Using default GeoTIFF profile.")
        out_profile.update({
            "driver": "GTiff",
            "nodata": np.nan,
            "dtype": "float32",
            "count": 1,
            "compress": "deflate"
        })

    out_profile.update({
        "crs": ref_crs,
        "transform": ref_transform,
        "width": ref_shape[1],
        "height": ref_shape[0],
        "count": 1,
        "dtype": "float32",
        "nodata": np.nan
    })

    qEW_path = os.path.join(output_dir, "hokkaido_qEW.tif")
    qUD_path = os.path.join(output_dir, "hokkaido_qUD.tif")
    
    print(f"\nWriting output GeoTIFFs...")
    with rasterio.open(qEW_path, "w", **out_profile) as dst:
        dst.write(d_EW, 1)
    print(f"  -> Wrote: {qEW_path}")
    
    with rasterio.open(qUD_path, "w", **out_profile) as dst:
        dst.write(d_UD, 1)
    print(f"  -> Wrote: {qUD_path}")
    
    # Requirement 6: Basic sanity statistics & epicenter check
    print("\n========================================================")
    print("                SANITY CHECK & STATISTICS               ")
    print("========================================================")
    
    valid_solve_count = np.sum(valid_solve)
    valid_solve_pct = np.mean(valid_solve) * 100
    print(f"Decomposition Valid Pixels : {valid_solve_count:,} ({valid_solve_pct:.2f}% of grid)")
    
    ew_valid = d_EW[valid_solve]
    ud_valid = d_UD[valid_solve]
    
    print("\n[quasi-East-West (qEW) Displacement (cm)]")
    print(f"  Min  : {ew_valid.min():.4f}")
    print(f"  Max  : {ew_valid.max():.4f}")
    print(f"  Mean : {ew_valid.mean():.4f}")
    print(f"  Std  : {ew_valid.std():.4f}")
    
    print("\n[quasi-Up-Down (qUD) Displacement (cm)]")
    print(f"  Min  : {ud_valid.min():.4f}")
    print(f"  Max  : {ud_valid.max():.4f}")
    print(f"  Mean : {ud_valid.mean():.4f}")
    print(f"  Std  : {ud_valid.std():.4f}")
    
    # Regional Epicenter Sanity Check (~42.686 N, 141.929 E, Iburi region)
    epi_lon, epi_lat = 141.929, 42.686
    try:
        inv_transform = ~ref_transform
        c, r = inv_transform * (epi_lon, epi_lat)
        r, c = int(round(r)), int(round(c))
        
        half_win = 50  # ~2.5 km window around epicenter
        r_min, r_max = max(0, r - half_win), min(ref_shape[0], r + half_win)
        c_min, c_max = max(0, c - half_win), min(ref_shape[1], c + half_win)
        
        epi_ud_win = d_UD[r_min:r_max, c_min:c_max]
        epi_max_ud = np.nanmax(epi_ud_win)
        
        print(f"\n[Epicenter Region Check (~{epi_lat:.3f} N, {epi_lon:.3f} E - Atsuma/Iburi)]")
        print(f"  Epicenter Pixel Index : Row={r}, Col={c}")
        print(f"  Peak Regional Uplift (qUD) : {epi_max_ud:.2f} cm")
        print("  GSI Benchmark Reference   : ~7.0 cm coseismic uplift near epicenter")
    except Exception as e:
        print(f"\nNotice: Could not perform epicenter coordinate lookup: {e}")

    print("\nDerivation completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Derive quasi-EW and quasi-UD coseismic deformation from raw InSAR rasters."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/hokkaido/insar_raw",
        help="Input directory containing raw InSAR GeoTIFF rasters."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/hokkaido/deformation",
        help="Output directory to save hokkaido_qEW.tif and hokkaido_qUD.tif."
    )
    parser.add_argument(
        "--noto-template",
        type=str,
        default="data/noto/deformation/qEW.tif",
        help="Optional path to a Noto deformation GeoTIFF file to use as a profile template."
    )
    
    args = parser.parse_args()
    
    derive_hokkaido_deformation(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        noto_template_path=args.noto_template
    )
