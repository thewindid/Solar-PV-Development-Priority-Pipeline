import os
import sys
import json
import math
import logging
import warnings
from pathlib   import Path
from datetime  import datetime

import numpy           as np
import pandas          as pd
import geopandas       as gpd
from   shapely.geometry import Point, mapping
from   scipy.spatial    import cKDTree
from   scipy.interpolate import griddata
from   scipy.stats       import pearsonr

from   sklearn.preprocessing    import MinMaxScaler
from   sklearn.ensemble         import RandomForestRegressor
from   sklearn.model_selection  import KFold, cross_val_predict
from   sklearn.metrics          import r2_score, mean_absolute_error

import matplotlib
import matplotlib.pyplot    as plt
import matplotlib.patches   as mpatches
import matplotlib.colors    as mcolors
from   matplotlib.colors     import LinearSegmentedColormap, BoundaryNorm
import matplotlib.ticker     as ticker
import seaborn               as sns

_opt = {}
for _pkg, _key in [
    ('contextily', 'ctx'),
    ('folium',     'folium'),
    ('rasterstats','rst'),
    ('rasterio',   'rio'),
]:
    try:
        _opt[_key] = __import__(_pkg)
    except ImportError:
        _opt[_key] = None

warnings.filterwarnings('ignore')
matplotlib.rcParams.update({
    'font.family'      : 'DejaVu Sans',
    'figure.dpi'       : 150,
    'savefig.dpi'      : 300,
    'savefig.bbox'     : 'tight',
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'axes.grid'        : True,
    'grid.alpha'       : 0.3,
})


CFG = dict(

    # ── Jalur file ────────────────────────────────────────────────────────────
    bldg_path  = 'data/Building_Footprint.geojson',
    rad_path   = 'data/radiance_data.csv',       # CSV | GeoJSON | .tif
    cf_path    = 'data/cloud_fraction.csv',       # dari task1_gee_fetch.py
    out_dir    = 'output/task1/',

    # ── CRS ───────────────────────────────────────────────────────────────────
    # geo_crs  : untuk koordinat geografis (lon/lat)
    # proj_crs : untuk perhitungan metrik (jarak, luas).
    #            GANTI dengan UTM zone yang sesuai area studi.
    #            Contoh: UTM48N=32648, UTM49S=32749, UTM47N=32647
    geo_crs    = 'EPSG:4326',
    proj_crs   = 'EPSG:32648',

    # ── Parameter teknis PV ───────────────────────────────────────────────────
    # Referensi: IEC 61215, PVGIS, NREL SAM
    eta        = 0.20,     # Efisiensi panel (monocrystalline Si, STC 25°C)
    PR         = 0.78,     # Performance Ratio (kabel + inverter + debu + mismatch)
    rho_roof   = 0.65,     # Fraksi atap yang dapat digunakan (after HVAC, parapet, dll)
    G_stc      = 1000,     # W/m² — Standard Test Conditions irradiance
    NOCT       = 45,       # Normal Operating Cell Temperature (°C)
    T_coeff    = -0.004,   # Koefisien temperatur daya (%/°C di atas 25°C)
    T_amb      = 28,       # Temperatur ambient rata-rata tahunan (°C, tropis)
    albedo     = 0.20,     # Reflektansi permukaan (tanah/jalan perkotaan)

    # ── Geometri atap ─────────────────────────────────────────────────────────
    tilt_deg   = 10,       # Kemiringan panel di atas atap datar (°) — optimal tropis
    azimuth    = 360,      # Azimuth panel (180° = menghadap equator)

    # ── Model shading ─────────────────────────────────────────────────────────
    shadow_r_m = 50,       # Radius pencarian bangunan tetangga untuk shading (m)
    solar_elv  = 60,       # Sudut elevasi matahari rata-rata (°, tropis)
    cf_weight  = 0.30,     # Bobot Cloud Fraction dalam shading factor total
    # Bobot building shadow = 1 - cf_weight

    # ── Klasifikasi indeks ────────────────────────────────────────────────────
    thr_low    = 0.33,     # Batas bawah kelas Tinggi
    thr_high   = 0.67,     # Batas atas kelas Sedang

    # ── ML (RandomForest height smoothing) ───────────────────────────────────
    rf_n       = 150,      # Jumlah pohon
    rf_seed    = 42,
    cv_k       = 5,        # Lipatan K-Fold cross-validation
    min_valid_h= 20,       # Minimal data valid untuk melatih model
)

# ── Buat direktori output ────────────────────────────────────────────────────
OUT = CFG['out_dir']
for _sub in ['', 'maps/', 'figures/', 'data/']:
    Path(OUT + _sub).mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# §2  LOGGING
# ═══════════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s │ %(levelname)-7s │ %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(OUT + 'task1.log', mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger('Task1')


# ── Warna & colormap ─────────────────────────────────────────────────────────
PV_CMAP = LinearSegmentedColormap.from_list(
    'solar_pv', ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#1a9850'], N=256
)
CLASS_COLORS = {'Rendah': '#d73027', 'Sedang': '#fee08b', 'Tinggi': '#1a9850'}
CLASS_ORDER  = ['Rendah', 'Sedang', 'Tinggi']


# ═══════════════════════════════════════════════════════════════════════════════
# §3  LOADING DATA
# ═══════════════════════════════════════════════════════════════════════════════

def load_buildings(path: str) -> gpd.GeoDataFrame:
    """
    Muat polygon bangunan dari GeoJSON.
    Atribut yang diharapkan: geometry (Polygon), height (m),
    area_m2 / luas (opsional, dihitung jika tidak ada).
    """
    log.info(f'[LOAD] Building Footprint ← {path}')
    gdf = gpd.read_file(path)
    gdf = gdf[gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])].copy()
    log.info(f'       {len(gdf):,} polygon bangunan | CRS: {gdf.crs}')
    return gdf


def _detect_format(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext in ('.csv', '.tsv', '.txt'): return 'csv'
    if ext in ('.geojson', '.json'):    return 'geojson'
    if ext in ('.tif', '.tiff'):        return 'raster'
    return 'csv'


def _erbs_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimasi DHI dan DNI dari GHI menggunakan model Erbs (1982).
    Asumsi clearness index kt = 0.55 (tropis).
    Referensi: Erbs, D.G., Klein, S.A., Duffie, J.A. (1982).
    """
    kt = 0.55
    if kt <= 0.22:
        fd = 1.0 - 0.09 * kt
    elif kt <= 0.80:
        fd = (0.9511 - 0.1604*kt + 4.388*kt**2
              - 16.638*kt**3 + 12.336*kt**4)
    else:
        fd = 0.165
    df = df.copy()
    df['DHI'] = df['GHI'] * fd
    cos_z     = max(math.cos(math.radians(30)), 0.01)
    df['DNI'] = (df['GHI'] - df['DHI']) / cos_z
    return df


def load_radiance(path: str) -> pd.DataFrame:
    """
    Muat data irradiance per titik (GHI/DNI/DHI).
    Format CSV kolom yang diterima: lat/latitude/y, lon/longitude/x,
    GHI [kWh/m²/yr], DNI [kWh/m²/yr], DHI [kWh/m²/yr].
    Jika DNI/DHI tidak ada, diturunkan via model Erbs.
    """
    log.info(f'[LOAD] Radiance data ← {path}')
    fmt = _detect_format(path)

    if fmt == 'csv':
        df = pd.read_csv(path)
    elif fmt == 'geojson':
        gdf = gpd.read_file(path)
        df  = pd.DataFrame(gdf.drop(columns='geometry'))
        df['lon'] = gdf.geometry.x
        df['lat'] = gdf.geometry.y
    else:
        raise ValueError(f'Format raster: gunakan fungsi load_radiance_raster()')

    # Normalisasi nama kolom
    rename = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl in ('lat', 'latitude', 'y'):            rename[c] = 'lat'
        elif cl in ('lon', 'longitude', 'x', 'lng'): rename[c] = 'lon'
        elif cl == 'ghi':                             rename[c] = 'GHI'
        elif cl == 'dni':                             rename[c] = 'DNI'
        elif cl == 'dhi':                             rename[c] = 'DHI'
    df = df.rename(columns=rename)

    if 'GHI' not in df.columns:
        raise ValueError('Kolom GHI tidak ditemukan di data radiance!')

    if 'DNI' not in df.columns or 'DHI' not in df.columns:
        log.warning('DNI/DHI tidak ada — estimasi via model Erbs (kt=0.55)')
        df = _erbs_decomposition(df)

    df = df.dropna(subset=['lat', 'lon', 'GHI'])
    log.info(f'       {len(df):,} titik irradiance | '
             f'GHI mean: {df["GHI"].mean():.1f} kWh/m²/yr')
    return df


def load_radiance_raster(tif_path: str, band_map: dict = None) -> pd.DataFrame:
    """
    Muat data irradiance dari GeoTIFF.
    band_map: {1: 'GHI', 2: 'DNI', 3: 'DHI'} — sesuaikan urutan band.
    Mengembalikan DataFrame titik sampel dengan kolom lat, lon, GHI, DNI, DHI.
    """
    if _opt['rio'] is None:
        raise ImportError('rasterio diperlukan untuk membaca .tif')

    rio = _opt['rio']
    bm  = band_map or {1: 'GHI', 2: 'DNI', 3: 'DHI'}
    log.info(f'[LOAD] Radiance raster ← {tif_path}')

    rows = []
    with rio.open(tif_path) as src:
        data = src.read()
        xs, ys = np.meshgrid(
            np.linspace(src.bounds.left,  src.bounds.right,  src.width),
            np.linspace(src.bounds.bottom, src.bounds.top,   src.height)
        )
        for band_idx, col_name in bm.items():
            band_data = data[band_idx - 1].ravel()
            if not rows:
                rows = [{'lon': x, 'lat': y} for x, y in
                        zip(xs.ravel(), ys.ravel())]
            for i, row in enumerate(rows):
                row[col_name] = float(band_data[i])

    df = pd.DataFrame(rows)
    df = df.dropna()
    log.info(f'       {len(df):,} piksel raster | GHI mean: {df["GHI"].mean():.1f}')
    return df


def load_cloud_fraction(path: str, buildings: gpd.GeoDataFrame = None) -> pd.DataFrame:
    """
    Muat Cloud Fraction dari CSV (output task1_gee_fetch.py).
    Kolom: lat, lon, CF ∈ [0, 1].
    Jika file tidak ada, gunakan nilai default CF = 0.30 (tropis moderat).
    """
    if not Path(path).exists():
        log.warning(f'File CF tidak ditemukan ({path}) → default CF = 0.30')
        if buildings is not None:
            c = buildings.to_crs('EPSG:4326').geometry.centroid
            return pd.DataFrame({'lon': c.x, 'lat': c.y, 'CF': 0.30})
        return pd.DataFrame({'lon': [0.0], 'lat': [0.0], 'CF': [0.30]})

    log.info(f'[LOAD] Cloud Fraction ← {path}')
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    for old, new in [('latitude','lat'),('longitude','lon'),
                     ('cloud_fraction','cf'),('cloud_fraction_mean','cf')]:
        if old in df.columns:
            df = df.rename(columns={old: new})
    df = df.rename(columns={'cf': 'CF'})
    df = df.dropna(subset=['lat', 'lon', 'CF'])
    df['CF'] = df['CF'].clip(0, 1)
    log.info(f'       {len(df):,} titik CF | mean CF = {df["CF"].mean():.3f}')
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# §4  PRE-PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_buildings(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Standardisasi CRS, hitung atribut geometri, bersihkan data.
    Output: GDF dalam CFG['proj_crs'] untuk perhitungan metrik.
    """
    log.info('[PREP] Pre-processing bangunan...')

    # 1. Standardisasi CRS
    if gdf.crs is None:
        gdf = gdf.set_crs(CFG['geo_crs'])
        log.info(f'       CRS tidak terdeteksi → set ke {CFG["geo_crs"]}')
    proj_epsg = int(CFG['proj_crs'].split(':')[1])
    if gdf.crs.to_epsg() != proj_epsg:
        gdf = gdf.to_crs(CFG['proj_crs'])
    log.info(f'       CRS diubah → {gdf.crs}')

    # 2. Perbaiki geometri (self-intersection, dll)
    gdf['geometry'] = gdf['geometry'].buffer(0)
    gdf = gdf[gdf.geometry.is_valid & ~gdf.geometry.is_empty].copy()

    # 3. Hitung luas atap (footprint area)
    if 'area_m2' not in gdf.columns:
        # Cek kolom luas dalam berbagai nama
        area_col = next(
            (c for c in gdf.columns
             if any(k in c.lower() for k in ('area', 'luas', 'luas_m'))),
            None
        )
        if area_col:
            gdf = gdf.rename(columns={area_col: 'area_m2'})
            log.info(f'       area_m2 ← kolom "{area_col}"')
        else:
            gdf['area_m2'] = gdf.geometry.area
            log.info('       area_m2 dihitung dari geometri')

   # 4. Normalisasi kolom tinggi bangunan
    height_col = next(
        (c for c in gdf.columns
         if any(k in c.lower() for k in ('height', 'tinggi', 'elev', 'h_'))),
        None
    )
    if height_col and height_col != 'height':
        gdf = gdf.rename(columns={height_col: 'height'})
    if 'height' not in gdf.columns:
        log.warning('       Kolom tinggi tidak ditemukan → default 6 m (2 lantai)')
        gdf['height'] = 6.0

    # =====================================================================
    # TAMBAHKAN BARIS INI UNTUK MEMAKSA KONVERSI TEKS MENJADI ANGKA (FLOAT)
    # =====================================================================
    gdf['height'] = pd.to_numeric(gdf['height'], errors='coerce')

    # 5. Isi nilai tinggi yang hilang (akan diperbaiki oleh ML di §6)
    n_miss = gdf['height'].isna().sum()
    if n_miss:
        med = gdf['height'].median()
        gdf['height'] = gdf['height'].fillna(med)
        log.warning(f'       {n_miss} nilai height kosong → diisi median ({med:.1f} m)')

    # 6. Hapus slivers (bangunan terlalu kecil: < 3×3 m)
    gdf = gdf[gdf['area_m2'] > 9.0].copy()

    # 7. Tambah ID unik dan fitur geometri tambahan
    gdf = gdf.reset_index(drop=True)
    gdf['bldg_id']    = gdf.index.astype(int)
    gdf['perimeter_m'] = gdf.geometry.length
    gdf['compactness'] = (4 * math.pi * gdf['area_m2'] /
                          (gdf['perimeter_m'].clip(lower=1e-6)**2))

    log.info(f'       {len(gdf):,} bangunan valid | '
             f'luas: {gdf["area_m2"].mean():.0f} m² (rata-rata) | '
             f'tinggi: {gdf["height"].mean():.1f} m (rata-rata)')
    return gdf


# ═══════════════════════════════════════════════════════════════════════════════
# §5  PENUGASAN IRRADIANCE KE BANGUNAN
# ═══════════════════════════════════════════════════════════════════════════════

def assign_irradiance(buildings: gpd.GeoDataFrame,
                       rad_df: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Task 1.1a — Menetapkan nilai GHI/DNI/DHI dari titik-titik radiance
    ke setiap bangunan menggunakan Inverse Distance Weighting (IDW).

    Jika hanya ada satu titik radiance, semua bangunan mendapatkan nilai sama.
    Jika ada banyak titik, interpolasi IDW (k-nearest, w = 1/d²).
    """
    log.info('[TASK1.1a] Penugasan irradiance ke bangunan (IDW)...')

    bldg_geo = buildings.to_crs(CFG['geo_crs'])
    bldg_pts = np.column_stack([bldg_geo.geometry.centroid.x.values,
                                  bldg_geo.geometry.centroid.y.values])
    rad_pts  = rad_df[['lon', 'lat']].values

    tree = cKDTree(rad_pts)
    k    = min(4, len(rad_pts))
    dist, idx = tree.query(bldg_pts, k=k)

    for col in ['GHI', 'DNI', 'DHI']:
        if col not in rad_df.columns:
            continue
        vals = rad_df[col].values[idx]          # shape (n_bldg, k)
        if k == 1:
            buildings[col] = vals.ravel()
        else:
            w   = 1.0 / (dist + 1e-10)**2       # IDW weight
            w  /= w.sum(axis=1, keepdims=True)
            buildings[col] = (vals * w).sum(axis=1)

    if 'GHI' in buildings.columns:
        log.info(f'       GHI mean: {buildings["GHI"].mean():.1f} kWh/m²/yr '
                 f'| range [{buildings["GHI"].min():.0f}, {buildings["GHI"].max():.0f}]')
    return buildings


def assign_cloud_fraction(buildings: gpd.GeoDataFrame,
                           cf_df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Tetapkan Cloud Fraction ke bangunan via nearest-neighbour."""
    log.info('[TASK1.1b] Penugasan Cloud Fraction...')

    if cf_df.empty or 'CF' not in cf_df.columns:
        buildings['CF'] = 0.30
        log.warning('       CF data kosong → default CF = 0.30')
        return buildings

    bldg_geo = buildings.to_crs(CFG['geo_crs'])
    bldg_pts = np.column_stack([bldg_geo.geometry.centroid.x.values,
                                  bldg_geo.geometry.centroid.y.values])
    cf_pts   = cf_df[['lon', 'lat']].values
    tree     = cKDTree(cf_pts)
    _, idx   = tree.query(bldg_pts, k=1)
    buildings['CF'] = cf_df['CF'].values[idx]

    log.info(f'       CF mean: {buildings["CF"].mean():.3f} '
             f'| range [{buildings["CF"].min():.3f}, {buildings["CF"].max():.3f}]')
    return buildings


# ═══════════════════════════════════════════════════════════════════════════════
# §6  ML — SMOOTHING TINGGI BANGUNAN (RandomForest)   [Bonus §8.1]
# ═══════════════════════════════════════════════════════════════════════════════

def smooth_heights_with_rf(buildings: gpd.GeoDataFrame) -> tuple:
    """
    Gunakan RandomForest untuk menghaluskan / mengisi nilai tinggi bangunan
    yang hilang atau tidak realistis, menggunakan fitur spasial kontekstual.

    Fitur:
      x, y (centroid), area_m2, perimeter_m, compactness,
      rata-rata tinggi tetangga dalam 100 m, jumlah tetangga dalam 100 m

    Validasi: K-Fold CV dengan laporan R² dan MAE.
    Referensi pendekatan: Geiß et al. (2015), Li et al. (2020)
    """
    log.info('[ML] RandomForest building-height smoothing...')

    bldg_p = (buildings.to_crs(CFG['proj_crs'])
              if buildings.crs.to_epsg() != int(CFG['proj_crs'].split(':')[1])
              else buildings)

    cx = bldg_p.geometry.centroid.x.values
    cy = bldg_p.geometry.centroid.y.values
    centroids   = np.column_stack([cx, cy])
    heights_arr = buildings['height'].values.copy().astype(float)

    # Fitur kontekstual spasial
    tree = cKDTree(centroids)
    nbr_h_mean = np.zeros(len(bldg_p))
    nbr_count  = np.zeros(len(bldg_p))
    for i in range(len(bldg_p)):
        nb = [j for j in tree.query_ball_point(centroids[i], 100) if j != i]
        if nb:
            nbr_h_mean[i] = float(np.mean(heights_arr[nb]))
            nbr_count[i]  = len(nb)
        else:
            nbr_h_mean[i] = heights_arr[i]
            nbr_count[i]  = 0

    X = np.column_stack([
        cx, cy,
        buildings['area_m2'].values,
        buildings['perimeter_m'].values,
        buildings['compactness'].values,
        nbr_h_mean,
        nbr_count,
    ])
    y = heights_arr

    # Pisahkan data valid (tinggi tidak default)
    default_h  = np.median(y)
    valid_mask = (y > 0) & (y != 6.0) & (y < 300)   # filter outlier kasar
    n_valid    = int(valid_mask.sum())

    metrics = {'r2': None, 'mae': None, 'n_train': n_valid}

    if n_valid < CFG['min_valid_h']:
        log.warning(f'       Hanya {n_valid} data tinggi valid — RF dilewati')
        buildings['height_smooth'] = y
        return buildings, metrics

    X_tr, y_tr = X[valid_mask], y[valid_mask]

    rf = RandomForestRegressor(
        n_estimators=CFG['rf_n'],
        max_features='sqrt',
        min_samples_leaf=3,
        random_state=CFG['rf_seed'],
        n_jobs=-1,
    )

    # K-Fold Cross Validation
    kf   = KFold(n_splits=CFG['cv_k'], shuffle=True, random_state=CFG['rf_seed'])
    y_cv = cross_val_predict(rf, X_tr, y_tr, cv=kf)
    r2   = float(r2_score(y_tr, y_cv))
    mae  = float(mean_absolute_error(y_tr, y_cv))
    metrics.update(r2=r2, mae=mae)
    log.info(f'       RF Height Model ({CFG["cv_k"]}-Fold CV) → '
             f'R² = {r2:.4f} | MAE = {mae:.2f} m | n_train = {n_valid}')

    # Latih ulang pada seluruh data valid
    rf.fit(X_tr, y_tr)
    y_pred   = rf.predict(X)
    y_smooth = y.copy()

    # Isi celah (nilai default/nol/negatif)
    mask_fill = ~valid_mask
    y_smooth[mask_fill] = y_pred[mask_fill]

    # Feature importance log
    feat_names = ['x','y','area','perimeter','compactness','nbr_h_mean','nbr_count']
    fi         = dict(zip(feat_names, rf.feature_importances_))
    top2       = sorted(fi, key=fi.get, reverse=True)[:2]
    log.info(f'       Fitur terpenting: {top2[0]}={fi[top2[0]]:.3f}, '
             f'{top2[1]}={fi[top2[1]]:.3f}')

    buildings['height_smooth'] = y_smooth
    buildings['height']        = y_smooth   # update kolom height utama
    log.info(f'       Height diperbarui: mean={y_smooth.mean():.1f} m | '
             f'max={y_smooth.max():.1f} m')
    return buildings, metrics


# ═══════════════════════════════════════════════════════════════════════════════
# §7  AREA ATAP EFEKTIF + POA IRRADIANCE  (Task 1.1)
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_roof_area(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Task 1.1 — Area atap efektif dengan faktor kegunaan atap (ρ_roof = 0.65).
    ρ_roof memperhitungkan: peralatan HVAC, tangga, parapet, struktur atap,
    dan ruang antar panel untuk perawatan.
    """
    buildings['A_gross'] = buildings['area_m2']
    buildings['A_eff']   = buildings['area_m2'] * CFG['rho_roof']
    log.info(f'[TASK1.1] Area atap efektif: '
             f'total = {buildings["A_eff"].sum()/1e4:.2f} ha | '
             f'mean per bangunan = {buildings["A_eff"].mean():.1f} m²')
    return buildings


def calculate_poa_irradiance(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Task 1.1 — Plane-of-Array (POA) irradiance untuk panel miring (β = tilt_deg).

    Model Hay-Davies (isotropic diffuse):
      POA_beam      = DNI × cos(θ_incidence)
      POA_diffuse   = DHI × (1 + cos(β)) / 2
      POA_reflected = GHI × ρ × (1 − cos(β)) / 2

    Untuk atap datar tropis (β = 10°, azimuth menghadap equator):
      θ_incidence ≈ β (perkiraan yang baik untuk rata-rata tahunan).

    Referensi: Hay & Davies (1980), Duffie & Beckman (2013).
    """
    log.info('[TASK1.1] Menghitung POA irradiance (Hay-Davies model)...')

    beta_rad  = math.radians(CFG['tilt_deg'])
    cos_beta  = math.cos(beta_rad)
    cos_inc   = math.cos(math.radians(CFG['tilt_deg']))  # approx for equator-facing

    GHI = buildings['GHI']
    DHI = buildings.get('DHI', GHI * 0.25)
    DNI = buildings.get('DNI', (GHI - DHI) /
                        max(math.cos(math.radians(30)), 0.01))

    POA_beam      = DNI * cos_inc
    POA_diffuse   = DHI * (1 + cos_beta) / 2
    POA_reflected = GHI * CFG['albedo'] * (1 - cos_beta) / 2

    buildings['POA_annual'] = (POA_beam + POA_diffuse + POA_reflected).clip(lower=0)
    buildings['tilt_factor'] = buildings['POA_annual'] / GHI.replace(0, np.nan)

    log.info(f'       POA mean: {buildings["POA_annual"].mean():.1f} kWh/m²/yr | '
             f'tilt gain: {(buildings["tilt_factor"].mean()-1)*100:+.1f}%')
    return buildings


# ═══════════════════════════════════════════════════════════════════════════════
# §8  MODEL SHADING (Task 1.1)
# ═══════════════════════════════════════════════════════════════════════════════

def _building_shading(buildings: gpd.GeoDataFrame) -> np.ndarray:
    """
    Shading dari bangunan tetangga yang lebih tinggi (self-shading / mutual shading).

    Pendekatan:
      Untuk tiap bangunan B, cari tetangga dalam radius R.
      Bagi setiap tetangga N yang lebih tinggi:
        - Panjang bayangan horizontal: L = (h_N - h_B) / tan(solar_elevation)
        - Luas bayangan yang jatuh di atap B diperkirakan proporsional
          terhadap L × √A_N (lebar efektif bangunan N)
      SF_bldg = min(total_shadow_area / A_B, 0.80)

    Referensi: Freitas et al. (2015), Hofierka & Kaňuk (2009).
    """
    log.info('       Menghitung building mutual shading...')
    bldg_p = (buildings.to_crs(CFG['proj_crs'])
              if buildings.crs.to_epsg() != int(CFG['proj_crs'].split(':')[1])
              else buildings)

    cx = bldg_p.geometry.centroid.x.values
    cy = bldg_p.geometry.centroid.y.values
    centroids   = np.column_stack([cx, cy])
    heights_arr = buildings['height'].values
    areas_arr   = buildings['A_eff'].values
    tan_elv     = math.tan(math.radians(CFG['solar_elv']))

    tree    = cKDTree(centroids)
    R       = CFG['shadow_r_m']
    sf_bldg = np.zeros(len(bldg_p))

    for i in range(len(bldg_p)):
        nbs  = [j for j in tree.query_ball_point(centroids[i], R)
                if j != i and heights_arr[j] > heights_arr[i]]
        if not nbs:
            continue
        shadow_total = 0.0
        for j in nbs:
            dh            = heights_arr[j] - heights_arr[i]
            shadow_len    = dh / max(tan_elv, 0.01)
            shadow_width  = math.sqrt(max(areas_arr[j], 1.0))
            shadow_area   = shadow_len * shadow_width
            # Kurangi dengan fraksi overlap berdasarkan jarak
            dist_ij       = math.hypot(cx[i]-cx[j], cy[i]-cy[j])
            overlap_f     = max(0, 1 - dist_ij / (R + 1e-6))
            shadow_total += shadow_area * overlap_f

        sf_bldg[i] = min(shadow_total / max(areas_arr[i], 1), 0.80)

    return sf_bldg


def calculate_shading_factor(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Task 1.1 — Shading factor gabungan SF_total ∈ [0, 1].

    Komponen:
      SF_cloud  = CF × 0.50  (awan menutup hingga 50% irradiance)
      SF_bldg   = fraksi atap yang tertutup bayangan bangunan sekitar
      SF_total  = (1 - w_cf × SF_cloud) × (1 - w_bldg × SF_bldg)

    SF_total mendekati 0 = banyak shading (buruk)
    SF_total mendekati 1 = sedikit shading (baik)
    """
    log.info('[TASK1.1] Model shading...')

    # ── Shading dari awan ────────────────────────────────────────────────────
    buildings['SF_cloud'] = (buildings['CF'] * 0.50).clip(0, 1)

    # ── Shading dari bangunan tetangga ────────────────────────────────────────
    buildings['SF_bldg']  = _building_shading(buildings)

    # ── Gabungan ─────────────────────────────────────────────────────────────
    w_cf   = CFG['cf_weight']
    w_bldg = 1.0 - w_cf
    buildings['SF_total'] = (
        (1 - w_cf   * buildings['SF_cloud']) *
        (1 - w_bldg * buildings['SF_bldg'])
    ).clip(0, 1)

    log.info(f'       SF_cloud mean: {buildings["SF_cloud"].mean():.3f}')
    log.info(f'       SF_bldg  mean: {buildings["SF_bldg"].mean():.3f}')
    log.info(f'       SF_total mean: {buildings["SF_total"].mean():.3f}')
    return buildings


# ═══════════════════════════════════════════════════════════════════════════════
# §9  MODEL ENERGI PV  (Task 1.2)
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_pv_energy(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Task 1.2 — Kapasitas terpasang (kWp) dan produksi energi tahunan (kWh/yr).

    Rumus:
      kWp         = A_eff × η_panel                           [kW_peak]
      T_cell      = T_amb + (NOCT − 20) × (GHI_avg / 800)    [°C]
      T_corr      = 1 + T_coeff × (T_cell − 25)              [fraksi]
      E_annual    = kWp × POA_annual × PR × SF_total × T_corr [kWh/yr]
      yield_spec  = E_annual / kWp                            [kWh/kWp/yr]

    Referensi:
      - IEC 61724-1 (performance monitoring)
      - PVGIS methodology report (EU Joint Research Centre)
      - Duffie & Beckman, Solar Engineering of Thermal Processes (2013)
    """
    log.info('[TASK1.2] Menghitung kapasitas (kWp) dan energi (kWh/yr)...')

    η     = CFG['eta']
    PR    = CFG['PR']
    NOCT  = CFG['NOCT']
    Tc    = CFG['T_coeff']
    T_amb = CFG['T_amb']

    # Kapasitas terpasang
    buildings['kWp'] = buildings['A_eff'] * η

    # Temperatur sel rata-rata (menggunakan rata-rata irradiance tahunan)
    ghi_mean_wm2        = buildings['GHI'] * 1000 / 8760   # kWh/yr → rata-rata W/m²
    buildings['T_cell'] = T_amb + (NOCT - 20) * ghi_mean_wm2 / 800

    # Faktor koreksi temperatur
    buildings['T_corr'] = (1 + Tc * (buildings['T_cell'] - 25)).clip(0.70, 1.05)

    # Produksi energi tahunan
    buildings['E_annual_kWh'] = (
        buildings['kWp'] *
        buildings['POA_annual'] *
        PR *
        buildings['SF_total'] *
        buildings['T_corr']
    ).clip(lower=0)

    # Specific yield
    buildings['yield_spec'] = (
        buildings['E_annual_kWh'] /
        buildings['kWp'].replace(0, np.nan)
    ).fillna(0)

    log.info(f'       kWp    mean: {buildings["kWp"].mean():.2f} | '
             f'total: {buildings["kWp"].sum()/1000:.1f} MWp')
    log.info(f'       E/yr   mean: {buildings["E_annual_kWh"].mean():.0f} kWh | '
             f'total: {buildings["E_annual_kWh"].sum()/1e6:.2f} GWh')
    log.info(f'       T_cell mean: {buildings["T_cell"].mean():.1f} °C | '
             f'T_corr mean: {buildings["T_corr"].mean():.3f}')
    return buildings


# ═══════════════════════════════════════════════════════════════════════════════
# §10  NORMALISASI & KLASIFIKASI  (Task 1.3)
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_index(buildings: gpd.GeoDataFrame,
                    col: str = 'E_annual_kWh') -> gpd.GeoDataFrame:
    """
    Task 1.3 — Min-Max scaling produksi energi tahunan ke [0, 1].

    solar_pv_index = (E - E_min) / (E_max - E_min)

    Menghasilkan Solar PV Potential Index per bangunan.
    """
    log.info(f'[TASK1.3] Normalisasi indeks dari kolom: {col}')

    vals   = buildings[col].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    buildings['solar_pv_index'] = scaler.fit_transform(vals).ravel()

    log.info(f'       Index: min={buildings["solar_pv_index"].min():.4f} | '
             f'mean={buildings["solar_pv_index"].mean():.4f} | '
             f'max={buildings["solar_pv_index"].max():.4f}')
    return buildings


def classify_index(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Task 1.3 — Klasifikasi Solar PV Potential Index menjadi 3 kelas:
      Rendah  : [0.00, thr_low)   → potensi rendah
      Sedang  : [thr_low, thr_hi) → potensi sedang
      Tinggi  : [thr_hi, 1.00]    → potensi tinggi, prioritas pemasangan
    """
    lo, hi = CFG['thr_low'], CFG['thr_high']
    bins   = [-np.inf, lo, hi, np.inf]
    labels = ['Rendah', 'Sedang', 'Tinggi']
    buildings['pv_class'] = pd.cut(
        buildings['solar_pv_index'], bins=bins, labels=labels
    )
    dist = buildings['pv_class'].value_counts()
    log.info(f'       Distribusi kelas: {dict(dist)}')

    # Ranking (1 = tertinggi)
    buildings['pv_rank'] = (buildings['solar_pv_index']
                            .rank(ascending=False, method='min')
                            .astype(int))
    return buildings


# ═══════════════════════════════════════════════════════════════════════════════
# §11  VISUALISASI
# ═══════════════════════════════════════════════════════════════════════════════

def _format_large(x, pos=None):
    """Formatter angka besar untuk sumbu matplotlib."""
    if abs(x) >= 1e6:  return f'{x/1e6:.1f}M'
    if abs(x) >= 1e3:  return f'{x/1e3:.1f}k'
    return str(int(x))


def plot_static_map(buildings: gpd.GeoDataFrame, output_path: str) -> None:
    """
    Peta kartografi statis choropleth Solar PV Potential Index.
    Elemen: judul, colorbar, skala, arah utara, sumber data, basemap.
    """
    log.info('[VIZ] Membuat peta statis choropleth...')

    bldg_wm = buildings.to_crs('EPSG:3857')   # Web Mercator untuk basemap
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # Basemap
    if _opt['ctx']:
        try:
            _opt['ctx'].add_basemap(
                ax, source=_opt['ctx'].providers.CartoDB.Positron,
                zoom='auto', alpha=0.55, crs='EPSG:3857'
            )
        except Exception:
            pass

    # Choropleth
    bldg_wm.plot(
        column='solar_pv_index', cmap=PV_CMAP,
        linewidth=0.2, edgecolor='#555555',
        legend=False, ax=ax, vmin=0, vmax=1
    )

    # Colorbar
    sm   = plt.cm.ScalarMappable(cmap=PV_CMAP,
                                  norm=mcolors.Normalize(vmin=0, vmax=1))
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, aspect=30)
    cbar.set_label('Solar PV Potential Index', fontsize=11, fontweight='bold')
    cbar.set_ticks([0, CFG['thr_low'], CFG['thr_high'], 1.0])
    cbar.set_ticklabels(
        [f'0.0\n(Rendah)', f'{CFG["thr_low"]}', f'{CFG["thr_high"]}', '1.0\n(Tinggi)']
    )

    # North arrow
    x0, x1, y0, y1 = *ax.get_xlim(), *ax.get_ylim()
    nx = x0 + (x1-x0)*0.96
    ny = y0 + (y1-y0)*0.92
    ax.annotate(
        '', xy=(nx, ny), xytext=(nx, ny-(y1-y0)*0.05),
        arrowprops=dict(arrowstyle='->', color='black', lw=2)
    )
    ax.text(nx, ny+(y1-y0)*0.008, 'N', ha='center', va='bottom',
            fontsize=12, fontweight='bold')

    # Scalebar manual ~100 m
    sb_len  = 100                          # meter
    sb_x    = x0 + (x1-x0)*0.05
    sb_y    = y0 + (y1-y0)*0.04
    ax.plot([sb_x, sb_x+sb_len], [sb_y, sb_y], 'k-', linewidth=4)
    ax.text(sb_x + sb_len/2, sb_y+(y1-y0)*0.012, '100 m',
            ha='center', fontsize=8)

    # Judul & metadata
    n_high = int((buildings['pv_class'] == 'Tinggi').sum())
    ax.set_title(
        'Solar PV Potential Index — Per Bangunan\n'
        f'Task 1 Output | {len(buildings):,} bangunan | '
        f'{n_high} prioritas Tinggi',
        fontsize=14, fontweight='bold', pad=12
    )
    ax.text(
        0.005, 0.005,
        f'Sumber: Building Footprint + Radiance Satelit + Cloud Fraction (MODIS)\n'
        f'CRS: Web Mercator (EPSG:3857) | Dibuat: {datetime.now():%Y-%m-%d}',
        transform=ax.transAxes, fontsize=7, color='#555', va='bottom'
    )
    ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    log.info(f'       Peta statis tersimpan → {output_path}')


def plot_interactive_map(buildings: gpd.GeoDataFrame, output_path: str) -> None:
    """
    Peta interaktif Folium dengan tooltip per bangunan dan layer control.
    """
    if _opt['folium'] is None:
        log.warning('[VIZ] folium tidak terinstall → skip peta interaktif')
        return

    log.info('[VIZ] Membuat peta interaktif Folium...')
    folium = _opt['folium']
    plugins = getattr(folium, 'plugins', None)

    bldg_geo = buildings.to_crs('EPSG:4326').copy()
    bldg_geo['pv_class'] = bldg_geo['pv_class'].astype(str)
    centre   = [bldg_geo.geometry.centroid.y.mean(),
                 bldg_geo.geometry.centroid.x.mean()]

    m = folium.Map(location=centre, zoom_start=16, tiles='CartoDB positron')

    # Fungsi warna berdasarkan index
    def _color(idx_val):
        if idx_val >= CFG['thr_high']: return CLASS_COLORS['Tinggi']
        if idx_val >= CFG['thr_low']:  return CLASS_COLORS['Sedang']
        return CLASS_COLORS['Rendah']

    # GeoJSON layer utama
    geojson_layer = folium.GeoJson(
        bldg_geo[['geometry', 'bldg_id', 'solar_pv_index', 'pv_class',
                   'kWp', 'E_annual_kWh', 'area_m2', 'height',
                   'GHI', 'SF_total', 'pv_rank']].to_json(),
        name='Solar PV Potential Index',
        style_function=lambda feat: {
            'fillColor' : _color(feat['properties']['solar_pv_index']),
            'color'     : '#444',
            'weight'    : 0.4,
            'fillOpacity': 0.75,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['bldg_id', 'solar_pv_index', 'pv_class',
                    'kWp', 'E_annual_kWh', 'area_m2', 'height', 'pv_rank'],
            aliases=['Bangunan ID', 'PV Index', 'Kelas',
                     'Kapasitas (kWp)', 'Energi/thn (kWh)', 'Luas (m²)',
                     'Tinggi (m)', 'Ranking'],
            localize=True, sticky=True,
            style='font-size:11px;background:rgba(255,255,255,0.9);'
                  'border-radius:4px;'
        ),
        highlight_function=lambda _: {'weight': 2, 'color': 'black'}
    )
    geojson_layer.add_to(m)

    # Layer bangunan TOP 20 prioritas
    top20 = bldg_geo.nsmallest(20, 'pv_rank')
    for _, row in top20.iterrows():
        c = row.geometry.centroid
        folium.Marker(
            location=[c.y, c.x],
            icon=folium.Icon(color='green', icon='star', prefix='fa'),
            tooltip=f"Rank #{row['pv_rank']} | Index {row['solar_pv_index']:.3f}"
        ).add_to(m)

    # Legend HTML
    legend_html = f"""
    <div style="position:fixed;bottom:25px;right:15px;z-index:9999;
                background:white;padding:12px 16px;border-radius:8px;
                border:1px solid #ccc;font-family:sans-serif;font-size:12px;
                box-shadow:2px 2px 6px rgba(0,0,0,0.15);">
      <b>Solar PV Potential</b><br><br>
      <span style="display:inline-block;width:14px;height:14px;
        background:{CLASS_COLORS['Tinggi']};margin-right:6px;"></span>Tinggi (&ge;{CFG['thr_high']})<br>
      <span style="display:inline-block;width:14px;height:14px;
        background:{CLASS_COLORS['Sedang']};margin-right:6px;"></span>Sedang ({CFG['thr_low']}–{CFG['thr_high']})<br>
      <span style="display:inline-block;width:14px;height:14px;
        background:{CLASS_COLORS['Rendah']};margin-right:6px;"></span>Rendah (&lt;{CFG['thr_low']})<br>
      <hr style="margin:6px 0"><span style="font-size:10px;color:#666">
        ★ = Top 20 Prioritas</span>
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl().add_to(m)

    m.save(output_path)
    log.info(f'       Peta interaktif tersimpan → {output_path}')


def plot_analysis_charts(buildings: gpd.GeoDataFrame,
                          ml_metrics: dict = None) -> None:
    """
    Panel analisis 4-in-1:
      A. Distribusi Solar PV Potential Index (histogram + KDE)
      B. Box plot energi per kelas PV
      C. Scatter area atap vs energi (warna = index)
      D. Distribusi komponen shading factor
    """
    log.info('[VIZ] Membuat grafik analisis (4-panel)...')

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Task 1 — Analisis Solar PV Potential Index',
                  fontsize=15, fontweight='bold', y=0.99)

    # ── Panel A: Distribusi indeks ────────────────────────────────────────────
    ax = axes[0, 0]
    sns.histplot(buildings['solar_pv_index'], bins=35, kde=True,
                  color='#1a9850', ax=ax, edgecolor='white',
                  linewidth=0.5, alpha=0.75)
    ax.axvline(CFG['thr_low'],  color='#fc8d59', ls='--', lw=1.8, label='Batas kelas')
    ax.axvline(CFG['thr_high'], color='#d73027', ls='--', lw=1.8)
    ax.set_xlabel('Solar PV Potential Index', fontsize=10)
    ax.set_ylabel('Jumlah Bangunan', fontsize=10)
    ax.set_title('A. Distribusi Solar PV Potential Index', fontweight='bold')
    ax.legend(fontsize=8)
    for i, (cls, clr) in enumerate(CLASS_COLORS.items()):
        n   = (buildings['pv_class'] == cls).sum()
        pct = n / len(buildings) * 100
        ax.text(0.98, 0.96 - i*0.10,
                f'{cls}: {n} bangunan ({pct:.1f}%)',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=8, color=clr, fontweight='bold')

    # ── Panel B: Box plot energi per kelas ────────────────────────────────────
    ax = axes[0, 1]
    bldg_cls = buildings.copy()
    bldg_cls['pv_class'] = bldg_cls['pv_class'].astype(str)
    sns.boxplot(
        data=bldg_cls, x='pv_class', y='E_annual_kWh',
        order=CLASS_ORDER,
        palette={k: v for k, v in CLASS_COLORS.items()},
        showfliers=False, linewidth=1.3, ax=ax
    )
    ax.set_xlabel('Kelas PV Potential', fontsize=10)
    ax.set_ylabel('Produksi Energi (kWh/yr)', fontsize=10)
    ax.set_title('B. Distribusi Energi per Kelas', fontweight='bold')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_format_large))
    # Tambah median label
    for i, cls in enumerate(CLASS_ORDER):
        med = buildings[buildings['pv_class'] == cls]['E_annual_kWh'].median()
        ax.text(i, med, f'{_format_large(med)}', ha='center', va='bottom',
                fontsize=8, fontweight='bold', color='black')

    # ── Panel C: Scatter luas atap vs energi ─────────────────────────────────
    ax = axes[1, 0]
    sc = ax.scatter(
        buildings['A_eff'], buildings['E_annual_kWh'],
        c=buildings['solar_pv_index'], cmap=PV_CMAP,
        s=18, alpha=0.60, edgecolors='none', vmin=0, vmax=1
    )
    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label('PV Index', fontsize=8)
    ax.set_xlabel('Area Atap Efektif (m²)', fontsize=10)
    ax.set_ylabel('Produksi Energi (kWh/yr)', fontsize=10)
    ax.set_title('C. Luas Atap vs Produksi Energi', fontweight='bold')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_format_large))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_format_large))
    if len(buildings) > 2:
        r, p = pearsonr(buildings['A_eff'], buildings['E_annual_kWh'])
        ax.text(0.05, 0.95, f'r = {r:.3f} (p {"<0.001" if p<0.001 else f"={p:.3f}"})',
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round', fc='white', alpha=0.7))

    # ── Panel D: Komponen shading ─────────────────────────────────────────────
    ax = axes[1, 1]
    shade_data = {
        'Cloud Shading (SF_cloud)': ('SF_cloud', '#74c476'),
        'Building Shading (SF_bldg)': ('SF_bldg', '#fd8d3c'),
        'Total Shading Factor': ('SF_total', '#9e9ac8'),
    }
    for lbl, (col, clr) in shade_data.items():
        if col in buildings.columns:
            sns.kdeplot(buildings[col], label=lbl, color=clr,
                         ax=ax, fill=True, alpha=0.25, linewidth=1.8)
    ax.set_xlabel('Nilai Faktor', fontsize=10)
    ax.set_ylabel('Densitas', fontsize=10)
    ax.set_title('D. Distribusi Komponen Shading Factor', fontweight='bold')
    ax.legend(fontsize=8)
    # Rata-rata vertikal
    for col, clr in [('SF_cloud','#74c476'),('SF_total','#9e9ac8')]:
        if col in buildings.columns:
            ax.axvline(buildings[col].mean(), color=clr, ls=':', lw=1.5, alpha=0.8)

    # Anotasi metrik ML
    if ml_metrics and ml_metrics.get('r2') is not None:
        fig.text(
            0.01, 0.005,
            f'[ML] RF Height Smoothing ({CFG["cv_k"]}-Fold CV) → '
            f'R² = {ml_metrics["r2"]:.4f} | MAE = {ml_metrics["mae"]:.2f} m | '
            f'n_train = {ml_metrics["n_train"]}',
            fontsize=8, color='#666'
        )

    plt.tight_layout(rect=[0, 0.025, 1, 0.985])
    path = OUT + 'figures/task1_analysis_charts.png'
    plt.savefig(path)
    plt.close()
    log.info(f'       Grafik analisis tersimpan → {path}')


def plot_top_buildings(buildings: gpd.GeoDataFrame, top_n: int = 20) -> None:
    """Bar chart Top-N bangunan prioritas dengan detail atribut."""
    log.info(f'[VIZ] Top {top_n} bangunan prioritas...')

    top = buildings.nsmallest(top_n, 'pv_rank').copy()
    top['label'] = [f"Bldg-{int(r['bldg_id'])}" for _, r in top.iterrows()]

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle(f'Top {top_n} Bangunan Prioritas Solar PV',
                  fontsize=14, fontweight='bold')

    metrics = [
        ('solar_pv_index', 'PV Index', 'Indeks [0–1]', PV_CMAP),
        ('kWp',            'Kapasitas', 'kWp',          'Blues'),
        ('E_annual_kWh',   'Energi/Tahun', 'kWh/yr',   'Greens'),
    ]
    for ax, (col, title, xlabel, cmap) in zip(axes, metrics):
        vals = top[col].values
        norm = mcolors.Normalize(vals.min(), vals.max())
        cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
        colors_bar = [cmap_obj(norm(v)) for v in vals]
        ax.barh(top['label'], vals, color=colors_bar, edgecolor='white', lw=0.5)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.invert_yaxis()
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(_format_large))
        for i, v in enumerate(vals):
            ax.text(v*1.01, i, f'{_format_large(v)}', va='center', fontsize=7)

    plt.tight_layout()
    path = OUT + f'figures/top{top_n}_buildings.png'
    plt.savefig(path)
    plt.close()
    log.info(f'       Top-{top_n} chart tersimpan → {path}')


# ═══════════════════════════════════════════════════════════════════════════════
# §12  EKSPOR HASIL
# ═══════════════════════════════════════════════════════════════════════════════

def export_results(buildings: gpd.GeoDataFrame) -> None:
    """
    Ekspor hasil ke GeoJSON, CSV, dan ringkasan JSON.
    Output kolom mencakup semua atribut task 1.
    """
    log.info('[EXPORT] Mengekspor hasil...')

    keep_cols = [
        'bldg_id', 'geometry',
        'area_m2', 'height', 'height_smooth', 'A_gross', 'A_eff',
        'GHI', 'DNI', 'DHI', 'POA_annual', 'tilt_factor',
        'CF', 'SF_cloud', 'SF_bldg', 'SF_total',
        'T_cell', 'T_corr',
        'kWp', 'E_annual_kWh', 'yield_spec',
        'solar_pv_index', 'pv_class', 'pv_rank',
    ]
    out_cols = [c for c in keep_cols if c in buildings.columns]
    out_gdf  = buildings[out_cols].copy()
    out_gdf['pv_class'] = out_gdf['pv_class'].astype(str)

    # ── GeoJSON (EPSG:4326) ──────────────────────────────────────────────────
    geo_out = out_gdf.to_crs('EPSG:4326')
    gjson_path = OUT + 'data/solar_pv_index.geojson'
    geo_out.to_file(gjson_path, driver='GeoJSON')
    log.info(f'       → {gjson_path}')

    # ── CSV ringkasan (tanpa geometry) ───────────────────────────────────────
    csv_path = OUT + 'data/solar_pv_summary.csv'
    geo_out.drop(columns='geometry').to_csv(csv_path, index=False)
    log.info(f'       → {csv_path}')

    # ── Top-20 ranking ───────────────────────────────────────────────────────
    top_path = OUT + 'data/top20_priority.csv'
    (geo_out.drop(columns='geometry')
     .nsmallest(20, 'pv_rank')
     .to_csv(top_path, index=False))
    log.info(f'       → {top_path}')

    # ── Ringkasan JSON ────────────────────────────────────────────────────────
    summary = {
        'n_buildings'         : int(len(buildings)),
        'total_A_eff_ha'      : round(float(buildings['A_eff'].sum())/1e4, 3),
        'total_kWp'           : round(float(buildings['kWp'].sum()), 2),
        'total_MWp'           : round(float(buildings['kWp'].sum())/1000, 4),
        'total_GWh_per_year'  : round(float(buildings['E_annual_kWh'].sum())/1e6, 4),
        'mean_pv_index'       : round(float(buildings['solar_pv_index'].mean()), 4),
        'std_pv_index'        : round(float(buildings['solar_pv_index'].std()), 4),
        'count_Tinggi'        : int((buildings['pv_class']=='Tinggi').sum()),
        'count_Sedang'        : int((buildings['pv_class']=='Sedang').sum()),
        'count_Rendah'        : int((buildings['pv_class']=='Rendah').sum()),
        'mean_GHI_kWh_m2_yr'  : round(float(buildings['GHI'].mean()), 2),
        'mean_POA_kWh_m2_yr'  : round(float(buildings['POA_annual'].mean()), 2),
        'mean_CF'             : round(float(buildings['CF'].mean()), 4),
        'mean_SF_total'       : round(float(buildings['SF_total'].mean()), 4),
        'mean_T_cell_C'       : round(float(buildings['T_cell'].mean()), 2),
        'mean_T_corr'         : round(float(buildings['T_corr'].mean()), 4),
        'pv_index_thresholds' : {'low': CFG['thr_low'], 'high': CFG['thr_high']},
        'parameters'          : {
            'eta_panel'   : CFG['eta'],
            'PR'          : CFG['PR'],
            'rho_roof'    : CFG['rho_roof'],
            'tilt_deg'    : CFG['tilt_deg'],
            'NOCT'        : CFG['NOCT'],
            'T_coeff'     : CFG['T_coeff'],
            'T_amb'       : CFG['T_amb'],
        },
        'generated_at'        : datetime.now().isoformat(),
    }

    json_path = OUT + 'data/task1_summary.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ── Log ringkasan ke konsol ────────────────────────────────────────────
    log.info('\n' + '═'*60)
    log.info('  RINGKASAN HASIL TASK 1')
    log.info('═'*60)
    log.info(f'  Bangunan dianalisis : {summary["n_buildings"]:,}')
    log.info(f'  Total area efektif  : {summary["total_A_eff_ha"]:.2f} ha')
    log.info(f'  Total kapasitas     : {summary["total_MWp"]:.2f} MWp')
    log.info(f'  Total produksi      : {summary["total_GWh_per_year"]:.2f} GWh/tahun')
    log.info(f'  Kelas Tinggi        : {summary["count_Tinggi"]:,} bangunan '
             f'({summary["count_Tinggi"]/summary["n_buildings"]*100:.1f}%)')
    log.info(f'  Kelas Sedang        : {summary["count_Sedang"]:,} bangunan')
    log.info(f'  Kelas Rendah        : {summary["count_Rendah"]:,} bangunan')
    log.info(f'  Mean PV Index       : {summary["mean_pv_index"]:.4f} '
             f'± {summary["std_pv_index"]:.4f}')
    log.info('═'*60)
    log.info(f'  JSON summary → {json_path}')


# ═══════════════════════════════════════════════════════════════════════════════
# §13  PIPELINE UTAMA
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> gpd.GeoDataFrame:
    """
    Pipeline end-to-end Task 1.
    Urutan: Load → Preprocess → ML → Irradiance → Roof → POA
            → Shading → PV Energy → Normalize → Classify → Viz → Export
    """
    t_start = datetime.now()
    log.info('=' * 68)
    log.info('  TASK 1 — SOLAR PV POTENTIAL INDEX')
    log.info('  Technical Assessment · Geospatial Data Engineer')
    log.info(f'  Dimulai: {t_start:%Y-%m-%d %H:%M:%S}')
    log.info('=' * 68)

    # ── [1] Load data ─────────────────────────────────────────────────────────
    buildings = load_buildings(CFG['bldg_path'])
    rad_df    = load_radiance(CFG['rad_path'])
    cf_df     = load_cloud_fraction(CFG['cf_path'], buildings)

    # ── [2] Preprocess ────────────────────────────────────────────────────────
    buildings = preprocess_buildings(buildings)

    # ── [3] ML: RF height smoothing ───────────────────────────────────────────
    buildings, ml_metrics = smooth_heights_with_rf(buildings)

    # ── [4] Penugasan irradiance & cloud fraction ─────────────────────────────
    buildings = assign_irradiance(buildings, rad_df)
    buildings = assign_cloud_fraction(buildings, cf_df)

    # ── [5] Area atap efektif ─────────────────────────────────────────────────
    buildings = calculate_roof_area(buildings)

    # ── [6] Task 1.1 — POA irradiance ────────────────────────────────────────
    buildings = calculate_poa_irradiance(buildings)

    # ── [7] Task 1.1 — Shading model ─────────────────────────────────────────
    buildings = calculate_shading_factor(buildings)

    # ── [8] Task 1.2 — PV energy ─────────────────────────────────────────────
    buildings = calculate_pv_energy(buildings)

    # ── [9] Task 1.3 — Normalisasi & klasifikasi ─────────────────────────────
    buildings = normalize_index(buildings, col='E_annual_kWh')
    buildings = classify_index(buildings)

    # ── [10] Visualisasi ──────────────────────────────────────────────────────
    plot_static_map(
        buildings,
        output_path=OUT + 'maps/task1_pv_index_map.png'
    )
    plot_interactive_map(
        buildings,
        output_path=OUT + 'maps/task1_pv_interactive.html'
    )
    plot_analysis_charts(buildings, ml_metrics)
    plot_top_buildings(buildings, top_n=20)

    # ── [11] Ekspor ───────────────────────────────────────────────────────────
    export_results(buildings)

    elapsed = (datetime.now() - t_start).seconds
    log.info(f'\n✓ Task 1 selesai dalam {elapsed} detik')
    log.info(f'  Output tersimpan di: {OUT}')
    return buildings


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    result = main()
