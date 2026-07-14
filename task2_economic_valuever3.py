#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


"""

import os
import sys
import json
import logging
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial import cKDTree

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error

'''
Jangan dipake matplotlib yang ini
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
'''
import matplotlib
matplotlib.use('Agg')  # <--- TAMBAHKAN BARIS INI
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

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

# ═══════════════════════════════════════════════════════════════════════════════
# §1  KONFIGURASI
# ═══════════════════════════════════════════════════════════════════════════════
CFG = dict(
    # Input files
    bldg_path    = 'Building_Footprint.geojson',
    poi_path     = 'data/osm/economic_poi.geojson',
    road_path    = 'data/osm/road_network.geojson',
    trans_path   = 'data/osm/transport_stops.geojson',
    lu_path      = 'data/osm/landuse.geojson',
    
    # Output dir
    out_dir      = 'output/task2/',
    
    # CRS Setup (EPSG:32648 untuk proyeksi UTM metrik di wilayah Asia Tenggara)
    geo_crs      = 'EPSG:4326',
    proj_crs     = 'EPSG:32648',
    
    # Radius pencarian spasial (meter)
    radius_bldg  = 100,  # Radius hitung kepadatan bangunan
    radius_poi   = 300,  # Radius hitung kepadatan POI ekonomi
    
    # Parameter Machine Learning
    rf_n_trees   = 200,
    rf_seed      = 42,
    cv_k         = 5,
    
    # Threshold Klasifikasi
    thr_low      = 0.33,
    thr_high     = 0.67,
)

# Buat direktori output
for _sub in ['', 'data/', 'figures/', 'maps/']:
    Path(CFG['out_dir'] + _sub).mkdir(parents=True, exist_ok=True)

# Konfigurasi Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s │ %(levelname)-7s │ %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(CFG['out_dir'] + 'task2.log', mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger('Task2_Econ')


# ═══════════════════════════════════════════════════════════════════════════════
# §2  FUNGSI BANTUAN SPASIAL (KD-TREE)
# ═══════════════════════════════════════════════════════════════════════════════

def get_nearest_distance(src_pts: np.ndarray, tgt_pts: np.ndarray) -> np.ndarray:
    """Mengembalikan jarak terdekat dari src ke tgt."""
    if len(tgt_pts) == 0: return np.full(len(src_pts), 5000.0) # default jauh
    tree = cKDTree(tgt_pts)
    dist, _ = tree.query(src_pts, k=1)
    return dist

def get_nearest_attribute(src_pts: np.ndarray, tgt_pts: np.ndarray, 
                          tgt_vals: np.ndarray) -> np.ndarray:
    """Mengembalikan nilai atribut target yang lokasinya terdekat dengan src."""
    if len(tgt_pts) == 0: return np.zeros(len(src_pts))
    tree = cKDTree(tgt_pts)
    _, idx = tree.query(src_pts, k=1)
    return tgt_vals[idx]

def get_density_within_radius(src_pts: np.ndarray, tgt_pts: np.ndarray, 
                              radius: float, weights: np.ndarray = None) -> np.ndarray:
    """Menghitung jumlah/bobot titik target dalam radius tertentu dari titik src."""
    if len(tgt_pts) == 0: return np.zeros(len(src_pts))
    tree = cKDTree(tgt_pts)
    density = np.zeros(len(src_pts))
    for i, pt in enumerate(src_pts):
        idx = tree.query_ball_point(pt, radius)
        if weights is not None:
            density[i] = np.sum(weights[idx])
        else:
            density[i] = len(idx)
    return density


# ═══════════════════════════════════════════════════════════════════════════════
# §3  TASK 2.1 — REKAYASA FITUR (FEATURE ENGINEERING)
# ═══════════════════════════════════════════════════════════════════════════════

def feature_engineering() -> gpd.GeoDataFrame:
    log.info('[TASK 2.1] Memulai Rekayasa Fitur Geospasial...')
    
    # 1. Load Data & Reproject ke CRS Metrik
    log.info('       Memuat data bangunan dan OSM...')
    bldg = gpd.read_file(CFG['bldg_path']).to_crs(CFG['proj_crs'])
    poi  = gpd.read_file(CFG['poi_path']).to_crs(CFG['proj_crs'])
    road = gpd.read_file(CFG['road_path']).to_crs(CFG['proj_crs'])
    tran = gpd.read_file(CFG['trans_path']).to_crs(CFG['proj_crs'])
    luse = gpd.read_file(CFG['lu_path']).to_crs(CFG['proj_crs'])
    
    # Validasi dan Ekstraksi Luas/Tinggi Bangunan
    if 'area_m2' not in bldg.columns:
        bldg['area_m2'] = bldg.geometry.area
    if 'height' not in bldg.columns:
        bldg['height'] = 6.0 # asumsi default
    bldg['height'] = pd.to_numeric(bldg['height'], errors='coerce').fillna(6.0)
    
    # Ekstraksi koordinat X,Y bangunan (Centroid)
    bldg_pts = np.column_stack((bldg.centroid.x, bldg.centroid.y))
    
    # --- A. Fitur Anchor Ekonomi & POI ---
    log.info('       Mengekstraksi fitur POI dan Anchor Ekonomi...')
    poi_pts = np.column_stack((poi.geometry.x, poi.geometry.y))
    anchor_poi = poi[poi['is_anchor'] == 1]
    anchor_pts = np.column_stack((anchor_poi.geometry.x, anchor_poi.geometry.y))
    
    bldg['dist_to_anchor'] = get_nearest_distance(bldg_pts, anchor_pts)
    bldg['poi_density']    = get_density_within_radius(
        bldg_pts, poi_pts, CFG['radius_poi'], weights=poi['econ_weight'].values
    )
    
    # --- B. Fitur Jaringan Jalan & Aksesibilitas ---
    log.info('       Mengekstraksi fitur Jaringan Jalan...')
    # Titik proksi dari garis jalan (menggunakan titik tengah atau banyak titik)
    road_pts = np.column_stack((road.centroid.x, road.centroid.y))
    bldg['road_hierarchy'] = get_nearest_attribute(bldg_pts, road_pts, road['road_level'].values)
    
    # Jalan Utama (Level >= 3: Primary, Secondary, Trunk, Motorway)
    main_road = road[road['road_level'] >= 3]
    if not main_road.empty:
        mr_pts = np.column_stack((main_road.centroid.x, main_road.centroid.y))
        bldg['dist_to_main_road'] = get_nearest_distance(bldg_pts, mr_pts)
    else:
        bldg['dist_to_main_road'] = get_nearest_distance(bldg_pts, road_pts)
    
    # --- C. Kedekatan Transportasi Publik ---
    log.info('       Mengekstraksi fitur Transportasi Publik...')
    tran_pts = np.column_stack((tran.geometry.x, tran.geometry.y))
    bldg['dist_to_transport'] = get_nearest_distance(bldg_pts, tran_pts)
    
    # --- D. Kepadatan Bangunan Sekitar ---
    log.info('       Menghitung Kepadatan Bangunan (Urban Density)...')
    bldg['bldg_density'] = get_density_within_radius(
        bldg_pts, bldg_pts, CFG['radius_bldg'], weights=bldg['area_m2'].values
    )
    
    # --- E. Penggunaan Lahan (Landuse Score) ---
    log.info('       Mengintegrasikan Land Use Score...')
    luse_pts = np.column_stack((luse.centroid.x, luse.centroid.y))
    bldg['landuse_score'] = get_nearest_attribute(bldg_pts, luse_pts, luse['lu_score'].values)
    
    # --- F. Indeks Aksesibilitas Terpadu ---
    # Fungsi eksponensial decay: jarak makin dekat = skor makin tinggi
    bldg['accessibility_index'] = (
        np.exp(-bldg['dist_to_main_road'] / 500) * 0.5 + 
        np.exp(-bldg['dist_to_transport'] / 800) * 0.5
    )
    
    log.info('       ✓ Feature Engineering selesai.')
    return bldg


# ═══════════════════════════════════════════════════════════════════════════════
# §4  TASK 2.2 — ESTIMASI ML NILAI EKONOMI RELATIF
# ═══════════════════════════════════════════════════════════════════════════════

def build_economic_model(bldg: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Menggunakan Machine Learning (Random Forest) untuk memprediksi nilai ekonomi.
    Karena tidak ada data ground truth (transaksi riil), kita menciptakan 'proxy_target' 
    menggunakan Teori Gravitasi Urban & Hedonic Pricing. ML mempelajari struktur
    fitur ini dan memperhalus hasil akhirnya.
    """
    log.info('[TASK 2.2] Estimasi Nilai Ekonomi dengan ML (Random Forest)...')
    
    features = [
        'dist_to_anchor', 'dist_to_main_road', 'dist_to_transport',
        'poi_density', 'bldg_density', 'road_hierarchy',
        'landuse_score', 'accessibility_index', 'area_m2', 'height'
    ]
    
    # 1. Sintesis Proxy Target (Urban Gravity Model)
    # Nilai berbanding lurus dengan Luas, Kepadatan, Landuse, dan Aksesibilitas.
    # Berbanding terbalik dengan Jarak ke Pusat Anchor.
    volume = bldg['area_m2'] * bldg['height']
    gravity = (bldg['poi_density'] * 0.4 + bldg['bldg_density'] / bldg['bldg_density'].max() * 0.3)
    proximity = 1 / (1 + (bldg['dist_to_anchor'] / 100))
    
    proxy_target = np.log1p(volume * bldg['landuse_score'] * (1 + gravity) * (1 + proximity))
    
    X = bldg[features].values
    y = proxy_target.values
    
    # 2. Pelatihan Model ML & Cross Validation
    rf = RandomForestRegressor(
        n_estimators=CFG['rf_n_trees'], 
        random_state=CFG['rf_seed'],
        max_depth=12,
        n_jobs=-1
    )
    
    kf = KFold(n_splits=CFG['cv_k'], shuffle=True, random_state=CFG['rf_seed'])
    y_pred_cv = cross_val_predict(rf, X, y, cv=kf)
    
    r2 = r2_score(y, y_pred_cv)
    mae = mean_absolute_error(y, y_pred_cv)
    
    log.info(f'       ML Model ({CFG["cv_k"]}-Fold CV) → R²: {r2:.4f} | MAE: {mae:.4f}')
    if r2 > 0.7:
        log.info('       Model sangat baik dalam menangkap varians ekonomi keruangan.')
    
    # Train full model dan prediksi
    rf.fit(X, y)
    bldg['raw_economic_value'] = rf.predict(X)
    
    # Feature Importance Logging
    fi = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    log.info('       Feature Importances (Top 3):')
    for feat, imp in fi.head(3).items():
        log.info(f'         - {feat:20s}: {imp:.3f}')
        
    return bldg, fi


# ═══════════════════════════════════════════════════════════════════════════════
# §5  TASK 2.3 — NORMALISASI & KLASIFIKASI
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_and_classify(bldg: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    log.info('[TASK 2.3] Normalisasi Min-Max (0-1) dan Klasifikasi...')
    
    # Normalisasi Indeks (Min-Max Scaling)
    scaler = MinMaxScaler(feature_range=(0, 1))
    vals = bldg['raw_economic_value'].values.reshape(-1, 1)
    bldg['economic_value_index'] = scaler.fit_transform(vals).ravel()
    
    # Klasifikasi Rendah/Sedang/Tinggi
    bins = [-np.inf, CFG['thr_low'], CFG['thr_high'], np.inf]
    labels = ['Rendah', 'Sedang', 'Tinggi']
    bldg['economic_class'] = pd.cut(bldg['economic_value_index'], bins=bins, labels=labels)
    
    # Ranking
    bldg['economic_rank'] = bldg['economic_value_index'].rank(ascending=False, method='min').astype(int)
    
    dist = bldg['economic_class'].value_counts()
    log.info(f'       Distribusi Kelas: Tinggi={dist.get("Tinggi",0)}, '
             f'Sedang={dist.get("Sedang",0)}, Rendah={dist.get("Rendah",0)}')
    
    return bldg

# ═══════════════════════════════════════════════════════════════════════════════
# §6  VISUALISASI (GAYA KARTOGRAFI MODERN SEPERTI TASK 1)
# ═══════════════════════════════════════════════════════════════════════════════
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

# Colormap khusus ekonomi (Kuning ke Biru Gelap)
ECON_CMAP = plt.cm.YlGnBu
CLASS_COLORS = {'Rendah': '#ffffcc', 'Sedang': '#41b6c4', 'Tinggi': '#253494'}

'''def plot_static_map(bldg: gpd.GeoDataFrame, output_path: str):
    """Peta kartografi statis choropleth Economic Value Index."""
    log.info('[VIZ] Membuat peta statis choropleth ekonomi...')
    bldg_wm = bldg.to_crs('EPSG:3857') 
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Basemap (jika contextily tersedia)
    try:
        import contextily as ctx
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom='auto', alpha=0.6, crs='EPSG:3857')
    except ImportError:
        pass '''
def plot_static_map(bldg: gpd.GeoDataFrame, output_path: str):
    """Peta kartografi statis choropleth Economic Value Index (Versi Anti-Gagal)."""
    log.info('[VIZ] Membuat peta statis choropleth ekonomi...')
    
    # Bersihkan geometri yang mungkin rusak & ubah ke CRS Web Mercator
    bldg_clean = bldg[bldg.geometry.is_valid & ~bldg.geometry.is_empty].copy()
    bldg_wm = bldg_clean.to_crs('EPSG:3857') 
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # 1. PLOT BANGUNAN (Paksa Z-Order=10 agar selalu di atas basemap)
    try:
        bldg_wm.plot(
            column='economic_value_index', 
            cmap=ECON_CMAP,
            linewidth=0.5, 
            edgecolor='#333333', 
            legend=False, 
            ax=ax, 
            vmin=0, 
            vmax=1,
            zorder=10  # <-- Sangat Penting
        )
    except Exception as e:
        log.error(f"       [VIZ] Gagal menggambar poligon bangunan: {e}")

    # 2. TAMBAHKAN BASEMAP (Paksa Z-Order=1 agar selalu di bawah)
    try:
        import contextily as ctx
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, alpha=0.6, crs='EPSG:3857', zorder=1)
    except ImportError:
        log.warning("       [VIZ] Contextily tidak terinstall, basemap dilewati.")
    except Exception as e:
        log.warning(f"       [VIZ] Gagal memuat basemap: {e}")

    # 3. COLORBAR
    try:
        sm = plt.cm.ScalarMappable(cmap=ECON_CMAP, norm=mcolors.Normalize(vmin=0, vmax=1))
        cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, aspect=30)
        cbar.set_label('Economic Value Index', fontsize=11, fontweight='bold')
        cbar.set_ticks([0, CFG['thr_low'], CFG['thr_high'], 1.0])
        cbar.set_ticklabels([f'0.0\n(Rendah)', f'{CFG["thr_low"]}', f'{CFG["thr_high"]}', '1.0\n(Tinggi)'])
    except Exception as e:
        log.warning(f"       [VIZ] Gagal membuat colorbar: {e}")

    # 4. ORNAMEN KARTOGRAFI (Arah Utara & Skala)
    try:
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        
        # Arah Utara
        nx, ny = x0 + (x1-x0)*0.96, y0 + (y1-y0)*0.92
        ax.annotate('N', xy=(nx, ny), xytext=(nx, ny-(y1-y0)*0.05),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2),
                    ha='center', va='bottom', fontsize=12, fontweight='bold', zorder=20)

        # Scalebar (Perkiraan 100 meter)
        sb_len, sb_x, sb_y = 100, x0 + (x1-x0)*0.05, y0 + (y1-y0)*0.04
        ax.plot([sb_x, sb_x+sb_len], [sb_y, sb_y], 'k-', linewidth=4, zorder=20)
        ax.text(sb_x + sb_len/2, sb_y+(y1-y0)*0.012, '100 m', ha='center', fontsize=8, zorder=20)
    except Exception as e:
        log.warning(f"       [VIZ] Gagal membuat skala/arah utara: {e}")

    # 5. JUDUL & PENGATURAN AKHIR
    n_high = int((bldg['economic_class'] == 'Tinggi').sum())
    ax.set_title(
        'Economic Value Index — Per Bangunan\n'
        f'Task 2 Output | {len(bldg):,} bangunan | {n_high} prioritas Tinggi',
        fontsize=14, fontweight='bold', pad=12
    )
    ax.set_axis_off()
    
    # 6. PENYIMPANAN AMAN MENGGUNAKAN FIG OBJEK
    try:
        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        log.info(f'       Peta statis tersimpan → {output_path}')
    except Exception as e:
        log.error(f"       [VIZ] Gagal menyimpan file PNG: {e}")
    finally:
        plt.close(fig) # Memastikan memori dibersihkan dengan benar


def plot_interactive_map(bldg: gpd.GeoDataFrame, output_path: str):
    """Peta interaktif Folium dengan tooltip ekonomi per bangunan."""
    try:
        import folium
    except ImportError:
        log.warning('[VIZ] folium tidak terinstall → skip peta interaktif')
        return

    log.info('[VIZ] Membuat peta interaktif Folium ekonomi...')
    bldg_geo = bldg.to_crs('EPSG:4326').copy()
    bldg_geo['economic_class'] = bldg_geo['economic_class'].astype(str)
    
    centre = [bldg_geo.geometry.centroid.y.mean(), bldg_geo.geometry.centroid.x.mean()]
    m = folium.Map(location=centre, zoom_start=16, tiles='CartoDB positron')

    def _color(idx_val):
        if idx_val >= CFG['thr_high']: return CLASS_COLORS['Tinggi']
        if idx_val >= CFG['thr_low']:  return CLASS_COLORS['Sedang']
        return CLASS_COLORS['Rendah']

    # GeoJSON layer utama
    geojson_layer = folium.GeoJson(
        bldg_geo[['geometry', 'economic_value_index', 'economic_class', 'economic_rank',
                  'dist_to_anchor', 'dist_to_main_road', 'poi_density', 'accessibility_index']].to_json(),
        name='Economic Value Index',
        style_function=lambda feat: {
            'fillColor' : _color(feat['properties']['economic_value_index']),
            'color'     : '#444',
            'weight'    : 0.4,
            'fillOpacity': 0.75,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['economic_rank', 'economic_value_index', 'economic_class', 'dist_to_anchor', 'poi_density'],
            aliases=['Ranking', 'Econ Index', 'Kelas', 'Jarak ke Pusat (m)', 'Kepadatan POI'],
            localize=True, sticky=True,
            style='font-size:11px;background:rgba(255,255,255,0.9);border-radius:4px;'
        ),
        highlight_function=lambda _: {'weight': 2, 'color': 'black'}
    )
    geojson_layer.add_to(m)
    folium.LayerControl().add_to(m)
    m.save(output_path)


def plot_economic_analysis(bldg: gpd.GeoDataFrame, fi: pd.Series):
    """Menghasilkan grafik Python modern 3-Panel."""
    log.info('[VIZ] Membuat grafik analisis ekonomi...')
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Task 2 — Analisis Economic Value Index', fontsize=16, fontweight='bold', y=1.02)
    
    # Panel 1: Distribusi
    sns.histplot(bldg['economic_value_index'], bins=35, kde=True, color='#41b6c4', ax=axes[0])
    axes[0].axvline(CFG['thr_low'], color='#fdae61', ls='--')
    axes[0].axvline(CFG['thr_high'], color='#d7191c', ls='--')
    axes[0].set_title('A. Distribusi Economic Index', fontweight='bold')
    
    # Panel 2: Aksesibilitas vs Nilai
    sc = axes[1].scatter(bldg['accessibility_index'], bldg['economic_value_index'], 
                         c=bldg['economic_value_index'], cmap=ECON_CMAP, s=20, alpha=0.7)
    axes[1].set_title('B. Aksesibilitas vs Nilai Ekonomi', fontweight='bold')
    axes[1].set_xlabel('Indeks Aksesibilitas Gabungan')
    axes[1].set_ylabel('Economic Value Index')
    fig.colorbar(sc, ax=axes[1], fraction=0.04, pad=0.02)
    
    # Panel 3: Feature Importances
    sns.barplot(x=fi.values[:7], y=fi.index[:7], palette='YlGnBu_r', ax=axes[2])
    axes[2].set_title('C. Top 7 Fitur Geospasial (Random Forest)', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(CFG['out_dir'] + 'figures/task2_economic_analysis.png')
    plt.close()


def plot_top_buildings(bldg: gpd.GeoDataFrame, top_n: int = 20):
    """Bar chart Top-N bangunan prioritas ekonomi."""
    log.info(f'[VIZ] Membuat chart Top {top_n} bangunan ekonomi...')
    top = bldg.nsmallest(top_n, 'economic_rank').copy()
    top['label'] = [f"Bldg-{idx}" for idx in top.index]

    fig, ax = plt.subplots(figsize=(10, 8))
    vals = top['economic_value_index'].values
    colors_bar = [ECON_CMAP(v) for v in vals]
    
    ax.barh(top['label'], vals, color=colors_bar, edgecolor='#444', lw=0.5)
    ax.set_xlabel('Economic Value Index [0-1]', fontsize=10)
    ax.set_title(f'Top {top_n} Bangunan dengan Nilai Ekonomi Tertinggi', fontweight='bold')
    ax.invert_yaxis()
    
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(CFG['out_dir'] + f'figures/task2_top{top_n}_buildings.png')
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# §7  MAIN PIPELINE & EKSPOR
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    log.info('=' * 68)
    log.info('  TASK 2 — ECONOMIC VALUE INDEX')
    log.info('  Pipeline Eksekusi Rekayasa Fitur & Machine Learning')
    log.info('=' * 68)
    
    t_start = datetime.now()
    
    # 1. Feature Engineering
    bldg = feature_engineering()
    
    # 2. Estimasi ML
    bldg, fi = build_economic_model(bldg)
    
    # 3. Normalisasi & Klasifikasi
    bldg = normalize_and_classify(bldg)
    
    # 4. Visualisasi Grafik & Peta (Diperbarui seperti Task 1)
    plot_economic_analysis(bldg, fi)
    plot_static_map(bldg, CFG['out_dir'] + 'maps/task2_economic_map.png')
    plot_interactive_map(bldg, CFG['out_dir'] + 'maps/task2_economic_interactive.html')
    plot_top_buildings(bldg, top_n=20)
    
    # 5. Ekspor Hasil
    log.info('[EKSPOR] Menyimpan hasil pemrosesan...')
    
    keep_cols = [
        'area_m2', 'height', 'dist_to_anchor', 'dist_to_main_road',
        'poi_density', 'accessibility_index', 'economic_value_index', 
        'economic_class', 'economic_rank', 'geometry'
    ]
    
    out_gdf = bldg[[c for c in keep_cols if c in bldg.columns]].copy()
    out_gdf = out_gdf.to_crs(CFG['geo_crs']) # Kembalikan ke EPSG:4326 untuk GeoJSON
    out_gdf['economic_class'] = out_gdf['economic_class'].astype(str)
    
    # Simpan GeoJSON
    out_gdf.to_file(CFG['out_dir'] + 'data/economic_value_index.geojson', driver='GeoJSON')
    
    # Simpan CSV
    out_gdf.drop(columns='geometry').to_csv(CFG['out_dir'] + 'data/economic_summary.csv', index=False)
    
    # Simpan Metadata/Statistik JSON
    stats = {
        'n_buildings': len(out_gdf),
        'mean_economic_index': round(float(out_gdf['economic_value_index'].mean()), 4),
        'top_feature': fi.index[0],
        'generated_at': datetime.now().isoformat()
    }
    with open(CFG['out_dir'] + 'data/economic_metadata.json', 'w') as f:
        json.dump(stats, f, indent=2)
        
    log.info(f'       Data tersimpan di {CFG["out_dir"]}')
    
    elapsed = (datetime.now() - t_start).seconds
    log.info(f'\n✓ Task 2 selesai dalam {elapsed} detik')

if __name__ == '__main__':
    main()