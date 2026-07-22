import os, sys, json, math, logging, warnings
from pathlib   import Path
from datetime  import datetime
from itertools import product as iproduct

import numpy           as np
import pandas          as pd
import geopandas       as gpd
from   shapely.geometry import Point
from   scipy.spatial    import cKDTree
from   scipy.stats      import pearsonr, spearmanr

from   sklearn.preprocessing     import MinMaxScaler, RobustScaler
from   sklearn.decomposition     import PCA
from   sklearn.cluster           import KMeans
from   sklearn.metrics           import silhouette_score
from   sklearn.model_selection   import KFold, cross_val_predict
from   sklearn.metrics           import r2_score, mean_absolute_error
from   sklearn.ensemble          import GradientBoostingRegressor, RandomForestRegressor

import matplotlib
import matplotlib.pyplot    as plt
import matplotlib.patches   as mpatches
import matplotlib.colors    as mcolors
import matplotlib.ticker    as ticker
from   matplotlib.colors     import LinearSegmentedColormap, BoundaryNorm
from   matplotlib.gridspec   import GridSpec
import seaborn               as sns

_opt = {}
for _pkg, _key in [('contextily','ctx'),('folium','folium')]:
    try: _opt[_key] = __import__(_pkg)
    except ImportError: _opt[_key] = None

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

warnings.filterwarnings('ignore')
matplotlib.rcParams.update({
    'font.family'       : 'DejaVu Sans',
    'figure.dpi'        : 150,
    'savefig.dpi'       : 300,
    'savefig.bbox'      : 'tight',
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
    'axes.grid'         : True,
    'grid.alpha'        : 0.3,
})

CFG = dict(
   
    bldg_path    = 'data/Building_Footprint.geojson',
    aod_path     = 'data/satellite/aod_samples.csv',
    lst_path     = 'data/satellite/lst_samples.csv',
    ndvi_path    = 'data/satellite/ndvi_ndbi_samples.csv',
    cf_path      = 'data/satellite/cf_samples.csv',
    lc_path      = 'data/satellite/landcover_samples.csv',
    out_dir      = 'output/task3/',

    geo_crs      = 'EPSG:4326',
    proj_crs     = 'EPSG:32648', 

    weights = {
        'AOD'  : 0.30,  
        'LST'  : 0.25,  
        'NDVI' : 0.20,  
        'NDBI' : 0.15,  
        'CF'   : 0.05,  
        'LC'   : 0.05,  
    },

    scenarios = {
        'Baseline'      : {'AOD':0.30,'LST':0.25,'NDVI':0.20,'NDBI':0.15,'CF':0.05,'LC':0.05},
        'Air_Quality'   : {'AOD':0.45,'LST':0.20,'NDVI':0.15,'NDBI':0.10,'CF':0.05,'LC':0.05},
        'Heat_Stress'   : {'AOD':0.20,'LST':0.40,'NDVI':0.15,'NDBI':0.15,'CF':0.05,'LC':0.05},
        'Vegetation'    : {'AOD':0.20,'LST':0.15,'NDVI':0.40,'NDBI':0.15,'CF':0.05,'LC':0.05},
        'Equal_Weight'  : {'AOD':0.20,'LST':0.20,'NDVI':0.20,'NDBI':0.20,'CF':0.10,'LC':0.10},
    },

    thr_low      = 0.33,
    thr_high     = 0.67,

    idw_k        = 5,    
    idw_power    = 2,    

    gbr_n        = 200,  
    gbr_lr       = 0.05,
    gbr_depth    = 4,
    gbr_seed     = 42,
    km_max       = 6,    
    cv_k         = 5,    
    pca_n        = 4,    
    min_valid    = 30,   
)

OUT = CFG['out_dir']
for _s in ['','maps/','figures/','data/']:
    Path(OUT + _s).mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s │ %(levelname)-7s │ %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(OUT + 'task3.log', mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger('Task3')

EDI_CMAP = LinearSegmentedColormap.from_list(
    'edi', ['#1a9850', '#d9ef8b', '#fee08b', '#fc8d59', '#d73027'], N=256
)
AOD_CMAP  = LinearSegmentedColormap.from_list('aod', ['#fff7bc','#f03b20'], N=256)
LST_CMAP  = LinearSegmentedColormap.from_list('lst', ['#deebf7','#08306b'], N=256)
NDVI_CMAP = LinearSegmentedColormap.from_list('ndvi',['#d73027','#1a9850'], N=256)
NDBI_CMAP = LinearSegmentedColormap.from_list('ndbi',['#f7f7f7','#8c510a'], N=256)

CLASS_COLORS = {'Rendah':'#1a9850','Sedang':'#fee08b','Tinggi':'#d73027'}
CLASS_ORDER  = ['Rendah','Sedang','Tinggi']

IND_META = {  
    'AOD' : {'label':'AOD (Polusi Udara)',       'arah':+1, 'unit':'–',  'cmap':AOD_CMAP },
    'LST' : {'label':'LST (Suhu Permukaan)',     'arah':+1, 'unit':'°C', 'cmap':LST_CMAP },
    'NDVI': {'label':'NDVI (Vegetasi, inv.)',    'arah':-1, 'unit':'–',  'cmap':NDVI_CMAP},
    'NDBI': {'label':'NDBI (Lahan Terbangun)',   'arah':+1, 'unit':'–',  'cmap':NDBI_CMAP},
    'CF'  : {'label':'CF (Tutupan Awan)',        'arah':+1, 'unit':'–',  'cmap':'Blues'  },
    'LC'  : {'label':'LC (Land Cover deg.)',     'arah':+1, 'unit':'–',  'cmap':'Reds'   },
}

def load_buildings(path: str) -> gpd.GeoDataFrame:
    log.info(f'[LOAD] Building Footprint ← {path}')
    gdf = gpd.read_file(path)
    gdf = gdf[gdf.geometry.type.isin(['Polygon','MultiPolygon'])].copy()
    if gdf.crs is None:
        gdf = gdf.set_crs(CFG['geo_crs'])
    gdf = gdf.to_crs(CFG['proj_crs'])
    gdf['geometry'] = gdf['geometry'].buffer(0)
    gdf = gdf[gdf.geometry.is_valid & ~gdf.geometry.is_empty].reset_index(drop=True)
    gdf['bldg_id'] = gdf.index.astype(int)
    if 'area_m2' not in gdf.columns:
        gdf['area_m2'] = gdf.geometry.area
    log.info(f'       {len(gdf):,} bangunan valid | CRS: {gdf.crs}')
    return gdf

def _load_csv_indicator(path: str, col_map: dict,
                         fallback_value: float = None,
                         buildings: gpd.GeoDataFrame = None) -> pd.DataFrame:
    """
    Muat CSV indikator satelit. Jika file tidak ada:
      - Gunakan fallback_value sebagai nilai konstan (testing mode)
      - Atau generate synthetic gradient berdasarkan posisi bangunan
    """
    if not Path(path).exists():
        log.warning(f'  File tidak ditemukan: {path}')
        if buildings is not None:
            bldg_geo = buildings.to_crs(CFG['geo_crs'])
            cx = bldg_geo.geometry.centroid.x.values
            cy = bldg_geo.geometry.centroid.y.values
            df = pd.DataFrame({'lon': cx, 'lat': cy})
            for col_out, val in col_map.items():
                if fallback_value is not None:
                    df[col_out] = fallback_value
                else:
                   
                    cx_n = (cx - cx.min()) / (cx.max() - cx.min() + 1e-10)
                    cy_n = (cy - cy.min()) / (cy.max() - cy.min() + 1e-10)
                    df[col_out] = (0.3 + 0.4*cx_n + 0.2*cy_n
                                   + 0.1*np.random.RandomState(42).rand(len(cx)))
            log.warning(f'  → Synthetic fallback (testing): {list(col_map.keys())}')
            return df
        return pd.DataFrame(columns=['lat','lon'] + list(col_map.keys()))

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
   
    for c in df.columns:
        cl = c.lower()
        if cl in ('lat','latitude','y'):   df = df.rename(columns={c:'lat'})
        if cl in ('lon','longitude','x'):  df = df.rename(columns={c:'lon'})
   
    for c_in, c_out in col_map.items():
        if c_in in df.columns and c_in != c_out:
            df = df.rename(columns={c_in: c_out})
    return df.dropna(subset=['lat','lon'])

def load_all_satellite(buildings: gpd.GeoDataFrame) -> dict:
    """
    Muat semua data satelit. Return dict: {nama: DataFrame(lat,lon,nilai)}.
    """
    log.info('[LOAD] Memuat data satelit...')
    sat = {}

    sat['AOD']  = _load_csv_indicator(
        CFG['aod_path'], {'AOD':'AOD'},
        fallback_value=None, buildings=buildings
    )

    sat['LST']  = _load_csv_indicator(
        CFG['lst_path'], {'LST_C':'LST','LST_celsius':'LST'},
        fallback_value=None, buildings=buildings
    )
    if 'LST' not in sat['LST'].columns:
        for c in sat['LST'].columns:
            if c not in ('lat','lon'): sat['LST'] = sat['LST'].rename(columns={c:'LST'})

    sat['NDVI'] = _load_csv_indicator(
        CFG['ndvi_path'], {'NDVI':'NDVI', 'NDBI':'NDBI'},
        fallback_value=None, buildings=buildings
    )
    if 'NDBI' not in sat['NDVI'].columns:
        sat['NDVI']['NDBI'] = 0.0

    sat['CF']   = _load_csv_indicator(
        CFG['cf_path'], {'CF':'CF', 'Cloud_Fraction_Mean':'CF'},
        fallback_value=0.30, buildings=buildings
    )
    if 'CF' not in sat['CF'].columns:
        sat['CF']['CF'] = 0.30

    sat['LC']   = _load_csv_indicator(
        CFG['lc_path'], {'LC_deg_score':'LC'},
        fallback_value=None, buildings=buildings
    )
    if 'LC' not in sat['LC'].columns and 'LC_class' in sat['LC'].columns:
        lc_score = {10:0.05,20:0.15,30:0.20,40:0.35,50:0.90,
                    60:0.65,70:0.10,80:0.05,90:0.10,95:0.05,100:0.10}
        sat['LC']['LC'] = sat['LC']['LC_class'].map(
            lambda c: lc_score.get(int(c), 0.50)
        )

    for k, df in sat.items():
        available = [c for c in df.columns if c not in ('lat','lon')]
        log.info(f'  {k:<6}: {len(df):>5,} titik | kolom: {available}')
    return sat

def idw_interpolate(bldg_pts: np.ndarray,
                     src_pts : np.ndarray,
                     src_vals: np.ndarray,
                     k: int = 5, power: float = 2) -> np.ndarray:
    """
    Inverse Distance Weighting (IDW) interpolasi.
    Formula: ẑ = Σ(wᵢ × zᵢ) / Σwᵢ,  wᵢ = 1/dᵢᵖ

    Parameters
    ----------
    bldg_pts : (N, 2) array — koordinat bangunan (lon, lat atau x, y)
    src_pts  : (M, 2) array — koordinat titik sumber
    src_vals : (M,)   array — nilai di titik sumber
    k        : jumlah tetangga terdekat
    power    : eksponen jarak (umumnya 2)
    """
    valid = ~np.isnan(src_vals)
    if valid.sum() == 0:
        return np.full(len(bldg_pts), np.nan)

    src_pts  = src_pts[valid]
    src_vals = src_vals[valid]
    tree     = cKDTree(src_pts)
    k_use    = min(k, len(src_pts))
    dist, idx = tree.query(bldg_pts, k=k_use)

    if k_use == 1:
        return src_vals[idx.ravel()]

    dist  = np.where(dist < 1e-10, 1e-10, dist) 
    w     = 1.0 / dist**power
    w_sum = w.sum(axis=1, keepdims=True)
    vals  = src_vals[idx]
    return (w * vals).sum(axis=1) / w_sum.ravel()

def assign_indicators_to_buildings(buildings: gpd.GeoDataFrame,
                                    sat: dict) -> gpd.GeoDataFrame:
    """
    Task 3.1 — Interpolasi (IDW) semua indikator satelit ke centroid bangunan.
    """
    log.info('[TASK3.1] IDW interpolasi indikator ke bangunan...')

    bldg_geo  = buildings.to_crs(CFG['geo_crs'])
    bldg_cx   = bldg_geo.geometry.centroid.x.values
    bldg_cy   = bldg_geo.geometry.centroid.y.values
    bldg_pts  = np.column_stack([bldg_cx, bldg_cy])

    if 'AOD' in sat and 'AOD' in sat['AOD'].columns:
        src = sat['AOD'][['lon','lat','AOD']].dropna()
        buildings['AOD'] = idw_interpolate(
            bldg_pts, src[['lon','lat']].values, src['AOD'].values,
            CFG['idw_k'], CFG['idw_power']
        )
        log.info(f'  AOD  → {buildings["AOD"].notna().sum():,} bangunan valid | '
                 f'mean={buildings["AOD"].mean():.4f}')

    if 'LST' in sat and 'LST' in sat['LST'].columns:
        src = sat['LST'][['lon','lat','LST']].dropna()
        buildings['LST'] = idw_interpolate(
            bldg_pts, src[['lon','lat']].values, src['LST'].values,
            CFG['idw_k'], CFG['idw_power']
        )
        log.info(f'  LST  → mean={buildings["LST"].mean():.1f}°C')

    ndvi_df = sat.get('NDVI', pd.DataFrame())
    for col_name in ['NDVI','NDBI']:
        if col_name in ndvi_df.columns:
            src = ndvi_df[['lon','lat', col_name]].dropna(subset=[col_name])
            buildings[col_name] = idw_interpolate(
                bldg_pts, src[['lon','lat']].values, src[col_name].values,
                CFG['idw_k'], CFG['idw_power']
            )
            log.info(f'  {col_name:<5}→ mean={buildings[col_name].mean():.4f}')

    if 'CF' in sat and 'CF' in sat['CF'].columns:
        src = sat['CF'][['lon','lat','CF']].dropna()
        buildings['CF'] = idw_interpolate(
            bldg_pts, src[['lon','lat']].values, src['CF'].values,
            CFG['idw_k'], CFG['idw_power']
        ).clip(0, 1)
        log.info(f'  CF   → mean={buildings["CF"].mean():.4f}')

    if 'LC' in sat and 'LC' in sat['LC'].columns:
        src = sat['LC'][['lon','lat','LC']].dropna()
        buildings['LC'] = idw_interpolate(
            bldg_pts, src[['lon','lat']].values, src['LC'].values,
            CFG['idw_k'], CFG['idw_power']
        ).clip(0, 1)
        log.info(f'  LC   → mean={buildings["LC"].mean():.4f}')

    for col in ['AOD','LST','NDVI','NDBI','CF','LC']:
        if col in buildings.columns:
            med = buildings[col].median()
            n   = buildings[col].isna().sum()
            if n > 0:
                buildings[col] = buildings[col].fillna(med)
                log.warning(f'  {col}: {n} NaN diisi median ({med:.4f})')

    return buildings

def normalize_indicators(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Task 3.2 — Normalisasi Min-Max tiap indikator ke [0, 1].
    Kemudian sesuaikan arah: indikator yang TINGGI = RENDAH degradasi
    (yaitu NDVI) diinversi: kontribusi = 1 - nilai_norm.

    Kolom baru: {IND}_norm = nilai terkontribusi ke EDI [0–1]
    """
    log.info('[TASK3.2a] Normalisasi & penyesuaian arah indikator...')
    scaler = MinMaxScaler()

    for ind, meta in IND_META.items():
        if ind not in buildings.columns:
            log.warning(f'  {ind} tidak ditemukan → skip')
            continue

        vals = buildings[[ind]].values.astype(float)
        norm = scaler.fit_transform(vals).ravel()

        if meta['arah'] == -1:  
            norm = 1.0 - norm
            log.info(f'  {ind:<6}: normalisasi + DIINVERSI '
                     f'(mean_raw={buildings[ind].mean():.3f} → '
                     f'mean_EDI_contrib={norm.mean():.3f})')
        else:
            log.info(f'  {ind:<6}: normalisasi '
                     f'(mean_raw={buildings[ind].mean():.3f} → '
                     f'mean_EDI_contrib={norm.mean():.3f})')

        buildings[f'{ind}_norm'] = norm

    return buildings

def run_pca_analysis(buildings: gpd.GeoDataFrame) -> dict:
    """
    PCA (Principal Component Analysis) pada matriks indikator ternormalisasi.

    Tujuan:
      1. Identifikasi redudansi antar indikator
      2. Hitung PC1 sebagai composite alternative EDI
      3. Biplot untuk visualisasi struktur kovarian

    Justifikasi algoritma:
      PCA adalah metode standar untuk analisis struktur multi-variabel
      lingkungan (Boyacioglu & Boyacioglu 2008; Praus 2005). PC1
      menangkap variansi terbesar, sehingga mencerminkan 'mode utama'
      degradasi lingkungan pada area studi.
    """
    log.info('[ML-PCA] Principal Component Analysis indikator lingkungan...')

    norm_cols = [f'{ind}_norm' for ind in IND_META if f'{ind}_norm' in buildings.columns]
    if len(norm_cols) < 2:
        log.warning('  Kurang dari 2 indikator → PCA dilewati')
        return {}

    X = buildings[norm_cols].values
    n_comp = min(CFG['pca_n'], len(norm_cols))
    pca = PCA(n_components=n_comp, random_state=CFG['gbr_seed'])
    scores = pca.fit_transform(X)

    exp_var = pca.explained_variance_ratio_
    cum_var = np.cumsum(exp_var)
    loadings = pd.DataFrame(
        pca.components_.T,
        index=[c.replace('_norm','') for c in norm_cols],
        columns=[f'PC{i+1}' for i in range(n_comp)]
    )

    pc1 = scores[:, 0]
    if np.corrcoef(pc1, buildings['AOD_norm'].values)[0,1] < 0:
        pc1 = -pc1 
    mm = MinMaxScaler()
    buildings['EDI_pca'] = mm.fit_transform(pc1.reshape(-1,1)).ravel()

    log.info(f'  Variansi dijelaskan: ' +
             ' | '.join([f'PC{i+1}={v:.1%}' for i,v in enumerate(exp_var)]))
    log.info(f'  Kumulatif PC1–{n_comp}: {cum_var[-1]:.1%}')
    log.info(f'  Loadings PC1:\n' +
             '\n'.join([f'    {r}: {loadings["PC1"][r]:+.3f}' for r in loadings.index]))

    return {
        'pca'            : pca,
        'scores'         : scores,
        'loadings'       : loadings,
        'explained_var'  : exp_var.tolist(),
        'cumulative_var' : cum_var.tolist(),
        'norm_cols'      : norm_cols,
        'n_components'   : n_comp,
    }

def run_kmeans_clustering(buildings: gpd.GeoDataFrame,
                           pca_results: dict) -> dict:
    """
    K-Means clustering pada matriks indikator untuk segmentasi zona degradasi.

    Optimasi k: Silhouette Score untuk k = 2..km_max.
    Justifikasi:
      K-Means digunakan untuk menemukan kelompok bangunan alami berdasarkan
      profil lingkungan multi-dimensi, tanpa asumsi bentuk distribusi.
      Silhouette score memastikan pemilihan k optimal secara objektif
      (Rousseeuw 1987).
    """
    log.info('[ML-KMeans] Clustering zona degradasi lingkungan...')

    norm_cols = pca_results.get('norm_cols',
                [f'{i}_norm' for i in IND_META if f'{i}_norm' in buildings.columns])
    if len(norm_cols) < 2:
        log.warning('  Kurang dari 2 indikator → KMeans dilewati')
        return {}

    X = buildings[norm_cols].values
    if len(X) < 10:
        log.warning(f'  Terlalu sedikit data ({len(X)}) → KMeans dilewati')
        return {}

    best_k, best_sil, sil_scores = 2, -1, {}
    k_max = min(CFG['km_max'], len(X) - 1)
    for k in range(2, k_max + 1):
        km  = KMeans(n_clusters=k, random_state=CFG['gbr_seed'],
                     n_init=10, max_iter=300)
        lbl = km.fit_predict(X)
        sil = silhouette_score(X, lbl, sample_size=min(1000, len(X)))
        sil_scores[k] = sil
        log.info(f'  k={k} | silhouette={sil:.4f}')
        if sil > best_sil:
            best_sil, best_k = sil, k

    log.info(f'  ✓ k optimal = {best_k} (silhouette = {best_sil:.4f})')

    km_final = KMeans(n_clusters=best_k, random_state=CFG['gbr_seed'],
                       n_init=15, max_iter=500)
    labels   = km_final.fit_predict(X)

    buildings['km_cluster'] = labels

    centers = pd.DataFrame(km_final.cluster_centers_,
                            columns=[c.replace('_norm','') for c in norm_cols])
    log.info('  Profil cluster center:')
   
    log.info('\n' + centers.round(3).to_string())

    return {
        'model'       : km_final,
        'best_k'      : best_k,
        'best_sil'    : best_sil,
        'sil_scores'  : sil_scores,
        'centers'     : centers,
        'norm_cols'   : norm_cols,
    }

def smooth_edi_with_gbr(buildings: gpd.GeoDataFrame,
                          wlc_col: str = 'EDI_wlc') -> dict:
    """
    Gradient Boosting Regressor (GBR) untuk spatial smoothing EDI.

    Tujuan:
      Menangkap pola spasial non-linear EDI yang tidak dapat ditangkap
      oleh WLC linear, dengan memanfaatkan koordinat spasial dan
      interaksi antar indikator.

    Fitur:
      x, y (centroid proyeksi), semua {IND}_norm, area_m2 (jika ada)

    Target:
      EDI_wlc (WLC analytical score sebagai pseudo-label)

    Validasi:
      K-Fold CV dengan K=5 → R², MAE

    Justifikasi GBR vs Random Forest:
      GBR mengoptimalkan residual secara sekuensial (gradient descent),
      menghasilkan prediksi lebih halus dan akurat untuk data spasial
      kontinu dengan efek interaksi (Chen & Guestrin 2016). Depth
      pendek (4) mencegah overfitting pada data terbatas.
    """
    log.info('[ML-GBR] Gradient Boosting spatial smoothing EDI...')

    norm_cols = [f'{ind}_norm' for ind in IND_META if f'{ind}_norm' in buildings.columns]
    if len(norm_cols) < 2 or wlc_col not in buildings.columns:
        log.warning('  GBR: data belum siap → dilewati')
        return {}

    bldg_p  = buildings.copy() if buildings.crs.to_epsg() == int(CFG['proj_crs'].split(':')[1]) \
              else buildings.to_crs(CFG['proj_crs'])
    cx = bldg_p.geometry.centroid.x.values
    cy = bldg_p.geometry.centroid.y.values

    feat_parts = [cx.reshape(-1,1), cy.reshape(-1,1)]
    feat_names = ['x_m', 'y_m']
    for col in norm_cols:
        feat_parts.append(buildings[col].values.reshape(-1,1))
        feat_names.append(col)
    if 'area_m2' in buildings.columns:
        a_norm = MinMaxScaler().fit_transform(buildings[['area_m2']].values)
        feat_parts.append(a_norm)
        feat_names.append('area_norm')

    X = np.hstack(feat_parts)
    y = buildings[wlc_col].values

    if len(y) < CFG['min_valid']:
        log.warning(f'  GBR: hanya {len(y)} sampel → dilewati')
        return {}

    gbr = GradientBoostingRegressor(
        n_estimators   = CFG['gbr_n'],
        learning_rate  = CFG['gbr_lr'],
        max_depth      = CFG['gbr_depth'],
        subsample      = 0.80,
        min_samples_leaf = 5,
        random_state   = CFG['gbr_seed'],
    )

    kf   = KFold(n_splits=CFG['cv_k'], shuffle=True, random_state=CFG['gbr_seed'])
    y_cv = cross_val_predict(gbr, X, y, cv=kf)
    r2   = float(r2_score(y, y_cv))
    mae  = float(mean_absolute_error(y, y_cv))
    log.info(f'  GBR ({CFG["cv_k"]}-Fold CV) → R²={r2:.4f} | MAE={mae:.4f} '
             f'| n={len(y)}')

    gbr.fit(X, y)
    y_pred = gbr.predict(X).clip(0, 1)

    mm = MinMaxScaler()
    buildings['EDI_gbr'] = mm.fit_transform(y_pred.reshape(-1,1)).ravel()

    fi     = dict(zip(feat_names, gbr.feature_importances_))
    fi_top = sorted(fi, key=fi.get, reverse=True)[:3]
    log.info('  Feature importance (top 3): ' +
             ', '.join([f'{k}={fi[k]:.3f}' for k in fi_top]))

    xgb_metrics = {}
    if HAS_XGB:
        import xgboost as xgb
        xgb_model = xgb.XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8,
            random_state=CFG['gbr_seed'], verbosity=0
        )
        y_xgb = cross_val_predict(xgb_model, X, y, cv=kf)
        r2x   = float(r2_score(y, y_xgb))
        maex  = float(mean_absolute_error(y, y_xgb))
        log.info(f'  XGBoost ({CFG["cv_k"]}-Fold CV) → R²={r2x:.4f} | MAE={maex:.4f}')
        xgb_metrics = {'r2': r2x, 'mae': maex}

    return {
        'model'          : gbr,
        'r2'             : r2,
        'mae'            : mae,
        'xgb_metrics'    : xgb_metrics,
        'feature_names'  : feat_names,
        'feature_importance': fi,
        'n_samples'      : len(y),
        'cv_k'           : CFG['cv_k'],
        'algo'           : 'GradientBoostingRegressor (scikit-learn)',
        'justification'  : (
            'GBR dipilih karena: (1) menangkap interaksi non-linear antara '
            'koordinat spasial dan indikator lingkungan; (2) lebih robust '
            'terhadap outlier dibanding Linear Regression; (3) regularisasi '
            'implisit via subsample=0.8 mencegah overfitting.'
        ),
    }

def calculate_wlc_edi(buildings: gpd.GeoDataFrame,
                       weights: dict = None,
                       col_prefix: str = 'EDI_wlc') -> gpd.GeoDataFrame:
    """
    Task 3.2 — Weighted Linear Combination (WLC) Environmental Degradation Index.

    EDI = Σᵢ (wᵢ × IND_i_norm)

    Catatan penting:
      Semua {IND}_norm SUDAH disesuaikan arahnya di §5:
      → nilai tinggi selalu berarti LEBIH TERDEGRADASI
      → NDVI_norm sudah diinversi (1 - NDVI_norm_raw)
    """
    if weights is None:
        weights = CFG['weights']

    w_total = sum(weights.values())
    weights = {k: v/w_total for k, v in weights.items()}

    edi = np.zeros(len(buildings))
    w_used = 0.0
    for ind, w in weights.items():
        col = f'{ind}_norm'
        if col in buildings.columns:
            edi += w * buildings[col].values
            w_used += w
        else:
            log.warning(f'  WLC: {col} tidak ada → bobot dialihkan proporsional')

    if w_used < 0.01:
        log.error('  WLC: tidak ada indikator valid!')
        buildings[col_prefix] = 0.5
        return buildings

    edi = edi / w_used

    mm = MinMaxScaler()
    buildings[col_prefix] = mm.fit_transform(edi.reshape(-1,1)).ravel()
    log.info(f'  EDI WLC (bobot aktif={w_used:.2f}): '
             f'mean={buildings[col_prefix].mean():.4f} | '
             f'std={buildings[col_prefix].std():.4f}')
    return buildings

def calculate_final_edi(buildings: gpd.GeoDataFrame,
                         use_gbr: bool = True) -> gpd.GeoDataFrame:
    """
    EDI final = rata-rata berbobot:
      - EDI_wlc : 0.60 (WLC analytical — transparan dan dapat dipertanggungjawabkan)
      - EDI_gbr : 0.30 (GBR smooth — menangkap pola spasial non-linear)
      - EDI_pca : 0.10 (PCA composite — perspektif alternatif)
    """
    log.info('[TASK3.2b] Menghitung EDI final (ensemble WLC + GBR + PCA)...')

    cols_w = {'EDI_wlc': 0.60}
    if use_gbr and 'EDI_gbr' in buildings.columns:
        cols_w['EDI_gbr'] = 0.30
    if 'EDI_pca' in buildings.columns:
        cols_w['EDI_pca'] = 0.10

    total = sum(cols_w.values())
    edi_f = np.zeros(len(buildings))
    for col, w in cols_w.items():
        edi_f += (w/total) * buildings[col].values

    mm = MinMaxScaler()
    buildings['env_deg_index'] = mm.fit_transform(edi_f.reshape(-1,1)).ravel()
    log.info(f'  EDI final: mean={buildings["env_deg_index"].mean():.4f} | '
             f'komponen={list(cols_w.keys())}')
    return buildings

def classify_edi(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Klasifikasi 3 kelas:
      Rendah  [0.00, 0.33)  — kualitas lingkungan baik
      Sedang  [0.33, 0.67)  — degradasi menengah
      Tinggi  [0.67, 1.00]  — kualitas lingkungan paling terdegradasi
                              → PRIORITAS TERTINGGI untuk manfaat solar PV
    """
    lo, hi  = CFG['thr_low'], CFG['thr_high']
    bins    = [-np.inf, lo, hi, np.inf]
    labels  = ['Rendah','Sedang','Tinggi']
    buildings['env_class'] = pd.cut(
        buildings['env_deg_index'], bins=bins, labels=labels
    )
    buildings['env_rank'] = (
        buildings['env_deg_index']
        .rank(ascending=False, method='min')
        .astype(int)
    )
    dist = buildings['env_class'].value_counts()
    log.info(f'  Distribusi kelas: {dict(dist)}')
    return buildings

def sensitivity_analysis(buildings: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Uji sensitivitas EDI terhadap berbagai skenario bobot (Bagian 7.2).
    Untuk tiap skenario: hitung EDI, hitung Spearman rank correlation
    dengan baseline, identifikasi bangunan stabil (top-N tetap di top).
    """
    log.info('[SENS] Analisis sensitivitas bobot...')

    baseline = None
    rows     = []

    for scen_name, w in CFG['scenarios'].items():
        bldg_tmp = buildings.copy()
        bldg_tmp = calculate_wlc_edi(bldg_tmp, weights=w,
                                      col_prefix='EDI_tmp')
        if baseline is None:
            baseline = bldg_tmp['EDI_tmp'].values.copy()

        edi_this = bldg_tmp['EDI_tmp'].values
        rho, p   = spearmanr(baseline, edi_this)

        top20_base = set(np.argsort(baseline)[-20:])
        top20_this = set(np.argsort(edi_this)[-20:])
        stable_pct = len(top20_base & top20_this) / 20 * 100

        rows.append({
            'Skenario'       : scen_name,
            'w_AOD'          : w.get('AOD',0),
            'w_LST'          : w.get('LST',0),
            'w_NDVI'         : w.get('NDVI',0),
            'w_NDBI'         : w.get('NDBI',0),
            'w_CF'           : w.get('CF',0),
            'w_LC'           : w.get('LC',0),
            'EDI_mean'       : round(float(edi_this.mean()), 4),
            'EDI_std'        : round(float(edi_this.std()), 4),
            'Spearman_rho'   : round(float(rho), 4),
            'p_value'        : round(float(p), 6),
            'stable_top20_%' : round(stable_pct, 1),
        })
        log.info(f'  {scen_name:<18}: rho={rho:.4f} | stable={stable_pct:.0f}% top-20')

    df_sens = pd.DataFrame(rows)
    out = OUT + 'data/sensitivity_analysis.csv'
    df_sens.to_csv(out, index=False)
    log.info(f'  → {out}')
    return df_sens

def _north_arrow_scale(ax, bldg_wm: gpd.GeoDataFrame) -> None:
    """Tambahkan panah utara dan scalebar ke ax (Web Mercator)."""
    x0, x1 = ax.get_xlim()
    y0, y1  = ax.get_ylim()
    dx, dy  = x1-x0, y1-y0
   
    nx, ny = x0+dx*0.96, y0+dy*0.92
    ax.annotate('', xy=(nx, ny), xytext=(nx, ny-dy*0.06),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.text(nx, ny+dy*0.01, 'N', ha='center', va='bottom',
            fontsize=12, fontweight='bold')
   
    sb_x = x0 + dx*0.05
    sb_y = y0 + dy*0.04
    ax.plot([sb_x, sb_x+100], [sb_y, sb_y], 'k-', lw=4)
    ax.text(sb_x+50, sb_y+dy*0.012, '100 m', ha='center', fontsize=8)

def plot_edi_map(buildings: gpd.GeoDataFrame) -> None:
    """Peta kartografi statis choropleth EDI per bangunan."""
    log.info('[VIZ] Membuat peta statis EDI...')

    bldg_wm = buildings.to_crs('EPSG:3857')
    fig, ax  = plt.subplots(1, 1, figsize=(14, 10))

    bldg_wm.plot(column='env_deg_index', cmap=EDI_CMAP,
                  edgecolor='none', legend=False,
                  ax=ax, vmin=0, vmax=1, zorder=2)

    if _opt['ctx']:
        try:
            _opt['ctx'].add_basemap(ax, source=_opt['ctx'].providers.CartoDB.Positron,
                                     zoom='auto', alpha=0.6, crs='EPSG:3857', zorder=1)
        except Exception as e:
            log.warning(f'  Peta dasar gagal dimuat: {e}')

    xmin, ymin, xmax, ymax = bldg_wm.total_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    sm   = plt.cm.ScalarMappable(cmap=EDI_CMAP, norm=mcolors.Normalize(0,1))
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, aspect=30)
    cbar.set_label('Environmental Degradation Index (EDI)', fontweight='bold')
    cbar.set_ticks([0, CFG['thr_low'], CFG['thr_high'], 1])
    cbar.set_ticklabels(['0.0\n(Baik)', f'{CFG["thr_low"]}',
                          f'{CFG["thr_high"]}', '1.0\n(Terdegradasi)'])

    try:
        _north_arrow_scale(ax, bldg_wm)
    except Exception as e:
        log.warning(f'  Skala peta gagal dimuat: {e}')

    n_hi = int((buildings['env_class']=='Tinggi').sum())
    ax.set_title(
        'Environmental Degradation Index — Per Bangunan\n'
        f'Task 3 Output | {len(buildings):,} bangunan | '
        f'{n_hi} kelas Tinggi (paling terdegradasi)',
        fontsize=14, fontweight='bold', pad=12
    )
    ax.text(0.005, 0.005,
            'Sumber: MODIS AOD+LST+CF, Landsat 8/9 NDVI+NDBI, ESA WorldCover\n'
            f'CRS: Web Mercator (EPSG:3857) | Dibuat: {datetime.now():%Y-%m-%d}',
            transform=ax.transAxes, fontsize=7, color='#555', va='bottom')
    ax.set_axis_off()
    
    plt.tight_layout()

    path = OUT + 'maps/task3_edi_map.png'
    plt.savefig(path, dpi=300)
    plt.close()
    log.info(f'  → {path}')

def plot_indicator_maps(buildings: gpd.GeoDataFrame) -> None:
    """Panel peta 6 indikator lingkungan (satu subplot per indikator)."""
    log.info('[VIZ] Membuat panel peta 6 indikator...')

    inds = [(ind, meta) for ind, meta in IND_META.items()
            if ind in buildings.columns]
    n    = len(inds)
    cols = min(3, n)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    axes = np.array(axes).ravel()
    bldg_wm = buildings.to_crs('EPSG:3857')

    for i, (ind, meta) in enumerate(inds):
        ax = axes[i]
        
        bldg_wm.plot(column=ind, cmap=meta['cmap'], 
                      edgecolor='none', legend=False, ax=ax, zorder=2)
        
        if _opt['ctx']:
            try:
                _opt['ctx'].add_basemap(ax,
                    source=_opt['ctx'].providers.CartoDB.Positron,
                    zoom='auto', alpha=0.5, crs='EPSG:3857', zorder=1)
            except Exception: pass
            
        sm = plt.cm.ScalarMappable(
        
            cmap=meta['cmap'],
            norm=mcolors.Normalize(buildings[ind].min(), buildings[ind].max())
        )
        fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
        ax.set_title(meta['label'], fontsize=10, fontweight='bold')
        ax.text(0.5, -0.03,
                f'mean={buildings[ind].mean():.3f} | '
                f'std={buildings[ind].std():.3f}',
                transform=ax.transAxes, ha='center', fontsize=8, color='#555')
        ax.set_axis_off()

    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Peta Distribusi Indikator Lingkungan — Task 3',
                  fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = OUT + 'maps/task3_indicator_maps.png'
    plt.savefig(path, dpi=250, bbox_inches='tight')
    plt.close()
    log.info(f'  → {path}')

def plot_interactive_map(buildings: gpd.GeoDataFrame) -> None:
    """Peta interaktif Folium dengan layer-switcher per indikator."""
    if _opt['folium'] is None:
        log.warning('[VIZ] folium tidak terinstall → skip')
        return

    log.info('[VIZ] Membuat peta interaktif Folium (multi-layer)...')
    folium = _opt['folium']

    bldg_geo = buildings.to_crs('EPSG:4326').copy()
    bldg_geo['env_class'] = bldg_geo['env_class'].astype(str)
    centre   = [bldg_geo.geometry.centroid.y.mean(),
                 bldg_geo.geometry.centroid.x.mean()]
    m = folium.Map(location=centre, zoom_start=16, tiles='CartoDB positron')

    def _edi_color(v):
        if v >= CFG['thr_high']: return CLASS_COLORS['Tinggi']
        if v >= CFG['thr_low']:  return CLASS_COLORS['Sedang']
        return CLASS_COLORS['Rendah']

    tooltip_fields = ['bldg_id','env_deg_index','env_class','env_rank']
    tooltip_alias  = ['ID Bangunan','EDI','Kelas Degradasi','Ranking']
    for ind in IND_META:
        if ind in bldg_geo.columns:
            tooltip_fields.append(ind)
            tooltip_alias.append(IND_META[ind]['label'])

    folium.GeoJson(
        bldg_geo[tooltip_fields + ['geometry']].to_json(),
        name='EDI — Degradasi Lingkungan',
        style_function=lambda f: {
            'fillColor'  : _edi_color(f['properties']['env_deg_index']),
            'color'      : '#333',
            'weight'     : 0.4,
            'fillOpacity': 0.72,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=tooltip_fields, aliases=tooltip_alias,
            localize=True, sticky=True,
            style='font-size:11px;background:rgba(255,255,255,0.92);'
                  'border-radius:4px'
        ),
        highlight_function=lambda _: {'weight': 2, 'color': 'black'}
    ).add_to(m)

    top20 = bldg_geo.nsmallest(20, 'env_rank')
    for _, row in top20.iterrows():
        c = row.geometry.centroid
        folium.Marker(
            location=[c.y, c.x],
            icon=folium.Icon(color='red', icon='warning-sign', prefix='glyphicon'),
            tooltip=f"Rank #{row['env_rank']} | EDI={row['env_deg_index']:.3f}"
        ).add_to(m)

    legend = f"""
    <div style="position:fixed;bottom:25px;right:15px;z-index:9999;
                background:white;padding:12px 16px;border-radius:8px;
                border:1px solid #ccc;font-family:sans-serif;font-size:12px;
                box-shadow:2px 2px 6px rgba(0,0,0,0.15)">
      <b>Environmental Degradation</b><br><br>
      <span style="display:inline-block;width:14px;height:14px;
        background:{CLASS_COLORS['Tinggi']};margin-right:6px"></span>Tinggi ≥{CFG['thr_high']}<br>
      <span style="display:inline-block;width:14px;height:14px;
        background:{CLASS_COLORS['Sedang']};margin-right:6px"></span>Sedang {CFG['thr_low']}–{CFG['thr_high']}<br>
      <span style="display:inline-block;width:14px;height:14px;
        background:{CLASS_COLORS['Rendah']};margin-right:6px"></span>Rendah &lt;{CFG['thr_low']}<br>
      <hr style="margin:6px 0"><span style="font-size:10px;color:#666">
        ⚠ = Top 20 Paling Terdegradasi<br>
        Tinggi = prioritas manfaat solar PV</span>
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl().add_to(m)

    path = OUT + 'maps/task3_edi_interactive.html'
    m.save(path)
    log.info(f'  → {path}')

def plot_analysis_charts(buildings: gpd.GeoDataFrame,
                          pca_res: dict, km_res: dict,
                          gbr_res: dict, df_sens: pd.DataFrame) -> None:
    """
    Panel analisis 6-in-1:
      A. Distribusi EDI (histogram + KDE)
      B. Heatmap korelasi indikator
      C. PCA biplot (loading vectors + building scores)
      D. K-Means cluster profile (radar chart)
      E. Box plot indikator per kelas EDI
      F. Analisis sensitivitas skenario bobot
    """
    log.info('[VIZ] Panel analisis 6-panel...')

    fig = plt.figure(figsize=(20, 16))
    gs  = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[0, 2])
    ax_D = fig.add_subplot(gs[1, 0])
    ax_E = fig.add_subplot(gs[1, 1:])
    ax_F = fig.add_subplot(gs[2, :])
    fig.suptitle('Task 3 — Analisis Environmental Degradation Index (EDI)',
                  fontsize=15, fontweight='bold', y=1.002)

    norm_cols = [f'{ind}_norm' for ind in IND_META
                 if f'{ind}_norm' in buildings.columns]
    ind_labels = [ind for ind in IND_META if f'{ind}_norm' in buildings.columns]
    buildings['env_class'] = buildings['env_class'].astype(str)

    sns.histplot(buildings['env_deg_index'], bins=30, kde=True,
                  color='#d73027', ax=ax_A, edgecolor='white',
                  linewidth=0.5, alpha=0.70)
    ax_A.axvline(CFG['thr_low'],  color='#fc8d59', ls='--', lw=1.8)
    ax_A.axvline(CFG['thr_high'], color='#1a9850', ls='--', lw=1.8)
    ax_A.set_xlabel('Environmental Degradation Index', fontsize=10)
    ax_A.set_ylabel('Jumlah Bangunan', fontsize=10)
    ax_A.set_title('A. Distribusi EDI', fontweight='bold')
    for i, (cls, clr) in enumerate(CLASS_COLORS.items()):
        n   = (buildings['env_class'] == cls).sum()
        pct = n / len(buildings) * 100
        ax_A.text(0.98, 0.96-i*0.10, f'{cls}: {n} ({pct:.1f}%)',
                  transform=ax_A.transAxes, ha='right', va='top',
                  fontsize=8, color=clr, fontweight='bold')

    if len(norm_cols) >= 2:
        corr_data = buildings[norm_cols].copy()
        corr_data.columns = ind_labels
        corr_m  = corr_data.corr(method='pearson')
        mask    = np.triu(np.ones_like(corr_m, dtype=bool), k=1)
        sns.heatmap(corr_m, annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, vmin=-1, vmax=1, ax=ax_B,
                    linewidths=0.5, square=True,
                    cbar_kws={'shrink': 0.8},
                    annot_kws={'size': 9})
        ax_B.set_title('B. Korelasi Antar Indikator (Pearson)',
                        fontweight='bold')
        ax_B.tick_params(axis='x', rotation=30, labelsize=9)
        ax_B.tick_params(axis='y', rotation=0, labelsize=9)

    if pca_res and 'scores' in pca_res:
        scores   = pca_res['scores']
        loadings = pca_res['loadings']
        ev       = pca_res['explained_var']

        sc_x = scores[:, 0]
        sc_y = scores[:, 1] if scores.shape[1] > 1 else np.zeros(len(scores))
        edi_c = buildings['env_deg_index'].values

        sc = ax_C.scatter(sc_x, sc_y, c=edi_c, cmap=EDI_CMAP,
                           s=12, alpha=0.6, edgecolors='none', vmin=0, vmax=1)
        fig.colorbar(sc, ax=ax_C, fraction=0.04, pad=0.02).set_label('EDI', fontsize=8)

        scale = 2.0
        for j, ind_n in enumerate(loadings.index):
            lx = loadings['PC1'].iloc[j] * scale
            ly = (loadings['PC2'].iloc[j] if 'PC2' in loadings.columns
                  else 0) * scale
            ax_C.annotate('', xy=(lx, ly), xytext=(0, 0),
                           arrowprops=dict(arrowstyle='->', color='black',
                                           lw=1.5, alpha=0.85))
            ax_C.text(lx*1.1, ly*1.1, ind_n, fontsize=8,
                       ha='center', va='center', fontweight='bold')

        ax_C.axhline(0, color='gray', lw=0.5, ls='--')
        ax_C.axvline(0, color='gray', lw=0.5, ls='--')
        ax_C.set_xlabel(f'PC1 ({ev[0]:.1%})', fontsize=9)
        ylab = f'PC2 ({ev[1]:.1%})' if len(ev) > 1 else 'PC2'
        ax_C.set_ylabel(ylab, fontsize=9)
        ax_C.set_title('C. PCA Biplot (Loadings + Building Scores)',
                        fontweight='bold')

    if km_res and 'centers' in km_res:
        centers = km_res['centers']
        x_pos   = np.arange(len(centers.columns))
        colors_k = plt.cm.tab10(np.linspace(0, 0.9, len(centers)))
        for i, (_, row) in enumerate(centers.iterrows()):
            ax_D.plot(x_pos, row.values, 'o-', color=colors_k[i],
                       label=f'Cluster {i} (n={int((buildings["km_cluster"]==i).sum())})',
                       linewidth=1.8, markersize=6)
        ax_D.set_xticks(x_pos)
        ax_D.set_xticklabels(centers.columns, rotation=30, fontsize=8)
        ax_D.set_ylabel('Nilai Rata-rata', fontsize=9)
        ax_D.set_ylim(-0.05, 1.1)
        ax_D.set_title(f'D. K-Means Cluster Profile (k={km_res["best_k"]}, '
                        f'sil={km_res["best_sil"]:.3f})', fontweight='bold')
        ax_D.legend(fontsize=7, loc='upper right')

    if len(norm_cols) >= 1:
        melt_cols = ['env_class'] + ind_labels
        bldg_melt = buildings[[c for c in melt_cols
                                if c in buildings.columns]].copy()
       
        for ind in ind_labels:
            if ind in bldg_melt.columns:
                bldg_melt[ind] = buildings[ind].values
        bldg_long = bldg_melt.melt(id_vars='env_class',
                                     value_vars=[i for i in ind_labels
                                                 if i in bldg_melt.columns],
                                     var_name='Indikator',
                                     value_name='Nilai')
        sns.boxplot(data=bldg_long, x='Indikator', y='Nilai',
                    hue='env_class', hue_order=CLASS_ORDER,
                    palette=CLASS_COLORS,
                    showfliers=False, linewidth=0.9, ax=ax_E)
        ax_E.set_xlabel('', fontsize=9)
        ax_E.set_ylabel('Nilai Indikator (raw)', fontsize=9)
        ax_E.set_title('E. Distribusi Indikator per Kelas EDI',
                        fontweight='bold')
        ax_E.tick_params(axis='x', rotation=20, labelsize=9)
        ax_E.legend(title='Kelas EDI', fontsize=8, title_fontsize=8,
                    loc='upper right')

    if not df_sens.empty:
        x_scen = range(len(df_sens))
        ax_F.bar(x_scen, df_sens['EDI_mean'], alpha=0.5,
                  label='EDI mean', color='#4575b4', width=0.35)
        ax_F.bar([x+0.35 for x in x_scen],
                  df_sens['stable_top20_%'] / 100,
                  alpha=0.7, label='Stable top-20 (%/100)',
                  color='#d73027', width=0.35)
        ax_F2 = ax_F.twinx()
        ax_F2.plot(x_scen, df_sens['Spearman_rho'], 'k^--',
                   label='Spearman ρ vs Baseline', linewidth=2, markersize=7)
        ax_F2.set_ylabel('Spearman ρ', fontsize=9)
        ax_F2.set_ylim(0, 1.1)
        ax_F.set_xticks(list(x_scen))
        ax_F.set_xticklabels(df_sens['Skenario'], rotation=15, fontsize=9)
        ax_F.set_ylabel('EDI mean | Stable top-20', fontsize=9)
        ax_F.set_title('F. Analisis Sensitivitas Skenario Bobot',
                        fontweight='bold')
        lines1, lbl1 = ax_F.get_legend_handles_labels()
        lines2, lbl2 = ax_F2.get_legend_handles_labels()
        ax_F.legend(lines1+lines2, lbl1+lbl2, fontsize=8, loc='upper left')

    if gbr_res:
        note = (f'[ML] GBR {CFG["cv_k"]}-Fold CV → '
                f'R²={gbr_res.get("r2",0):.4f} | MAE={gbr_res.get("mae",0):.4f}')
        if gbr_res.get('xgb_metrics'):
            xm = gbr_res['xgb_metrics']
            note += f' || XGBoost R²={xm["r2"]:.4f} | MAE={xm["mae"]:.4f}'
        if pca_res:
            ev = pca_res.get('explained_var', [])
            note += f' | PCA PC1+PC2={sum(ev[:2]):.1%}'
        if km_res:
            note += (f' | KMeans k={km_res["best_k"]} '
                     f'sil={km_res["best_sil"]:.4f}')
        fig.text(0.01, -0.008, note, fontsize=8, color='#666')

    path = OUT + 'figures/task3_analysis_charts.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    log.info(f'  → {path}')

def plot_sensitivity_detail(df_sens: pd.DataFrame) -> None:
    """Grafik sensitivitas bobot yang lebih detail: radar + heatmap."""
    if df_sens.empty:
        return
    log.info('[VIZ] Detail sensitivitas...')

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Analisis Sensitivitas Bobot EDI', fontsize=13, fontweight='bold')

    w_cols = [c for c in df_sens.columns if c.startswith('w_')]
    W      = df_sens.set_index('Skenario')[w_cols].T
    W.index = [c.replace('w_','') for c in W.index]
    sns.heatmap(W, annot=True, fmt='.2f', cmap='Blues',
                linewidths=0.5, vmin=0, vmax=0.5, ax=axes[0],
                annot_kws={'size': 10})
    axes[0].set_title('Matriks Bobot per Skenario', fontweight='bold')
    axes[0].set_xlabel('Skenario', fontsize=10)
    axes[0].set_ylabel('Indikator', fontsize=10)

    x    = np.arange(len(df_sens))
    ax_b = axes[1]
    bars = ax_b.barh(x, df_sens['stable_top20_%'], color='#4575b4',
                      alpha=0.75, edgecolor='white')
    ax_b.set_yticks(x)
    ax_b.set_yticklabels(df_sens['Skenario'], fontsize=10)
    ax_b.set_xlabel('Top-20 Stability (%)', fontsize=10)
    ax_b.set_title('Stabilitas Ranking Top-20 per Skenario',
                    fontweight='bold')
    ax_b.axvline(80, color='red', ls='--', lw=1.5, label='80% threshold')
    for i, (bar, rho) in enumerate(zip(bars, df_sens['Spearman_rho'])):
        ax_b.text(bar.get_width()+1, bar.get_y()+bar.get_height()/2,
                  f'ρ={rho:.3f}', va='center', fontsize=9)
    ax_b.legend(fontsize=9)
    ax_b.set_xlim(0, 115)

    plt.tight_layout()
    path = OUT + 'figures/task3_sensitivity.png'
    plt.savefig(path, dpi=300)
    plt.close()
    log.info(f'  → {path}')

def export_results(buildings: gpd.GeoDataFrame,
                    pca_res: dict, km_res: dict,
                    gbr_res: dict) -> None:
    """Ekspor GeoJSON, CSV, dan ringkasan JSON."""
    log.info('[EXPORT] Mengekspor hasil...')

    keep = ['bldg_id', 'geometry', 'area_m2'] + \
           list(IND_META.keys()) + \
           [f'{i}_norm' for i in IND_META] + \
           ['EDI_wlc', 'EDI_pca', 'EDI_gbr', 'env_deg_index',
            'env_class', 'env_rank', 'km_cluster']
    out_cols = [c for c in keep if c in buildings.columns]
    out_gdf  = buildings[out_cols].copy()
    out_gdf['env_class'] = out_gdf['env_class'].astype(str)

    geo_out = out_gdf.to_crs('EPSG:4326')
    gj_path = OUT + 'data/env_degradation_index.geojson'
    geo_out.to_file(gj_path, driver='GeoJSON')
    log.info(f'  → {gj_path}')

    csv_path = OUT + 'data/env_summary.csv'
    geo_out.drop(columns='geometry').to_csv(csv_path, index=False)
    log.info(f'  → {csv_path}')

    top_path = OUT + 'data/top20_degraded.csv'
    geo_out.drop(columns='geometry').nsmallest(20,'env_rank').to_csv(top_path, index=False)
    log.info(f'  → {top_path}')

    n_hi = int((buildings['env_class']=='Tinggi').sum())
    n_md = int((buildings['env_class']=='Sedang').sum())
    n_lo = int((buildings['env_class']=='Rendah').sum())

    summary = {
        'n_buildings'        : int(len(buildings)),
        'mean_edi'           : round(float(buildings['env_deg_index'].mean()), 4),
        'std_edi'            : round(float(buildings['env_deg_index'].std()), 4),
        'count_Tinggi'       : n_hi,
        'count_Sedang'       : n_md,
        'count_Rendah'       : n_lo,
        'pct_Tinggi'         : round(n_hi/len(buildings)*100, 1),
        'indicator_means_raw': {
            ind: round(float(buildings[ind].mean()), 4)
            for ind in IND_META if ind in buildings.columns
        },
        'edi_thresholds'     : {'low': CFG['thr_low'], 'high': CFG['thr_high']},
        'weights_baseline'   : CFG['weights'],
        'ml_metrics'         : {
            'pca_explained_var_pc1'  : round(pca_res.get('explained_var', [0])[0], 4)
                                       if pca_res else None,
            'pca_cum_var_pc1_pc2'    : (round(pca_res['cumulative_var'][1], 4)
                                        if pca_res and len(pca_res.get('cumulative_var',[]))>1
                                        else None),
            'kmeans_best_k'          : km_res.get('best_k') if km_res else None,
            'kmeans_silhouette'      : round(km_res.get('best_sil',0), 4) if km_res else None,
            'gbr_r2_cv'              : round(gbr_res.get('r2',0), 4) if gbr_res else None,
            'gbr_mae_cv'             : round(gbr_res.get('mae',0), 4) if gbr_res else None,
            'gbr_n_samples'          : gbr_res.get('n_samples') if gbr_res else None,
            'gbr_cv_k'               : gbr_res.get('cv_k') if gbr_res else None,
            'gbr_justification'      : gbr_res.get('justification') if gbr_res else None,
            'xgb_r2_cv'              : (round(gbr_res.get('xgb_metrics',{}).get('r2',0),4)
                                        if gbr_res and gbr_res.get('xgb_metrics') else None),
        },
        'data_sources'       : {
            'AOD'  : 'MODIS MCD19A2 MAIAC (1 km, annual mean)',
            'LST'  : 'MODIS MOD11A2 Terra 8-day composite (1 km, annual mean, °C)',
            'NDVI' : 'Landsat 8/9 C2L2 SR median composite (30 m, annual)',
            'NDBI' : 'Landsat 8/9 C2L2 SR median composite (30 m, annual)',
            'CF'   : 'MODIS MOD08_M3 (monthly, annual mean)',
            'LC'   : 'ESA WorldCover v200 (10 m, 2021)',
        },
        'methodological_note': (
            'EDI tinggi = kondisi lingkungan lebih terdegradasi. '
            'NDVI diinversi (1-NDVI_norm) karena vegetasi lebih lebat = '
            'kualitas lingkungan lebih baik. '
            'EDI final = 60% WLC + 30% GBR smooth + 10% PCA composite.'
        ),
        'generated_at'       : datetime.now().isoformat(),
    }

    json_path = OUT + 'data/task3_summary.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log.info('\n' + '═'*60)
    log.info('  RINGKASAN HASIL TASK 3')
    log.info('═'*60)
    log.info(f'  Bangunan dianalisis : {len(buildings):,}')
    log.info(f'  EDI mean            : {summary["mean_edi"]:.4f} '
             f'± {summary["std_edi"]:.4f}')
    log.info(f'  Kelas Tinggi (prioritas): {n_hi:,} bangunan ({summary["pct_Tinggi"]:.1f}%)')
    log.info(f'  Kelas Sedang        : {n_md:,} bangunan')
    log.info(f'  Kelas Rendah        : {n_lo:,} bangunan')
    if gbr_res:
        log.info(f'  GBR R² (CV)         : {summary["ml_metrics"]["gbr_r2_cv"]}')
        log.info(f'  GBR MAE (CV)        : {summary["ml_metrics"]["gbr_mae_cv"]}')
    if km_res:
        log.info(f'  KMeans k optimal    : {km_res["best_k"]} '
                 f'(sil={km_res["best_sil"]:.4f})')
    log.info('═'*60)
    log.info(f'  JSON summary → {json_path}')

def main() -> gpd.GeoDataFrame:
    t_start = datetime.now()
    log.info('=' * 68)
    log.info('  TASK 3 — ENVIRONMENTAL DEGRADATION INDEX')
    log.info('  Technical Assessment · Geospatial Data Engineer')
    log.info(f'  Dimulai: {t_start:%Y-%m-%d %H:%M:%S}')
    log.info('=' * 68)

    buildings = load_buildings(CFG['bldg_path'])
    sat_data  = load_all_satellite(buildings)

    buildings = assign_indicators_to_buildings(buildings, sat_data)

    buildings = normalize_indicators(buildings)

    buildings = calculate_wlc_edi(buildings, CFG['weights'], 'EDI_wlc')

    pca_res = run_pca_analysis(buildings)

    km_res  = run_kmeans_clustering(buildings, pca_res)

    gbr_res = smooth_edi_with_gbr(buildings, wlc_col='EDI_wlc')

    buildings = calculate_final_edi(buildings, use_gbr=bool(gbr_res))

    buildings = classify_edi(buildings)

    df_sens = sensitivity_analysis(buildings)

    plot_edi_map(buildings)
    plot_indicator_maps(buildings)
    plot_interactive_map(buildings)
    plot_analysis_charts(buildings, pca_res, km_res, gbr_res, df_sens)
    plot_sensitivity_detail(df_sens)

    export_results(buildings, pca_res, km_res, gbr_res)

    elapsed = (datetime.now() - t_start).seconds
    log.info(f'\n✓ Task 3 selesai dalam {elapsed} detik')
    log.info(f'  Output tersimpan di: {OUT}')
    return buildings

if __name__ == '__main__':
    result = main()
