import os, sys, json, math, logging, warnings, argparse
from pathlib    import Path
from datetime   import datetime
from copy       import deepcopy

import numpy           as np
import pandas          as pd
import geopandas       as gpd
from   shapely.geometry import Point
from   scipy.spatial    import cKDTree
from   scipy.stats      import spearmanr, pearsonr, kendalltau
from   scipy.linalg     import eig as scipy_eig

from   sklearn.preprocessing    import MinMaxScaler
from   sklearn.ensemble         import RandomForestRegressor, GradientBoostingRegressor
from   sklearn.model_selection  import KFold, cross_val_predict
from   sklearn.metrics          import r2_score, mean_absolute_error
from   sklearn.inspection       import permutation_importance

import matplotlib
import matplotlib.pyplot       as plt
import matplotlib.patches      as mpatches
import matplotlib.colors       as mcolors
import matplotlib.ticker       as ticker
import matplotlib.patheffects  as pe
from   matplotlib.colors        import LinearSegmentedColormap, BoundaryNorm
from   matplotlib.gridspec      import GridSpec
import seaborn                  as sns

# Optional packages
_opt = {}
for _pkg, _key in [('contextily','ctx'), ('folium','folium'),
                    ('PIL.Image','pil'), ('xgboost','xgb')]:
    try:
        module = _pkg.split('.')[0]
        _opt[_key] = __import__(module)
    except ImportError:
        _opt[_key] = None

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
    'axes.labelsize'    : 10,
    'xtick.labelsize'   : 9,
    'ytick.labelsize'   : 9,
})

# ── Path input (output dari Task 1–3) ────────────────────────────────────────
INPUT_PATHS = dict(
    task1 = 'output/task1/data/solar_pv_index.geojson',
    task2 = 'output/task2/data/economic_value_index.geojson',
    task3 = 'output/task3/data/env_degradation_index.geojson',
    # Fallback CSV jika GeoJSON tidak tersedia
    task1_csv = 'output/task1/data/solar_pv_summary.csv',
    task2_csv = 'output/task2/data/economic_value_summary.csv',
    task3_csv = 'output/task3/data/env_summary.csv',
)

OUT = 'output/task4/'
for _s in ['', 'maps/', 'figures/', 'data/', 'report/']:
    Path(OUT + _s).mkdir(parents=True, exist_ok=True)

# ── CRS ──────────────────────────────────────────────────────────────────────
GEO_CRS  = 'EPSG:4326'
PROJ_CRS = 'EPSG:32648'   # ← Sesuaikan dengan zona UTM area studi

# ── Nama kolom indeks dari setiap task ───────────────────────────────────────
# Format: (kolom_utama, daftar_alternatif_jika_tidak_ada)
COL_MAP = {
    'PV'   : ('solar_pv_index',   ['pv_index','pv_score','solar_index']),
    'Econ' : ('econ_value_index', ['economic_index','econ_index','economic_value']),
    'Env'  : ('env_deg_index',    ['env_index','environmental_index','edi','env_degradation_index']),
}

# ── Bobot WLC — 6 skenario ───────────────────────────────────────────────────
WLC_SCENARIOS = {
    'Baseline'         : {'PV': 0.40, 'Econ': 0.35, 'Env': 0.25},
    'PV_Dominant'      : {'PV': 0.60, 'Econ': 0.25, 'Env': 0.15},
    'Economics_Focus'  : {'PV': 0.25, 'Econ': 0.55, 'Env': 0.20},
    'Environment_Focus': {'PV': 0.25, 'Econ': 0.25, 'Env': 0.50},
    'Equal_Weight'     : {'PV': 1/3,  'Econ': 1/3,  'Env': 1/3 },
    'Tech_Viability'   : {'PV': 0.50, 'Econ': 0.35, 'Env': 0.15},
}

# ── Matriks AHP — 5 skenario (skala Saaty 1–9) ───────────────────────────────
# Baris/kolom = [PV, Econ, Env]
# aᵢⱼ > 1: baris i lebih penting dari kolom j
AHP_MATRICES = {
    'AHP_Balanced': [       # PV sedikit lebih penting
        [1,   2,   3  ],
        [1/2, 1,   2  ],
        [1/3, 1/2, 1  ],
    ],
    'AHP_PV_Focus': [       # PV sangat dominan
        [1,   3,   5  ],
        [1/3, 1,   2  ],
        [1/5, 1/2, 1  ],
    ],
    'AHP_Econ_Focus': [     # Economics dominan
        [1,   1/2, 2  ],
        [2,   1,   3  ],
        [1/2, 1/3, 1  ],
    ],
    'AHP_Env_Focus': [      # Environment dominan
        [1,   2,   1/3],
        [1/2, 1,   1/5],
        [3,   5,   1  ],
    ],
    'AHP_Equal': [          # Semua sama penting
        [1,   1,   1  ],
        [1,   1,   1  ],
        [1,   1,   1  ],
    ],
}

# ── Parameter ML ─────────────────────────────────────────────────────────────
ML_CFG = dict(
    rf_n        = 300,
    rf_seed     = 42,
    gbr_n       = 200,
    gbr_lr      = 0.05,
    gbr_depth   = 4,
    cv_k        = 5,
    min_samples = 20,
)

# ── Monte Carlo ───────────────────────────────────────────────────────────────
MC_N_SIM  = 10_000   # jumlah simulasi
MC_SEED   = 42
MC_ALPHA  = [1, 1, 1]   # Dirichlet hyperparameter (uniform simplex)

# ── Klasifikasi prioritas ─────────────────────────────────────────────────────
THR_LOW  = 0.33
THR_HIGH = 0.67
TOP_N    = 20    # jumlah bangunan prioritas tertinggi yang dilaporkan

# ── Warna ────────────────────────────────────────────────────────────────────
PRIORITY_COLORS = {
    'Tinggi' : '#d73027',   # merah — prioritas tertinggi
    'Sedang' : '#fdae61',   # oranye
    'Rendah' : '#4dac26',   # hijau — prioritas rendah
}
PRIORITY_ORDER = ['Rendah', 'Sedang', 'Tinggi']
PRIORITY_CMAP  = LinearSegmentedColormap.from_list(
    'priority', ['#4dac26','#ffffbf','#d73027'], N=256
)
CRITERIA_COLORS = {'PV': '#1d6ba0', 'Econ': '#e6850e', 'Env': '#2ca02c'}



logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s │ %(levelname)-7s │ %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(OUT + 'task4.log', mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger('Task4')



def _find_col(df: pd.DataFrame, key: str) -> str | None:
    """Cari kolom indeks dengan fallback ke nama alternatif."""
    primary, alternatives = COL_MAP[key]
    for candidate in [primary] + alternatives:
        if candidate in df.columns:
            return candidate
    # Coba partial match (kolom yang mengandung kata kunci)
    kw = key.lower()
    for c in df.columns:
        if kw in c.lower() and 'class' not in c.lower() and 'rank' not in c.lower():
            return c
    return None


def _load_geojson_or_csv(geojson_path: str, csv_path: str,
                           label: str) -> gpd.GeoDataFrame | None:
    """Muat GeoJSON (prioritas) atau CSV + geometry placeholder."""
    if Path(geojson_path).exists():
        gdf = gpd.read_file(geojson_path)
        log.info(f'  [{label}] GeoJSON ← {geojson_path} ({len(gdf):,} baris)')
        return gdf

    if Path(csv_path).exists():
        df  = pd.read_csv(csv_path)
        # Buat geometry dummy jika tidak ada kolom geometri
        if 'geometry' not in df.columns:
            df['geometry'] = [Point(0, 0)] * len(df)
        gdf = gpd.GeoDataFrame(df, crs=GEO_CRS)
        log.warning(f'  [{label}] GeoJSON tidak ada → CSV ← {csv_path}')
        return gdf

    log.warning(f'  [{label}] Tidak ada data ditemukan')
    return None


def _generate_demo_data(n: int = 200) -> gpd.GeoDataFrame:
    """
    Buat data demonstrasi sintetis bila output Task 1–3 belum tersedia.
    Menyimulasikan distribusi realistis 3 indeks dengan korelasi spasial.
    """
    log.warning('[DEMO] Membuat data sintetis (200 bangunan) untuk demonstrasi...')
    rng   = np.random.RandomState(42)
    theta = rng.uniform(0, 2*np.pi, n)
    r     = rng.uniform(0, 0.005, n)
    cx, cy = 106.82, -6.18   # Jakarta sebagai contoh
    lons  = cx + r * np.cos(theta)
    lats  = cy + r * np.sin(theta)

    # Simulasikan korelasi spasial ringan: PV ↔ Econ positif lemah
    pv_base   = rng.beta(2, 2, n)
    econ_base = 0.4*pv_base + 0.6*rng.beta(2, 3, n)
    env_base  = rng.beta(2, 2, n)
    econ_base = np.clip(econ_base, 0, 1)

    from shapely.geometry import Point as SPoint
    rows = []
    for i in range(n):
        rows.append({
            'bldg_id'         : i,
            'geometry'        : SPoint(lons[i], lats[i]).buffer(0.0001, cap_style=3),
            'area_m2'         : rng.uniform(80, 800),
            'height'          : rng.uniform(4, 30),
            'solar_pv_index'  : float(pv_base[i]),
            'pv_class'        : 'Tinggi' if pv_base[i]>0.67 else ('Sedang' if pv_base[i]>0.33 else 'Rendah'),
            'kWp'             : float(pv_base[i] * rng.uniform(10, 80)),
            'E_annual_kWh'    : float(pv_base[i] * rng.uniform(5000, 80000)),
            'econ_value_index': float(econ_base[i]),
            'econ_class'      : 'Tinggi' if econ_base[i]>0.67 else ('Sedang' if econ_base[i]>0.33 else 'Rendah'),
            'env_deg_index'   : float(env_base[i]),
            'env_class'       : 'Tinggi' if env_base[i]>0.67 else ('Sedang' if env_base[i]>0.33 else 'Rendah'),
        })
    gdf = gpd.GeoDataFrame(rows, crs=GEO_CRS)
    log.info(f'  [DEMO] {len(gdf):,} bangunan sintetis berhasil dibuat')
    return gdf


def load_and_merge(demo: bool = False) -> gpd.GeoDataFrame:
    """
    Muat output Task 1–3, normalisasi nama kolom, dan gabungkan ke satu GeoDataFrame.
    Strategi join: spatial join berdasarkan centroid jika bldg_id tidak cocok.
    """
    log.info('[LOAD] Memuat output Task 1–3...')

    if demo:
        return _generate_demo_data()

    gdf1 = _load_geojson_or_csv(INPUT_PATHS['task1'], INPUT_PATHS['task1_csv'], 'Task1')
    gdf2 = _load_geojson_or_csv(INPUT_PATHS['task2'], INPUT_PATHS['task2_csv'], 'Task2')
    gdf3 = _load_geojson_or_csv(INPUT_PATHS['task3'], INPUT_PATHS['task3_csv'], 'Task3')

    # Jika semua gagal → demo
    if all(g is None for g in [gdf1, gdf2, gdf3]):
        log.warning('[LOAD] Semua output Task hilang → mode demo')
        return _generate_demo_data()

    # Gunakan Task 1 sebagai base (memiliki geometri bangunan)
    base = gdf1 if gdf1 is not None else (gdf2 if gdf2 is not None else gdf3)
    if base.crs is None:
        base = base.set_crs(GEO_CRS)
    base = base.to_crs(GEO_CRS)

    # Temukan dan normalisasi kolom indeks dari tiap task
    def _extract(gdf, key, task_label):
        if gdf is None:
            return None, None
        col = _find_col(gdf, key)
        if col is None:
            log.warning(f'  [{task_label}] Kolom {key} tidak ditemukan')
            return None, None
        g = gdf.to_crs(GEO_CRS) if gdf.crs else gdf
        return g, col

    g1, c1 = _extract(gdf1, 'PV',   'Task1')
    g2, c2 = _extract(gdf2, 'Econ', 'Task2')
    g3, c3 = _extract(gdf3, 'Env',  'Task3')

    # ── Join via bldg_id (utama) ──────────────────────────────────────────────
    def _join_by_id(base_df, src_df, src_col, new_col):
        if src_df is None or src_col is None:
            base_df[new_col] = np.nan
            return base_df
        if 'bldg_id' in base_df.columns and 'bldg_id' in src_df.columns:
            lkp = src_df.set_index('bldg_id')[src_col]
            base_df[new_col] = base_df['bldg_id'].map(lkp)
            n_ok = base_df[new_col].notna().sum()
            log.info(f'  Join by bldg_id → {n_ok:,}/{len(base_df):,} cocok ({new_col})')
            if n_ok > 0:
                return base_df

        # Fallback: spatial nearest-neighbour join
        log.info(f'  Fallback → Spatial NN join ({new_col})')
        bpts = np.column_stack([
            base_df.geometry.centroid.x.values,
            base_df.geometry.centroid.y.values,
        ])
        spts = np.column_stack([
            src_df.geometry.centroid.x.values,
            src_df.geometry.centroid.y.values,
        ])
        tree = cKDTree(spts)
        _, idx = tree.query(bpts, k=1)
        base_df[new_col] = src_df[src_col].values[idx]
        return base_df

    base = _join_by_id(base, g1, c1, 'solar_pv_index')
    base = _join_by_id(base, g2, c2, 'econ_value_index')
    base = _join_by_id(base, g3, c3, 'env_deg_index')

    # Isi NaN residual dengan median
    for col in ['solar_pv_index', 'econ_value_index', 'env_deg_index']:
        n_nan = base[col].isna().sum()
        if n_nan:
            med = base[col].median()
            if np.isnan(med):
                med = 0.5
            base[col] = base[col].fillna(med)
            log.warning(f'  {n_nan} NaN di {col} → diisi median ({med:.4f})')

    # Clip ke [0, 1]
    for col in ['solar_pv_index', 'econ_value_index', 'env_deg_index']:
        base[col] = base[col].clip(0, 1)

    # Pastikan bldg_id ada
    if 'bldg_id' not in base.columns:
        base['bldg_id'] = base.index.astype(int)

    log.info(f'[LOAD] ✓ {len(base):,} bangunan merged | '
             f'PV={base["solar_pv_index"].mean():.3f} | '
             f'Econ={base["econ_value_index"].mean():.3f} | '
             f'Env={base["env_deg_index"].mean():.3f}')
    return base.reset_index(drop=True)



class AHPAnalyzer:
    """
    Implementasi penuh Analytic Hierarchy Process (Saaty, 1980).

    Langkah:
      1. Validasi matriks (resiprokal + positif)
      2. Eigenvector method → bobot prioritas
      3. Hitung λmax, Consistency Index (CI), Consistency Ratio (CR)
      4. CR < 0.10 → konsisten (acceptable)

    Referensi:
      Saaty, T.L. (1980). The Analytic Hierarchy Process.
      McGraw-Hill, New York.
    """
    RI = {1:0.00, 2:0.00, 3:0.58, 4:0.90, 5:1.12,
          6:1.24,  7:1.32, 8:1.41, 9:1.45, 10:1.49}

    def __init__(self, criteria: list[str], matrix: list[list]):
        self.criteria = criteria
        self.n        = len(criteria)
        self.A        = np.array(matrix, dtype=float)

    def _validate(self):
        assert self.A.shape == (self.n, self.n), 'Ukuran matriks tidak cocok'
        for i in range(self.n):
            assert self.A[i, i] == 1.0, f'Diagonal [{i},{i}] bukan 1'
            for j in range(self.n):
                assert abs(self.A[i,j] * self.A[j,i] - 1.0) < 1e-5, \
                    f'Matriks tidak resiprokal pada [{i},{j}]'
        assert np.all(self.A > 0), 'Semua elemen harus positif'

    def compute(self) -> dict:
        """
        Eigenvalue method (eksak):
          w = eigenvector terkait eigenvalue terbesar (λmax)
          CI = (λmax - n) / (n - 1)
          CR = CI / RI[n]
        """
        self._validate()
        eigenvalues, eigenvectors = np.linalg.eig(self.A)
        idx     = np.argmax(eigenvalues.real)
        w_raw   = eigenvectors[:, idx].real
        w       = np.abs(w_raw) / np.abs(w_raw).sum()

        lambda_max = float(eigenvalues[idx].real)
        CI  = (lambda_max - self.n) / max(self.n - 1, 1)
        RI  = self.RI.get(self.n, 1.49)
        CR  = CI / RI if RI > 0 else 0.0

        # Geometric mean approximation (cross-check)
        gm      = np.prod(self.A, axis=1) ** (1.0 / self.n)
        w_gm    = gm / gm.sum()

        result = {
            'criteria'    : self.criteria,
            'weights'     : dict(zip(self.criteria, w)),
            'weights_arr' : w,
            'weights_gm'  : dict(zip(self.criteria, w_gm)),
            'lambda_max'  : lambda_max,
            'CI'          : CI,
            'CR'          : CR,
            'RI'          : RI,
            'consistent'  : CR < 0.10,
            'matrix'      : self.A.tolist(),
        }
        return result


def run_all_ahp(buildings: pd.DataFrame) -> dict:
    """Jalankan semua skenario AHP dan kembalikan ringkasan."""
    log.info('[AHP] Menjalankan semua skenario AHP...')
    criteria = ['PV', 'Econ', 'Env']
    results  = {}

    for name, matrix in AHP_MATRICES.items():
        ahp = AHPAnalyzer(criteria, matrix)
        r   = ahp.compute()
        results[name] = r
        cr_str = f'CR={r["CR"]:.4f} {"✓" if r["consistent"] else "✗ (>0.10!)"}'
        log.info(f'  {name:<22}: '
                 f'w=[{", ".join([f"{k}={v:.3f}" for k,v in r["weights"].items()])}] '
                 f'| {cr_str}')

    return results



def _normalize_weights(w: dict) -> dict:
    """Normalisasi bobot agar Σwᵢ = 1."""
    total = sum(w.values())
    return {k: v/total for k, v in w.items()}


def wlc_score(buildings: gpd.GeoDataFrame,
               weights: dict,
               col_out: str = 'score_wlc') -> gpd.GeoDataFrame:
    """
    Weighted Linear Combination:
      S = w₁·PV + w₂·Econ + w₃·Env

    Semua indeks sudah [0–1] → tidak perlu normalisasi ulang.
    """
    w = _normalize_weights(weights)
    s = (w.get('PV',   0) * buildings['solar_pv_index'] +
         w.get('Econ', 0) * buildings['econ_value_index'] +
         w.get('Env',  0) * buildings['env_deg_index'])
    buildings[col_out] = s.clip(0, 1)
    return buildings


def topsis_score(buildings: gpd.GeoDataFrame,
                  weights: dict,
                  col_out: str = 'score_topsis') -> gpd.GeoDataFrame:
    """
    TOPSIS — Technique for Order Preference by Similarity to Ideal Solution.
    (Hwang & Yoon, 1981)

    Langkah:
      1. Normalisasi vektoral: rᵢⱼ = xᵢⱼ / √Σxᵢⱼ²
      2. Weighted: vᵢⱼ = wⱼ × rᵢⱼ
      3. Ideal terbaik A⁺ = {max vᵢⱼ} (semua kriteria beneficial)
         Ideal terburuk A⁻ = {min vᵢⱼ}
      4. Jarak Euclidean ke A⁺ dan A⁻
      5. Koefisien kedekatan: Cᵢ = d⁻ᵢ / (d⁺ᵢ + d⁻ᵢ)

    Seluruh kriteria bersifat BENEFICIAL (tinggi = lebih baik/prioritas).
    """
    w   = _normalize_weights(weights)
    cols = ['solar_pv_index', 'econ_value_index', 'env_deg_index']
    keys = ['PV', 'Econ', 'Env']
    X    = buildings[cols].values.astype(float)
    W    = np.array([w.get(k, 0) for k in keys])

    # Step 1–2: weighted normalized matrix
    norms = np.sqrt((X**2).sum(axis=0))
    norms = np.where(norms < 1e-10, 1.0, norms)
    V     = (X / norms) * W

    # Step 3: ideal solutions (semua beneficial)
    A_pos = V.max(axis=0)
    A_neg = V.min(axis=0)

    # Step 4: Euclidean distances
    d_pos = np.sqrt(((V - A_pos)**2).sum(axis=1))
    d_neg = np.sqrt(((V - A_neg)**2).sum(axis=1))

    # Step 5: closeness coefficient [0, 1]
    C = d_neg / (d_pos + d_neg + 1e-12)
    buildings[col_out] = MinMaxScaler().fit_transform(C.reshape(-1,1)).ravel()
    return buildings


def run_all_wlc(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Jalankan semua skenario WLC + TOPSIS dan simpan tiap kolom."""
    log.info('[WLC/TOPSIS] Menjalankan semua skenario scoring...')
    for name, w in WLC_SCENARIOS.items():
        buildings = wlc_score(buildings, w, f'wlc_{name}')
        log.info(f'  WLC {name:<22}: mean={buildings[f"wlc_{name}"].mean():.4f}')

    # TOPSIS dengan bobot Baseline
    buildings = topsis_score(buildings, WLC_SCENARIOS['Baseline'], 'score_topsis')
    log.info(f'  TOPSIS (Baseline)      : mean={buildings["score_topsis"].mean():.4f}')
    return buildings




def _engineer_features(buildings: gpd.GeoDataFrame) -> np.ndarray:
    """
    Rekayasa fitur untuk ML ensemble:
      - 3 indeks dasar (PV, Econ, Env)
      - 3 interaksi pairwise (PV×Econ, PV×Env, Econ×Env)
      - Interaksi tiga-arah (PV×Econ×Env)
      - Rata-rata, minimum, maksimum
      - Koordinat spasial (proxy untuk autokorelasi spasial)
    """
    pv   = buildings['solar_pv_index'].values
    ec   = buildings['econ_value_index'].values
    ev   = buildings['env_deg_index'].values

    # Koordinat proyeksi (jika tersedia)
    try:
        bldg_p = buildings.to_crs(PROJ_CRS)
        cx = MinMaxScaler().fit_transform(
                bldg_p.geometry.centroid.x.values.reshape(-1,1)).ravel()
        cy = MinMaxScaler().fit_transform(
                bldg_p.geometry.centroid.y.values.reshape(-1,1)).ravel()
    except Exception:
        cx = cy = np.zeros(len(buildings))

    X = np.column_stack([
        pv, ec, ev,           # fitur utama
        pv*ec,                # PV × Economics
        pv*ev,                # PV × Environment
        ec*ev,                # Economics × Environment
        pv*ec*ev,             # tiga-arah
        np.mean([pv,ec,ev], axis=0),   # rata-rata
        np.min([pv,ec,ev], axis=0),    # minimum
        np.max([pv,ec,ev], axis=0),    # maksimum
        np.sqrt(pv**2+ec**2+ev**2),    # norma Euclidean
        cx, cy,               # spasial
    ])
    return X


def ml_ensemble(buildings: gpd.GeoDataFrame,
                 target_col: str = 'wlc_Baseline') -> tuple:
    """
    ML Ensemble: RandomForest + GBR mempelajari hubungan non-linear antara
    tiga indeks dan skor prioritas.

    Target: WLC Baseline (pseudo-label analitik)
    Justifikasi:
      (1) RF menangkap interaksi non-linear antar indeks yang tidak dapat
          dimodelkan oleh WLC linear (Chen & Breiman, 2001).
      (2) GBR mengoptimalkan sisa residual secara sekuensial → lebih akurat
          untuk pola spasial kontinyu (Friedman, 2001).
      (3) Ensemble RF+GBR mengurangi varians prediksi (model averaging).
    Validasi: K-Fold CV → R², MAE per model.
    """
    log.info('[ML] RandomForest + GBR ensemble scoring...')

    if target_col not in buildings.columns or len(buildings) < ML_CFG['min_samples']:
        log.warning(f'  ML: data tidak cukup atau target {target_col} tidak ada → skip')
        buildings['priority_ml'] = buildings.get(target_col,
                                                   buildings['solar_pv_index'])
        return buildings, {}

    X = _engineer_features(buildings)
    y = buildings[target_col].values
    kf = KFold(n_splits=ML_CFG['cv_k'], shuffle=True, random_state=ML_CFG['rf_seed'])

    # ── RandomForest ─────────────────────────────────────────────────────────
    rf = RandomForestRegressor(
        n_estimators   = ML_CFG['rf_n'],
        max_features   = 'sqrt',
        min_samples_leaf = 3,
        random_state   = ML_CFG['rf_seed'],
        n_jobs         = -1,
    )
    y_rf  = cross_val_predict(rf, X, y, cv=kf)
    r2_rf = float(r2_score(y, y_rf))
    mae_rf= float(mean_absolute_error(y, y_rf))
    rf.fit(X, y)
    log.info(f'  RandomForest {ML_CFG["cv_k"]}-Fold → R²={r2_rf:.4f} | MAE={mae_rf:.4f}')

    # ── GBR ──────────────────────────────────────────────────────────────────
    gbr = GradientBoostingRegressor(
        n_estimators   = ML_CFG['gbr_n'],
        learning_rate  = ML_CFG['gbr_lr'],
        max_depth      = ML_CFG['gbr_depth'],
        subsample      = 0.80,
        min_samples_leaf = 4,
        random_state   = ML_CFG['rf_seed'],
    )
    y_gbr  = cross_val_predict(gbr, X, y, cv=kf)
    r2_gbr = float(r2_score(y, y_gbr))
    mae_gbr= float(mean_absolute_error(y, y_gbr))
    gbr.fit(X, y)
    log.info(f'  GBR        {ML_CFG["cv_k"]}-Fold → R²={r2_gbr:.4f} | MAE={mae_gbr:.4f}')

    # ── XGBoost (opsional) ───────────────────────────────────────────────────
    xgb_metrics = {}
    if _opt['xgb']:
        xgb  = _opt['xgb'].XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8,
            random_state=ML_CFG['rf_seed'], verbosity=0
        )
        y_xgb  = cross_val_predict(xgb, X, y, cv=kf)
        r2_xgb = float(r2_score(y, y_xgb))
        mae_xgb= float(mean_absolute_error(y, y_xgb))
        xgb.fit(X, y)
        xgb_metrics = {'r2': r2_xgb, 'mae': mae_xgb}
        log.info(f'  XGBoost    {ML_CFG["cv_k"]}-Fold → R²={r2_xgb:.4f} | MAE={mae_xgb:.4f}')

    # ── Ensemble prediksi ─────────────────────────────────────────────────────
    pred_rf  = rf.predict(X)
    pred_gbr = gbr.predict(X)
    # Bobot ensemble proporsional R² (model lebih baik, bobot lebih besar)
    w_rf  = max(r2_rf,  0) + 1e-6
    w_gbr = max(r2_gbr, 0) + 1e-6
    pred  = (w_rf*pred_rf + w_gbr*pred_gbr) / (w_rf + w_gbr)
    buildings['priority_ml'] = MinMaxScaler().fit_transform(pred.reshape(-1,1)).ravel()

    # Feature importance
    feat_names = ['PV','Econ','Env',
                   'PV×Econ','PV×Env','Econ×Env','PV×Econ×Env',
                   'mean','min','max','norm','cx','cy']
    fi = dict(zip(feat_names[:X.shape[1]], rf.feature_importances_))
    top3 = sorted(fi, key=fi.get, reverse=True)[:3]
    log.info('  Feature importance (top 3): ' +
             ', '.join([f'{k}={fi[k]:.3f}' for k in top3]))

    metrics = {
        'rf':  {'r2': r2_rf,  'mae': mae_rf,  'n_estimators': ML_CFG['rf_n']},
        'gbr': {'r2': r2_gbr, 'mae': mae_gbr, 'n_estimators': ML_CFG['gbr_n']},
        'xgb': xgb_metrics,
        'feature_importance' : fi,
        'ensemble_weights'   : {'rf': w_rf/(w_rf+w_gbr), 'gbr': w_gbr/(w_rf+w_gbr)},
        'cv_k'               : ML_CFG['cv_k'],
        'n_samples'          : len(buildings),
        'justification': (
            'RF dipilih karena robust terhadap outlier dan menangkap interaksi '
            'non-linear. GBR melengkapi dengan optimasi gradien residual untuk '
            'pola spasial kontinu. Ensemble berbobot R² mengurangi varians prediksi.'
        ),
    }
    return buildings, metrics




def calculate_final_priority(buildings: gpd.GeoDataFrame,
                               ml_metrics: dict) -> gpd.GeoDataFrame:
    """
    Skor prioritas final = weighted ensemble:
      50% WLC Baseline  (transparan, dapat dipertanggungjawabkan secara teknis)
      20% AHP Balanced  (bobot expert-driven dari pairwise comparison)
      20% TOPSIS        (perspektif jarak ke solusi ideal)
      10% ML Ensemble   (koreksi non-linear, dinamis berdasarkan data)

    Rasional: WLC mendominasi untuk transparansi, dikoreksi oleh TOPSIS
    (perspektif geometris) dan ML (menangkap sinergi non-linear).
    """
    log.info('[FINAL] Menghitung skor prioritas final (ensemble 4 metode)...')

    # ── AHP score untuk skenario Balanced ────────────────────────────────────
    if 'ahp_weights' in buildings.columns:
        ahp_col = 'ahp_weights'
    else:
        # Hitung inline dari AHP Balanced
        ahp = AHPAnalyzer(['PV','Econ','Env'], AHP_MATRICES['AHP_Balanced'])
        r   = ahp.compute()
        w   = r['weights']
        buildings['score_ahp'] = (
            w['PV']   * buildings['solar_pv_index'] +
            w['Econ'] * buildings['econ_value_index'] +
            w['Env']  * buildings['env_deg_index']
        ).clip(0, 1)

    # ── Ensemble ─────────────────────────────────────────────────────────────
    has_ml = 'priority_ml' in buildings.columns and ml_metrics

    components = {
        'wlc_Baseline': 0.50,
        'score_ahp'   : 0.20,
        'score_topsis': 0.20,
        'priority_ml' : 0.10 if has_ml else 0.0,
    }
    if not has_ml:
        # Redistribusi bobot ML ke WLC
        components['wlc_Baseline'] += 0.10
    total_w = sum(components.values())

    score_f = np.zeros(len(buildings))
    for col, w in components.items():
        if col in buildings.columns and w > 0:
            score_f += (w / total_w) * buildings[col].values

    buildings['priority_score'] = MinMaxScaler().fit_transform(
        score_f.reshape(-1,1)
    ).ravel()

    log.info(f'  Priority Score final: '
             f'mean={buildings["priority_score"].mean():.4f} | '
             f'std={buildings["priority_score"].std():.4f} | '
             f'komponen={[k for k,v in components.items() if v>0]}')
    return buildings


def classify_and_rank(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Klasifikasi 3 kelas:
      Tinggi [0.67, 1.00] — prioritas TERTINGGI → instalasi segera
      Sedang [0.33, 0.67] — prioritas menengah → fase berikutnya
      Rendah [0.00, 0.33) — prioritas rendah → perlu kajian lebih lanjut
    """
    lo, hi = THR_LOW, THR_HIGH
    labels = ['Rendah', 'Sedang', 'Tinggi']
    buildings['priority_class'] = pd.cut(
        buildings['priority_score'],
        bins=[-np.inf, lo, hi, np.inf], labels=labels
    )
    buildings['priority_rank'] = (
        buildings['priority_score']
        .rank(ascending=False, method='min')
        .astype(int)
    )

    dist = buildings['priority_class'].value_counts()
    log.info(f'  Distribusi kelas: {dict(dist)}')
    return buildings




def monte_carlo_sensitivity(buildings: gpd.GeoDataFrame) -> tuple:
    """
    Monte Carlo dengan distribusi Dirichlet(1,1,1) untuk sampling bobot seragam
    di atas simplex 3D (seluruh kombinasi bobot w₁+w₂+w₃=1, wᵢ≥0).

    Untuk setiap simulasi:
      1. Sample (w₁, w₂, w₃) dari Dirichlet
      2. Hitung priority score = w·X
      3. Tentukan ranking setiap bangunan

    Output statistik:
      mc_mean_rank  — rata-rata ranking lintas semua simulasi
      mc_std_rank   — ketidakpastian ranking
      mc_prob_top10 — probabilitas masuk 10% teratas
      mc_stable     — True jika selalu di 20% teratas (robust)
    """
    log.info(f'[MONTE CARLO] {MC_N_SIM:,} simulasi Dirichlet...')
    rng   = np.random.RandomState(MC_SEED)
    X     = buildings[['solar_pv_index','econ_value_index',
                         'env_deg_index']].values.astype(float)
    n_alt = len(buildings)

    # Sample bobot sekaligus (vektorisasi penuh)
    W          = rng.dirichlet(MC_ALPHA, size=MC_N_SIM)   # (N_SIM, 3)
    scores_all = X @ W.T                                    # (n_alt, N_SIM)

    # Ranking: argsort descending → argsort → rank 1 = terbaik
    order_all  = np.argsort(-scores_all, axis=0)           # (n_alt, N_SIM)
    ranks_all  = np.empty_like(order_all)
    idx_rows   = np.arange(n_alt)
    for s in range(MC_N_SIM):
        ranks_all[order_all[:, s], s] = idx_rows + 1

    mean_rank   = ranks_all.mean(axis=1)
    std_rank    = ranks_all.std(axis=1)
    prob_top10  = (ranks_all <= max(1, n_alt * 0.10)).mean(axis=1)
    prob_top20  = (ranks_all <= max(1, n_alt * 0.20)).mean(axis=1)
    prob_top50  = (ranks_all <= max(1, n_alt * 0.50)).mean(axis=1)

    buildings['mc_mean_rank']  = mean_rank
    buildings['mc_std_rank']   = std_rank
    buildings['mc_prob_top10'] = prob_top10
    buildings['mc_prob_top20'] = prob_top20
    buildings['mc_stable']     = prob_top20 >= 0.70

    n_stable = int(buildings['mc_stable'].sum())
    log.info(f'  Bangunan STABIL (prob_top20 ≥ 70%): {n_stable}')
    log.info(f'  Mean rank uncertainty (σ): {std_rank.mean():.1f} posisi')

    mc_meta = {
        'n_sim'        : MC_N_SIM,
        'n_stable'     : n_stable,
        'mean_sigma'   : float(std_rank.mean()),
        'scores_all'   : scores_all,
        'ranks_all'    : ranks_all,
        'W_samples'    : W,
    }
    return buildings, mc_meta


def scenario_comparison(buildings: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Tabulasi perbandingan semua skenario (WLC + AHP):
      - Score mean/std per skenario
      - Spearman ρ vs Baseline
      - % bangunan yang tetap di top-N across skenario
    """
    log.info('[SENS] Perbandingan semua skenario...')
    baseline = buildings['wlc_Baseline'].values
    rows     = []

    all_wlc_cols = [c for c in buildings.columns if c.startswith('wlc_')]

    for col in all_wlc_cols:
        scen = col.replace('wlc_', '')
        vals = buildings[col].values
        rho, pval = spearmanr(baseline, vals)
        tau, _    = kendalltau(baseline, vals)

        top_n_base = set(np.argsort(baseline)[-TOP_N:])
        top_n_this = set(np.argsort(vals)[-TOP_N:])
        stable_pct = len(top_n_base & top_n_this) / TOP_N * 100

        w = WLC_SCENARIOS.get(scen, {})
        rows.append({
            'Skenario'       : scen,
            'Metode'         : 'WLC',
            'w_PV'           : round(w.get('PV',   0), 3),
            'w_Econ'         : round(w.get('Econ', 0), 3),
            'w_Env'          : round(w.get('Env',  0), 3),
            'mean_score'     : round(float(vals.mean()), 4),
            'std_score'      : round(float(vals.std()),  4),
            'Spearman_rho'   : round(float(rho),         4),
            'Kendall_tau'    : round(float(tau),         4),
            'p_value'        : round(float(pval),        6),
            f'stable_top{TOP_N}_pct': round(stable_pct, 1),
        })

    # Tambahkan TOPSIS dan ML
    for col, label in [('score_topsis','TOPSIS'), ('priority_ml','ML_Ensemble')]:
        if col in buildings.columns:
            vals = buildings[col].values
            rho, pval = spearmanr(baseline, vals)
            tau, _    = kendalltau(baseline, vals)
            top_n_this = set(np.argsort(vals)[-TOP_N:])
            stable_pct = len(top_n_base & top_n_this) / TOP_N * 100
            rows.append({
                'Skenario' : label, 'Metode': label,
                'w_PV': '-', 'w_Econ': '-', 'w_Env': '-',
                'mean_score': round(float(vals.mean()), 4),
                'std_score' : round(float(vals.std()),  4),
                'Spearman_rho': round(float(rho), 4),
                'Kendall_tau' : round(float(tau), 4),
                'p_value'     : round(float(pval), 6),
                f'stable_top{TOP_N}_pct': round(stable_pct, 1),
            })

    df = pd.DataFrame(rows)
    path = OUT + 'data/scenario_results.csv'
    df.to_csv(path, index=False)
    log.info(f'  {len(df)} skenario → {path}')
    return df


def quadrant_analysis(buildings: gpd.GeoDataFrame) -> dict:
    """
    Analisis kuadran 3 pasangan indeks:
      PV vs Econ   → kuadran A-B-C-D
      PV vs Env    → kuadran A-B-C-D
      Econ vs Env  → kuadran A-B-C-D

    Kuadran (menggunakan median sebagai batas):
      Q1 (Tinggi-Tinggi) → "Star"    — prioritas tinggi
      Q2 (Tinggi-Rendah) → "Leader"  — kuat di satu dimensi
      Q3 (Rendah-Tinggi) → "Follower"
      Q4 (Rendah-Rendah) → "Laggard" — prioritas rendah
    """
    log.info('[QUAD] Analisis kuadran...')
    pairs = [
        ('solar_pv_index', 'econ_value_index', 'PV', 'Econ'),
        ('solar_pv_index', 'env_deg_index',    'PV', 'Env'),
        ('econ_value_index','env_deg_index',   'Econ','Env'),
    ]
    results = {}
    for c1, c2, l1, l2 in pairs:
        m1 = buildings[c1].median()
        m2 = buildings[c2].median()
        q1 = ((buildings[c1] >= m1) & (buildings[c2] >= m2)).sum()
        q2 = ((buildings[c1] >= m1) & (buildings[c2] <  m2)).sum()
        q3 = ((buildings[c1] <  m1) & (buildings[c2] >= m2)).sum()
        q4 = ((buildings[c1] <  m1) & (buildings[c2] <  m2)).sum()
        results[f'{l1}_vs_{l2}'] = {
            'c1': c1, 'c2': c2, 'l1': l1, 'l2': l2,
            'm1': float(m1), 'm2': float(m2),
            'Q1_star': int(q1), 'Q2_leader_x': int(q2),
            'Q3_leader_y': int(q3), 'Q4_laggard': int(q4),
        }
        log.info(f'  {l1}×{l2}: Star={q1} Leader-{l1}={q2} '
                 f'Leader-{l2}={q3} Laggard={q4}')
    return results



def _add_map_elements(ax, bldg_wm: gpd.GeoDataFrame) -> None:
    """Tambah north arrow, scalebar, dan metadata ke axes peta."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    dx, dy  = x1-x0, y1-y0
    nx, ny  = x0+dx*0.96, y0+dy*0.93
    ax.annotate('', xy=(nx, ny), xytext=(nx, ny-dy*0.06),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.text(nx, ny+dy*0.01, 'N', ha='center', fontsize=12, fontweight='bold')
    sb_x = x0 + dx*0.05
    sb_y = y0 + dy*0.04
    ax.plot([sb_x, sb_x+100], [sb_y, sb_y], 'k-', lw=4)
    ax.text(sb_x+50, sb_y+dy*0.014, '100 m', ha='center', fontsize=8)


def _fmt(x, _=None):
    if abs(x) >= 1e6: return f'{x/1e6:.1f}M'
    if abs(x) >= 1e3: return f'{x/1e3:.1f}k'
    return f'{x:.0f}'


def plot_priority_map(buildings: gpd.GeoDataFrame) -> None:
    """
    Peta kartografi statis — choropleth Priority Score per bangunan.
    Elemen: colorbar, panah utara, scalebar, sumber data, basemap CartoDB.
    """
    log.info('[VIZ] Membuat peta prioritas statis...')
    bldg_wm = buildings.to_crs('EPSG:3857')
    fig, axes = plt.subplots(1, 2, figsize=(20, 9),
                               gridspec_kw={'width_ratios': [3, 1]})

    ax   = axes[0]
    ax_l = axes[1]   # panel legend + statistik

    # 1. Gambar bangunan tanpa garis tepi (zorder=2)
    bldg_wm.plot(column='priority_score', cmap=PRIORITY_CMAP,
                  edgecolor='none', legend=False,
                  ax=ax, vmin=0, vmax=1, zorder=2)

    # 2. Panggil peta dasar di layer paling bawah (zorder=1)
    if _opt['ctx']:
        try:
            _opt['ctx'].add_basemap(ax,
                source=_opt['ctx'].providers.CartoDB.Positron,
                zoom='auto', alpha=0.50, crs='EPSG:3857', zorder=1)
        except Exception: pass

    # 3. KUNCI KOORDINAT KAMERA KE AREA BANGUNAN
    xmin, ymin, xmax, ymax = bldg_wm.total_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Colorbar
    sm   = plt.cm.ScalarMappable(cmap=PRIORITY_CMAP, norm=mcolors.Normalize(0, 1))
    cbar = fig.colorbar(sm, ax=ax, fraction=0.022, pad=0.015, aspect=35)
    cbar.set_label('Priority Score', fontsize=11, fontweight='bold')
    cbar.set_ticks([0, THR_LOW, THR_HIGH, 1])
    cbar.set_ticklabels(['0.0\n(Rendah)', f'{THR_LOW}', f'{THR_HIGH}', '1.0\n(Tinggi)'])

    # Panah Utara & Skala
    try:
        _add_map_elements(ax, bldg_wm)
    except Exception as e:
        pass

    n_hi = int((buildings['priority_class']=='Tinggi').sum())
    ax.set_title(
        f'Peta Prioritas Solar PV — Per Bangunan\n'
        f'Task 4 | {len(buildings):,} bangunan | {n_hi} Prioritas Tinggi',
        fontsize=13, fontweight='bold', pad=12
    )
    ax.text(0.005, 0.005,
            f'Metode: WLC·50% + AHP·20% + TOPSIS·20% + ML·10%\n'
            f'Sumber: Task 1–3 + OSM + Satelit | Dibuat: {datetime.now():%Y-%m-%d}',
            transform=ax.transAxes, fontsize=7, color='#555', va='bottom')
    ax.set_axis_off()

    # ── Panel kanan: statistik & legend ──────────────────────────────────────
    ax_l.set_axis_off()
    y_pos = 0.98
    ax_l.text(0.05, y_pos, 'STATISTIK PRIORITAS', fontsize=11, fontweight='bold', transform=ax_l.transAxes, va='top')
    y_pos -= 0.07

    stats = [
        ('Total Bangunan', f'{len(buildings):,}'),
        ('Prioritas Tinggi', f'{n_hi:,} ({n_hi/len(buildings)*100:.1f}%)'),
        ('Prioritas Sedang', f'{int((buildings["priority_class"]=="Sedang").sum()):,}'),
        ('Prioritas Rendah', f'{int((buildings["priority_class"]=="Rendah").sum()):,}'),
        ('Score Mean', f'{buildings["priority_score"].mean():.4f}'),
        ('Score Std', f'{buildings["priority_score"].std():.4f}'),
    ]
    for label, val in stats:
        ax_l.text(0.05, y_pos, f'{label}:', fontsize=9, transform=ax_l.transAxes, va='top', color='#444')
        ax_l.text(0.65, y_pos, val, fontsize=9, fontweight='bold', transform=ax_l.transAxes, va='top')
        y_pos -= 0.055

    # Legend kelas
    y_pos -= 0.03
    ax_l.text(0.05, y_pos, 'KELAS PRIORITAS', fontsize=10, fontweight='bold', transform=ax_l.transAxes, va='top')
    y_pos -= 0.06
    for cls, clr in PRIORITY_COLORS.items():
        ax_l.add_patch(mpatches.FancyBboxPatch(
            (0.05, y_pos - 0.025), 0.15, 0.04,
            boxstyle='round,pad=0.01', color=clr, transform=ax_l.transAxes, zorder=5
        ))
        ax_l.text(0.25, y_pos - 0.008, cls, fontsize=9, transform=ax_l.transAxes, va='center')
        y_pos -= 0.06

    # Top-5 bangunan
    y_pos -= 0.02
    ax_l.text(0.05, y_pos, 'TOP 5 PRIORITAS', fontsize=10, fontweight='bold', transform=ax_l.transAxes, va='top')
    y_pos -= 0.06
    top5 = buildings.nsmallest(5, 'priority_rank')
    for _, row in top5.iterrows():
        ax_l.text(0.05, y_pos,
                  f"#{row['priority_rank']} Bldg-{int(row['bldg_id'])}: {row['priority_score']:.3f}",
                  fontsize=8, transform=ax_l.transAxes, va='top', color=PRIORITY_COLORS['Tinggi'])
        y_pos -= 0.05

    plt.tight_layout()
    path = OUT + 'maps/task4_priority_map.png'
    plt.savefig(path, dpi=300)
    plt.close()
    log.info(f'  → {path}')


def plot_interactive_map(buildings: gpd.GeoDataFrame) -> None:
    """
    Peta interaktif Folium multi-layer:
      Layer 1: Priority Score (choropleth utama)
      Layer 2: Solar PV Index
      Layer 3: Economic Value Index
      Layer 4: Environmental Degradation Index
      Layer 5: Triple Winners
    """
    if not _opt['folium']:
        log.warning('[VIZ] folium tidak terinstall → skip')
        return

    log.info('[VIZ] Membuat peta interaktif Folium multi-layer...')
    folium = _opt['folium']

    bldg_geo = buildings.to_crs(GEO_CRS).copy()
    bldg_geo['priority_class'] = bldg_geo['priority_class'].astype(str)
    centre   = [bldg_geo.geometry.centroid.y.mean(),
                 bldg_geo.geometry.centroid.x.mean()]

    m = folium.Map(location=centre, zoom_start=16, tiles='CartoDB positron',
                    prefer_canvas=True)

    def _p_color(v):
        if v >= THR_HIGH: return PRIORITY_COLORS['Tinggi']
        if v >= THR_LOW:  return PRIORITY_COLORS['Sedang']
        return PRIORITY_COLORS['Rendah']

    def _idx_color(v, cmap_name='Blues'):
        cmap = plt.get_cmap(cmap_name)
        rgba = cmap(v)
        return '#{:02x}{:02x}{:02x}'.format(
            int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255)
        )

    # Tooltip fields
    tt_fields  = ['bldg_id','priority_score','priority_class','priority_rank',
                   'triple_winner','solar_pv_index','econ_value_index',
                   'env_deg_index','mc_prob_top20']
    tt_fields  = [f for f in tt_fields if f in bldg_geo.columns]
    tt_aliases = {
        'bldg_id'        : 'ID Bangunan',
        'priority_score' : 'Priority Score',
        'priority_class' : 'Kelas Prioritas',
        'priority_rank'  : 'Ranking',
        'triple_winner'  : 'Triple Winner',
        'solar_pv_index' : 'PV Index (O1)',
        'econ_value_index':'Econ Index (O2)',
        'env_deg_index'  : 'Env Index (O3)',
        'mc_prob_top20'  : 'MC Prob Top-20%',
    }
    aliases = [tt_aliases.get(f, f) for f in tt_fields]

    geojson_data = bldg_geo[tt_fields + ['geometry']].to_json()

    # ── Layer 1: Priority Score ───────────────────────────────────────────────
    folium.GeoJson(
        geojson_data,
        name='🏆 Priority Score (Final)',
        style_function=lambda f: {
            'fillColor'   : _p_color(f['properties']['priority_score']),
            'color'       : '#222', 'weight': 0.4, 'fillOpacity': 0.78,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=tt_fields, aliases=aliases,
            localize=True, sticky=True,
            style='font-size:11px;background:rgba(255,255,255,0.93);border-radius:4px'
        ),
        highlight_function=lambda _: {'weight': 2.5, 'color': 'black'}
    ).add_to(m)

    # ── Layer 2–4: Individual Indices ─────────────────────────────────────────
    for idx_col, lbl, cmap_n in [
        ('solar_pv_index',   '☀ PV Index',   'YlOrRd'),
        ('econ_value_index', '💹 Econ Index', 'Blues'),
        ('env_deg_index',    '🌿 Env Index',  'Greens'),
    ]:
        if idx_col not in bldg_geo.columns: continue
        folium.GeoJson(
            bldg_geo[[idx_col,'geometry']].to_json(),
            name=lbl,
            show=False,
            style_function=lambda f, _c=idx_col, _cn=cmap_n: {
                'fillColor'  : _idx_color(f['properties'].get(_c, 0), _cn),
                'color'      : '#333', 'weight': 0.3, 'fillOpacity': 0.70,
            }
        ).add_to(m)

    # ── Layer 5: Triple Winners ───────────────────────────────────────────────
    tw_gdf = bldg_geo[bldg_geo['triple_winner']] if 'triple_winner' in bldg_geo.columns else pd.DataFrame()
    if len(tw_gdf):
        folium.GeoJson(
            tw_gdf[['geometry','bldg_id','priority_score']].to_json(),
            name='⭐ Triple Winners',
            style_function=lambda _: {
                'fillColor': 'gold', 'color': 'black',
                'weight': 1.5, 'fillOpacity': 0.90
            }
        ).add_to(m)

    # ── Top-N markers ─────────────────────────────────────────────────────────
    fg_top = folium.FeatureGroup(name=f'📍 Top-{TOP_N} Bangunan', show=True)
    topN   = bldg_geo.nsmallest(TOP_N, 'priority_rank')
    for _, row in topN.iterrows():
        c = row.geometry.centroid
        folium.Marker(
            location=[c.y, c.x],
            icon=folium.Icon(color='red', icon='star', prefix='fa'),
            tooltip=(f"Rank #{row['priority_rank']} | "
                     f"Score={row['priority_score']:.3f} | "
                     f"{'⭐' if row.get('triple_winner') else ''}")
        ).add_to(fg_top)
    fg_top.add_to(m)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_html = f"""
    <div style="position:fixed;bottom:25px;right:15px;z-index:9999;
                background:white;padding:14px 18px;border-radius:10px;
                border:1px solid #ccc;font-family:'Segoe UI',sans-serif;
                font-size:12px;box-shadow:2px 2px 8px rgba(0,0,0,0.2)">
      <b style="font-size:13px">🏆 Solar PV Priority</b><br><br>
      {''.join([
          f'<span style="display:inline-block;width:14px;height:14px;'
          f'background:{clr};margin-right:7px;border-radius:2px"></span>'
          f'{cls} ({"≥"+str(THR_HIGH) if cls=="Tinggi" else ("≥"+str(THR_LOW) if cls=="Sedang" else "<"+str(THR_LOW))})<br>'
          for cls, clr in PRIORITY_COLORS.items()
      ])}
      <hr style="margin:7px 0">
      <span style="font-size:10px;color:#666">
        ⭐ = Top-{TOP_N} | S=w₁·PV+w₂·Econ+w₃·Env<br>
        Sumber: Task 1–3 | {datetime.now():%Y-%m-%d}
      </span>
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(collapsed=False).add_to(m)

    path = OUT + 'maps/task4_priority_interactive.html'
    m.save(path)
    log.info(f'  → {path}')


def plot_analysis_charts(buildings: gpd.GeoDataFrame,
                          ahp_results: dict, ml_metrics: dict,
                          df_sens: pd.DataFrame) -> None:
    """
    Panel analisis 8-in-1:
      A. Distribusi Priority Score per kelas
      B. Scatter PV vs Econ (warna=priority)
      C. Scatter PV vs Env (warna=priority)
      D. Scatter Econ vs Env (warna=priority)
      E. AHP weight comparison antar skenario
      F. Scenario ranking comparison (heatmap mini)
      G. Top-20 bar chart
      H. Kontribusi tiap indeks ke total skor
    """
    log.info('[VIZ] Panel analisis 8-panel...')
    fig = plt.figure(figsize=(22, 18))
    gs  = GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.35)
    ax  = {
        'A': fig.add_subplot(gs[0, 0]),
        'B': fig.add_subplot(gs[0, 1]),
        'C': fig.add_subplot(gs[0, 2]),
        'D': fig.add_subplot(gs[1, 0]),
        'E': fig.add_subplot(gs[1, 1]),
        'F': fig.add_subplot(gs[1, 2]),
        'G': fig.add_subplot(gs[2, 0:2]),
        'H': fig.add_subplot(gs[2, 2]),
    }
    fig.suptitle('Task 4 — Analisis Multi-Kriteria Prioritas Solar PV',
                  fontsize=15, fontweight='bold', y=1.002)

    buildings = buildings.copy()
    buildings['priority_class'] = buildings['priority_class'].astype(str)
    c_pv = buildings['solar_pv_index'].values
    c_ec = buildings['econ_value_index'].values
    c_ev = buildings['env_deg_index'].values
    c_pr = buildings['priority_score'].values

    # ── A: Distribusi Priority Score ──────────────────────────────────────────
    for cls, clr in PRIORITY_COLORS.items():
        subset = buildings[buildings['priority_class']==cls]['priority_score']
        sns.kdeplot(subset, label=cls, color=clr, fill=True,
                     alpha=0.30, linewidth=1.8, ax=ax['A'])
    sns.histplot(buildings['priority_score'], bins=30, ax=ax['A'],
                  color='gray', alpha=0.25, edgecolor='white', lw=0.5)
    ax['A'].axvline(THR_LOW,  color='gray', ls='--', lw=1.5)
    ax['A'].axvline(THR_HIGH, color='gray', ls='--', lw=1.5)
    for i, (cls, clr) in enumerate(PRIORITY_COLORS.items()):
        n   = (buildings['priority_class']==cls).sum()
        pct = n/len(buildings)*100
        ax['A'].text(0.98, 0.96-i*0.10, f'{cls}: {n} ({pct:.1f}%)',
                     transform=ax['A'].transAxes, ha='right', va='top',
                     fontsize=8, color=clr, fontweight='bold')
    ax['A'].set_xlabel('Priority Score', fontsize=10)
    ax['A'].set_ylabel('Densitas / Jumlah', fontsize=10)
    ax['A'].set_title('A. Distribusi Priority Score', fontweight='bold')
    ax['A'].legend(fontsize=8, title='Kelas', title_fontsize=8)

    # ── B–D: Scatter tiga pasangan ────────────────────────────────────────────
    scatter_pairs = [
        ('B', c_pv, c_ec, 'PV Index (O1)', 'Econ Index (O2)', 'B. PV vs Ekonomi'),
        ('C', c_pv, c_ev, 'PV Index (O1)', 'Env Index (O3)',  'C. PV vs Lingkungan'),
        ('D', c_ec, c_ev, 'Econ Index (O2)','Env Index (O3)', 'D. Ekonomi vs Lingkungan'),
    ]
    for key, xs, ys, xl, yl, title in scatter_pairs:
        sc = ax[key].scatter(xs, ys, c=c_pr, cmap=PRIORITY_CMAP,
                              s=18, alpha=0.65, edgecolors='none', vmin=0, vmax=1)
        # Medians
        ax[key].axvline(np.median(xs), color='gray', ls=':', lw=1)
        ax[key].axhline(np.median(ys), color='gray', ls=':', lw=1)
        # Triple Winners
        tw_mask = buildings['triple_winner'].values if 'triple_winner' in buildings.columns \
                  else np.zeros(len(buildings), dtype=bool)
        if tw_mask.sum():
            ax[key].scatter(xs[tw_mask], ys[tw_mask], c='gold', s=60,
                             edgecolors='black', lw=0.8, zorder=5, marker='*')
        r, _ = pearsonr(xs, ys)
        ax[key].text(0.05, 0.96, f'r={r:.3f}', transform=ax[key].transAxes,
                     fontsize=8, va='top',
                     bbox=dict(boxstyle='round', fc='white', alpha=0.7))
        ax[key].set_xlabel(xl, fontsize=9)
        ax[key].set_ylabel(yl, fontsize=9)
        ax[key].set_title(title, fontweight='bold')
        cb = fig.colorbar(sc, ax=ax[key], fraction=0.04, pad=0.02)
        cb.set_label('Priority', fontsize=7)

    # ── E: AHP weight comparison ──────────────────────────────────────────────
    ahp_names  = list(ahp_results.keys())
    ahp_pv     = [ahp_results[n]['weights'].get('PV',   0) for n in ahp_names]
    ahp_ec     = [ahp_results[n]['weights'].get('Econ', 0) for n in ahp_names]
    ahp_ev     = [ahp_results[n]['weights'].get('Env',  0) for n in ahp_names]
    x_pos      = np.arange(len(ahp_names))
    width      = 0.25
    ax['E'].bar(x_pos-width, ahp_pv, width, label='PV',   color=CRITERIA_COLORS['PV'],   alpha=0.80)
    ax['E'].bar(x_pos,       ahp_ec, width, label='Econ', color=CRITERIA_COLORS['Econ'], alpha=0.80)
    ax['E'].bar(x_pos+width, ahp_ev, width, label='Env',  color=CRITERIA_COLORS['Env'],  alpha=0.80)
    ax['E'].set_xticks(x_pos)
    ax['E'].set_xticklabels([n.replace('AHP_','') for n in ahp_names], rotation=25, fontsize=8)
    ax['E'].set_ylabel('Bobot AHP', fontsize=9)
    ax['E'].set_ylim(0, 0.75)
    ax['E'].set_title('E. Perbandingan Bobot AHP per Skenario', fontweight='bold')
    ax['E'].legend(fontsize=8)
    for n, name in enumerate(ahp_names):
        cr   = ahp_results[name]['CR']
        clr  = 'green' if ahp_results[name]['consistent'] else 'red'
        ax['E'].text(n, 0.70, f'CR={cr:.3f}', ha='center', fontsize=7,
                     color=clr, fontweight='bold')

    # ── F: Scenario ranking heatmap mini ──────────────────────────────────────
    if not df_sens.empty:
        scen_cols = [c for c in buildings.columns if c.startswith('wlc_')][:5]
        if scen_cols:
            top10_idx = buildings.nsmallest(10, 'priority_rank').index
            heat_data  = buildings.loc[top10_idx, scen_cols]
            heat_data.index = [f"B{int(buildings.loc[i,'bldg_id'])}"
                                for i in top10_idx]
            heat_data.columns = [c.replace('wlc_','') for c in heat_data.columns]
            sns.heatmap(heat_data, cmap='RdYlGn', annot=True, fmt='.2f',
                        linewidths=0.4, ax=ax['F'], cbar=True,
                        vmin=0, vmax=1,
                        annot_kws={'size': 7})
            ax['F'].set_title('F. Skor Top-10 per Skenario WLC', fontweight='bold')
            ax['F'].tick_params(axis='x', rotation=30, labelsize=7)
            ax['F'].tick_params(axis='y', rotation=0, labelsize=7)

    # ── G: Top-20 horizontal bar ──────────────────────────────────────────────
    top20 = buildings.nsmallest(TOP_N, 'priority_rank').copy()
    top20['label'] = [f"#{int(r['priority_rank'])} B{int(r['bldg_id'])}"
                      for _, r in top20.iterrows()]
    bar_data = top20[['label','solar_pv_index','econ_value_index',
                        'env_deg_index','priority_score']].set_index('label')
    x_g  = np.arange(len(top20))
    w_g  = 0.20
    ax['G'].barh(x_g + w_g*1.5, top20['priority_score'], w_g*1.2,
                  color=PRIORITY_COLORS['Tinggi'], alpha=0.90, label='Priority Score')
    ax['G'].barh(x_g + w_g*0.5, top20['solar_pv_index'],    w_g,
                  color=CRITERIA_COLORS['PV'],   alpha=0.70, label='PV Index')
    ax['G'].barh(x_g - w_g*0.5, top20['econ_value_index'],  w_g,
                  color=CRITERIA_COLORS['Econ'], alpha=0.70, label='Econ Index')
    ax['G'].barh(x_g - w_g*1.5, top20['env_deg_index'],     w_g,
                  color=CRITERIA_COLORS['Env'],  alpha=0.70, label='Env Index')
    ax['G'].set_yticks(x_g)
    ax['G'].set_yticklabels(top20['label'].values, fontsize=7)
    ax['G'].invert_yaxis()
    ax['G'].set_xlabel('Nilai Indeks / Skor', fontsize=9)
    ax['G'].set_title(f'G. Top-{TOP_N} Bangunan Prioritas: Profil Indeks', fontweight='bold')
    ax['G'].legend(fontsize=8, loc='lower right')
    ax['G'].set_xlim(0, 1.1)

    # ── H: Kontribusi indikator ───────────────────────────────────────────────
    w_base  = WLC_SCENARIOS['Baseline']
    contrib = {
        'PV':   float(w_base['PV']   * buildings['solar_pv_index'].mean()),
        'Econ': float(w_base['Econ'] * buildings['econ_value_index'].mean()),
        'Env':  float(w_base['Env']  * buildings['env_deg_index'].mean()),
    }
    colors_h = [CRITERIA_COLORS[k] for k in contrib]
    wedges, texts, autotexts = ax['H'].pie(
        contrib.values(), labels=contrib.keys(),
        autopct='%1.1f%%', colors=colors_h,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2},
        startangle=140
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight('bold')
    ax['H'].set_title('H. Kontribusi Rata-rata\n(Baseline Weights)', fontweight='bold')

    # Anotasi ML metrics
    # Anotasi ML metrics
    if ml_metrics:
        rf_r2  = ml_metrics.get('rf',  {}).get('r2',  'N/A')
        gbr_r2 = ml_metrics.get('gbr', {}).get('r2',  'N/A')
        
        # PERBAIKAN: Format string diatur di luar f-string utama
        rf_str  = f"{rf_r2:.4f}" if isinstance(rf_r2, float) else str(rf_r2)
        gbr_str = f"{gbr_r2:.4f}" if isinstance(gbr_r2, float) else str(gbr_r2)
        
        note   = (f'[ML {ML_CFG["cv_k"]}-Fold CV] '
                  f'RF R²={rf_str} | GBR R²={gbr_str}')
        fig.text(0.01, -0.005, note, fontsize=8, color='#666')

    plt.tight_layout(rect=[0, 0.01, 1, 1])
    path = OUT + 'figures/task4_analysis_charts.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    log.info(f'  → {path}')


def plot_ahp_visualization(ahp_results: dict) -> None:
    """Diagram bobot AHP + matriks CR untuk semua skenario."""
    log.info('[VIZ] Diagram AHP...')
    n_scen = len(ahp_results)
    fig, axes = plt.subplots(2, n_scen, figsize=(4*n_scen, 8))
    fig.suptitle('Analisis AHP — Bobot dan Consistency Ratio',
                  fontsize=13, fontweight='bold')

    for j, (name, res) in enumerate(ahp_results.items()):
        # ── Baris atas: pie chart bobot ────────────────────────────────────
        ax_top = axes[0, j] if n_scen > 1 else axes[0]
        w = res['weights']
        vals   = [w.get('PV',0), w.get('Econ',0), w.get('Env',0)]
        lbls   = ['PV', 'Econ', 'Env']
        clrs   = [CRITERIA_COLORS[k] for k in lbls]
        ax_top.pie(vals, labels=lbls, autopct='%1.1f%%', colors=clrs,
                    wedgeprops={'edgecolor':'white','linewidth':2}, startangle=90)
        cr_lbl = f'CR={res["CR"]:.4f} {"✓" if res["consistent"] else "✗"}'
        cr_clr = 'green' if res['consistent'] else 'red'
        ax_top.set_title(f'{name.replace("AHP_","")}\n{cr_lbl}',
                          fontsize=9, fontweight='bold', color=cr_clr)

        # ── Baris bawah: heatmap matriks AHP ──────────────────────────────
        ax_bot = axes[1, j] if n_scen > 1 else axes[1]
        mat    = np.array(res['matrix'])
        sns.heatmap(np.log(mat), annot=mat, fmt='.2g', cmap='RdBu_r',
                    center=0, ax=ax_bot, linewidths=0.5,
                    xticklabels=['PV','Econ','Env'],
                    yticklabels=['PV','Econ','Env'],
                    cbar=False, annot_kws={'size':8})
        ax_bot.set_title('Matriks (log scale)', fontsize=8)
        ax_bot.tick_params(labelsize=8)

    plt.tight_layout()
    path = OUT + 'figures/task4_ahp_analysis.png'
    plt.savefig(path, dpi=300)
    plt.close()
    log.info(f'  → {path}')


def plot_monte_carlo(buildings: gpd.GeoDataFrame, mc_meta: dict) -> None:
    """
    Visualisasi Monte Carlo:
      Panel kiri  — Distribusi MC mean rank (histogram + KDE)
      Panel tengah— Scatter: mean_rank vs std_rank (warna=prob_top20)
      Panel kanan — Top-20 prob_top20 bar chart dengan uncertainty bar
    """
    log.info('[VIZ] Plot Monte Carlo...')
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Analisis Sensitivitas Monte Carlo ({mc_meta["n_sim"]:,} Simulasi Dirichlet)',
                  fontsize=13, fontweight='bold')

    # ── Kiri: Distribusi MC Mean Rank ─────────────────────────────────────────
    ax = axes[0]
    sns.histplot(buildings['mc_mean_rank'], bins=30, kde=True,
                  color='#4575b4', ax=ax, edgecolor='white', alpha=0.75)
    ax.set_xlabel('Rata-rata Ranking (MC)', fontsize=10)
    ax.set_ylabel('Jumlah Bangunan', fontsize=10)
    ax.set_title('Distribusi MC Mean Rank', fontweight='bold')
    ax.text(0.98, 0.96,
            f'σ_mean = {buildings["mc_std_rank"].mean():.1f} posisi\n'
            f'Stable ≥ 70%: {buildings["mc_stable"].sum()} bdg',
            transform=ax.transAxes, ha='right', va='top', fontsize=8,
            bbox=dict(boxstyle='round', fc='white', alpha=0.8))

    # ── Tengah: Mean Rank vs Std Rank ─────────────────────────────────────────
    ax = axes[1]
    sc = ax.scatter(buildings['mc_mean_rank'], buildings['mc_std_rank'],
                     c=buildings['mc_prob_top20'], cmap='RdYlGn',
                     s=20, alpha=0.65, edgecolors='none', vmin=0, vmax=1)
    # Stable buildings
    stab = buildings[buildings['mc_stable']]
    if len(stab):
        ax.scatter(stab['mc_mean_rank'], stab['mc_std_rank'],
                    c='gold', s=60, edgecolors='black', lw=0.8,
                    marker='*', zorder=5, label=f'Stable ({len(stab)})')
    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label('Prob. Top-20%', fontsize=8)
    ax.set_xlabel('MC Mean Rank', fontsize=10)
    ax.set_ylabel('MC Std Rank (ketidakpastian)', fontsize=10)
    ax.set_title('Mean Rank vs Ketidakpastian Ranking', fontweight='bold')
    if len(stab): ax.legend(fontsize=8)

    # ── Kanan: Top-20 MC probability ─────────────────────────────────────────
    ax   = axes[2]
    top20 = buildings.nsmallest(TOP_N, 'priority_rank').copy()
    top20['label'] = [f"#{int(r['priority_rank'])} B{int(r['bldg_id'])}"
                      for _, r in top20.iterrows()]
    x_pos = np.arange(len(top20))
    bars  = ax.barh(x_pos, top20['mc_prob_top20']*100,
                     color=['gold' if r['mc_stable'] else '#4575b4'
                            for _, r in top20.iterrows()],
                     edgecolor='white', lw=0.5, alpha=0.85)
    ax.errorbar(top20['mc_prob_top20']*100,
                x_pos,
                xerr=top20['mc_std_rank']/len(buildings)*100,
                fmt='none', color='black', capsize=3, elinewidth=0.8)
    ax.set_yticks(x_pos)
    ax.set_yticklabels(top20['label'].values, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel('Probabilitas Top-20% (%)', fontsize=9)
    ax.set_title(f'MC Stability Top-{TOP_N} Bangunan', fontweight='bold')
    ax.axvline(70, color='red', ls='--', lw=1.5, label='70% threshold')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 110)
    for bar in bars:
        w_val = bar.get_width()
        ax.text(w_val+1, bar.get_y()+bar.get_height()/2,
                f'{w_val:.1f}%', va='center', fontsize=7)

    plt.tight_layout()
    path = OUT + 'figures/task4_monte_carlo.png'
    plt.savefig(path, dpi=300)
    plt.close()
    log.info(f'  → {path}')


def plot_tornado_chart(buildings: gpd.GeoDataFrame) -> None:
    """
    Tornado chart: dampak perubahan bobot tiap indeks terhadap rata-rata skor
    dan persentase bangunan Prioritas Tinggi.
    Setiap indeks divariasikan ±0.20 dari baseline, dua lainnya disesuaikan.
    """
    log.info('[VIZ] Tornado chart sensitivitas...')
    base_w  = WLC_SCENARIOS['Baseline']
    delta   = 0.15
    indices = ['PV', 'Econ', 'Env']
    X       = buildings[['solar_pv_index','econ_value_index',
                           'env_deg_index']].values

    rows = []
    for ind in indices:
        for sign in [-1, +1]:
            w_test = deepcopy(base_w)
            w_test[ind] = np.clip(base_w[ind] + sign*delta, 0.05, 0.90)
            # Normalisasi
            total = sum(w_test.values())
            w_arr = np.array([w_test['PV']/total, w_test['Econ']/total,
                               w_test['Env']/total])
            scores = X @ w_arr
            pct_hi = (scores >= THR_HIGH).mean() * 100
            rows.append({
                'Indeks': ind, 'delta': sign*delta,
                'mean_score': float(scores.mean()),
                'pct_high': float(pct_hi),
                'w_PV': w_arr[0], 'w_Econ': w_arr[1], 'w_Env': w_arr[2],
            })

    df_t = pd.DataFrame(rows)
    base_mean = float(
        (X @ np.array([base_w['PV'], base_w['Econ'], base_w['Env']])).mean()
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Tornado Chart — Sensitivitas Bobot Terhadap Priority Score',
                  fontsize=12, fontweight='bold')

    for ax_i, metric, title in [
        (0, 'mean_score', 'Dampak terhadap Mean Priority Score'),
        (1, 'pct_high',   'Dampak terhadap % Bangunan Prioritas Tinggi'),
    ]:
        ax = axes[ax_i]
        y_pos = np.arange(len(indices))
        for j, ind in enumerate(indices):
            lo = df_t[(df_t['Indeks']==ind) & (df_t['delta']<0)][metric].values[0]
            hi = df_t[(df_t['Indeks']==ind) & (df_t['delta']>0)][metric].values[0]
            ref = base_mean if metric == 'mean_score' \
                  else float((X @ np.array([base_w['PV'], base_w['Econ'],
                                             base_w['Env']])).mean() >= THR_HIGH) * 100
            width_lo = abs(lo - ref)
            width_hi = abs(hi - ref)
            ax.barh(j, -width_lo, left=ref, color=CRITERIA_COLORS[ind],
                     alpha=0.60, edgecolor='white', height=0.5)
            ax.barh(j, +width_hi, left=ref, color=CRITERIA_COLORS[ind],
                     alpha=0.90, edgecolor='white', height=0.5,
                     label=ind if ax_i == 0 else '')
            ax.text(ref-width_lo-0.003, j, f'{lo:.3f}', va='center',
                     ha='right', fontsize=8)
            ax.text(ref+width_hi+0.003, j, f'{hi:.3f}', va='center',
                     ha='left', fontsize=8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(indices, fontsize=10, fontweight='bold')
        ax.axvline(ref, color='black', lw=1.5, ls='-')
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.set_xlabel(metric.replace('_',' ').title(), fontsize=9)
        if ax_i == 0:
            ax.legend(fontsize=8, title='Indeks', title_fontsize=8)

    plt.tight_layout()
    path = OUT + 'figures/task4_sensitivity_tornado.png'
    plt.savefig(path, dpi=300)
    plt.close()
    log.info(f'  → {path}')


def plot_quadrant_analysis(buildings: gpd.GeoDataFrame,
                             quad_res: dict) -> None:
    """Panel scatter 3 kuadran dengan anotasi jumlah bangunan per kuadran."""
    log.info('[VIZ] Plot quadrant analysis...')
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Analisis Kuadran Antar Indeks', fontsize=13, fontweight='bold')

    pairs_plot = [
        ('solar_pv_index',   'econ_value_index', 'PV', 'Econ'),
        ('solar_pv_index',   'env_deg_index',    'PV', 'Env'),
        ('econ_value_index', 'env_deg_index',    'Econ','Env'),
    ]
    for i, (c1, c2, l1, l2) in enumerate(pairs_plot):
        ax  = axes[i]
        key = f'{l1}_vs_{l2}'
        m1  = quad_res[key]['m1']
        m2  = quad_res[key]['m2']

        sc = ax.scatter(buildings[c1], buildings[c2],
                         c=buildings['priority_score'], cmap=PRIORITY_CMAP,
                         s=20, alpha=0.70, edgecolors='none', vmin=0, vmax=1)

        # Triple Winners
        tw = buildings[buildings['triple_winner']] if 'triple_winner' in buildings.columns else pd.DataFrame()
        if len(tw):
            ax.scatter(tw[c1], tw[c2], c='gold', s=80, edgecolors='black',
                        lw=0.8, zorder=5, marker='*', label='Triple Winner')

        # Garis median
        ax.axvline(m1, color='gray', ls='--', lw=1.2, alpha=0.7)
        ax.axhline(m2, color='gray', ls='--', lw=1.2, alpha=0.7)

        # Anotasi jumlah per kuadran
        q_labels = {
            'Q1 Star'          : (m1*1.25, m2*1.25, quad_res[key]['Q1_star']),
            f'Q2 {l1}-Leader'  : (m1*1.25, m2*0.5,  quad_res[key]['Q2_leader_x']),
            f'Q3 {l2}-Leader'  : (m1*0.4,  m2*1.25, quad_res[key]['Q3_leader_y']),
            'Q4 Laggard'       : (m1*0.4,  m2*0.5,  quad_res[key]['Q4_laggard']),
        }
        for lbl, (xq, yq, n) in q_labels.items():
            xq = min(xq, 0.95)
            yq = min(yq, 0.95)
            ax.text(xq, yq, f'{lbl}\nn={n}', fontsize=7.5,
                     ha='center', va='center',
                     bbox=dict(boxstyle='round', fc='white', alpha=0.75, lw=0.5))

        cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label('Priority Score', fontsize=7)
        ax.set_xlabel(f'{l1} Index', fontsize=10)
        ax.set_ylabel(f'{l2} Index', fontsize=10)
        ax.set_title(f'{l1} × {l2}', fontweight='bold')
        if len(tw): ax.legend(fontsize=7, loc='lower right')

    plt.tight_layout()
    path = OUT + 'maps/task4_quadrant_maps.png'
    plt.savefig(path, dpi=300)
    plt.close()
    log.info(f'  → {path}')


def export_all(buildings: gpd.GeoDataFrame,
                ahp_results: dict,
                ml_metrics: dict,
                df_sens: pd.DataFrame,
                mc_meta: dict) -> None:
    """Ekspor semua hasil ke GeoJSON, CSV, dan JSON summary."""
    log.info('[EXPORT] Mengekspor hasil Task 4...')

    # ── Kolom untuk ekspor ────────────────────────────────────────────────────
    geo_cols  = ['bldg_id','geometry',
                  'solar_pv_index','econ_value_index','env_deg_index',
                  'wlc_Baseline','score_topsis','score_ahp','priority_ml',
                  'priority_score','priority_class','priority_rank',
                  'mc_mean_rank','mc_std_rank',
                  'mc_prob_top10','mc_prob_top20','mc_stable']
    geo_cols  = [c for c in geo_cols if c in buildings.columns]
    out_gdf   = buildings[geo_cols].copy()
    out_gdf['priority_class'] = out_gdf['priority_class'].astype(str)

    # GeoJSON (WGS84)
    geo_out = out_gdf.to_crs(GEO_CRS)
    gj_path = OUT + 'data/priority_score_final.geojson'
    geo_out.to_file(gj_path, driver='GeoJSON')
    log.info(f'  → {gj_path}')

    # CSV Rankings
    csv_path = OUT + 'data/building_rankings.csv'
    geo_out.drop(columns='geometry').sort_values('priority_rank').to_csv(csv_path, index=False)
    log.info(f'  → {csv_path}')

    # Top-N
    top_path = OUT + f'data/top{TOP_N}_priority.csv'
    (geo_out.drop(columns='geometry')
     .nsmallest(TOP_N, 'priority_rank')
     .to_csv(top_path, index=False))
    log.info(f'  → {top_path}')

    # Monte Carlo
    mc_path = OUT + 'data/monte_carlo_results.csv'
    mc_cols = ['bldg_id','priority_rank','mc_mean_rank','mc_std_rank',
                'mc_prob_top10','mc_prob_top20','mc_stable']
    mc_cols = [c for c in mc_cols if c in buildings.columns]
    buildings[mc_cols].to_csv(mc_path, index=False)
    log.info(f'  → {mc_path}')

    # ── JSON Ringkasan ─────────────────────────────────────────────────────────
    n_hi = int((buildings['priority_class']=='Tinggi').sum())
    n_md = int((buildings['priority_class']=='Sedang').sum())
    n_lo = int((buildings['priority_class']=='Rendah').sum())
    n_st = int(buildings['mc_stable'].sum()) if 'mc_stable' in buildings.columns else 0

    top_n_rows = []
    for _, r in buildings.nsmallest(TOP_N, 'priority_rank').iterrows():
        top_n_rows.append({
            'rank'          : int(r['priority_rank']),
            'bldg_id'       : int(r['bldg_id']),
            'priority_score': round(float(r['priority_score']), 4),
            'pv_index'      : round(float(r['solar_pv_index']), 4),
            'econ_index'    : round(float(r['econ_value_index']), 4),
            'env_index'     : round(float(r['env_deg_index']), 4),
            'mc_stable'     : bool(r.get('mc_stable', False)),
        })

    summary = {
        'assessment'           : 'Geospatial Data Engineer — Task 4 Multi-Criteria Integration',
        'n_buildings'          : int(len(buildings)),
        'priority_distribution': {
            'Tinggi': n_hi, 'Sedang': n_md, 'Rendah': n_lo,
            'pct_Tinggi': round(n_hi/len(buildings)*100, 1),
        },
        'n_stable_buildings'   : n_st,
        'mean_priority_score'  : round(float(buildings['priority_score'].mean()), 4),
        'std_priority_score'   : round(float(buildings['priority_score'].std()),  4),
        'top_n_priority'       : top_n_rows,
        'mcda_methods'         : {
            'WLC'   : {'n_scenarios': len(WLC_SCENARIOS), 'scenarios': list(WLC_SCENARIOS)},
            'AHP'   : {
                'n_matrices': len(ahp_results),
                'results': {
                    k: {'weights': v['weights'], 'CR': round(v['CR'],4),
                         'consistent': v['consistent']}
                    for k, v in ahp_results.items()
                }
            },
            'TOPSIS': {'description': 'Hwang & Yoon (1981), Baseline weights'},
            'ML'    : ml_metrics,
        },
        'final_ensemble'       : {
            'WLC_Baseline': 0.50,
            'AHP_Balanced': 0.20,
            'TOPSIS'      : 0.20,
            'ML_Ensemble' : 0.10 if ml_metrics else 0.0,
        },
        'monte_carlo'          : {
            'n_simulations' : mc_meta.get('n_sim', MC_N_SIM),
            'distribution'  : 'Dirichlet(1,1,1)',
            'n_stable'      : n_st,
            'mean_rank_uncertainty': round(float(mc_meta.get('mean_sigma', 0)), 2),
        },
        'thresholds'           : {'low': THR_LOW, 'high': THR_HIGH},
        'recommendations'      : _generate_recommendations(buildings),
        'generated_at'         : datetime.now().isoformat(),
    }

    json_path = OUT + 'data/task4_summary.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ── Log ringkasan ──────────────────────────────────────────────────────────
    log.info('\n' + '═'*65)
    log.info('  RINGKASAN FINAL TASK 4 — PRIORITY SCORE')
    log.info('═'*65)
    log.info(f'  Total bangunan    : {len(buildings):,}')
    log.info(f'  Prioritas Tinggi  : {n_hi:,} ({summary["priority_distribution"]["pct_Tinggi"]:.1f}%)')
    log.info(f'  Prioritas Sedang  : {n_md:,}')
    log.info(f'  Prioritas Rendah  : {n_lo:,}')
    log.info(f'  Bangunan Stabil   : {n_st:,} (prob top-20% ≥ 70% in MC)')
    log.info(f'  Score mean ± std  : {summary["mean_priority_score"]:.4f} ± {summary["std_priority_score"]:.4f}')
    log.info('─' * 65)
    log.info(f'  TOP 5 BANGUNAN PRIORITAS:')
    for row in top_n_rows[:5]:
        flag = '🔒' if row['mc_stable'] else ''
        log.info(f'    #{row["rank"]:>3} | Bldg-{row["bldg_id"]:<6} | '
                 f'Score={row["priority_score"]:.4f} | '
                 f'PV={row["pv_index"]:.3f} Econ={row["econ_index"]:.3f} '
                 f'Env={row["env_index"]:.3f} {flag}')
    log.info('═'*65)
    log.info(f'  JSON → {json_path}')

def _generate_recommendations(buildings: gpd.GeoDataFrame) -> list:
    """
    Rekomendasi awal pengembangan Solar PV berdasarkan hasil prioritas.
    """
    recs = []
    n    = len(buildings)

    # Rekomendasi 1: Top prioritas stabil (MC)
    if 'mc_stable' in buildings.columns:
        stable = buildings[buildings['mc_stable']]
        if len(stable):
            recs.append({
                'fase'     : 'Fase 1 — Implementasi Segera',
                'kategori' : 'Stable High Priority (MC)',
                'n_bangunan': int(len(stable)),
                'deskripsi': (
                    f'{len(stable)} bangunan yang secara konsisten berada di '
                    f'top-20% pada >70% dari {MC_N_SIM:,} simulasi Monte Carlo. '
                    f'Peringkat robust terhadap ketidakpastian pembobotan.'
                ),
                'bldg_ids' : [int(i) for i in stable.nsmallest(5,'mc_mean_rank')['bldg_id'].tolist()],
            })

    # Rekomendasi 2: High PV, Moderate Economics
    pv_hi = buildings[buildings['solar_pv_index'] >= 0.70]
    recs.append({
        'fase'     : 'Fase 2 — Prioritas Menengah',
        'kategori' : 'PV Champion (PV tinggi)',
        'n_bangunan': int(len(pv_hi)),
        'deskripsi': (
            f'{len(pv_hi)} bangunan dengan PV Index ≥ 0.70. '
            f'Perlu kajian ekonomi lebih lanjut untuk memastikan '
            f'keberlanjutan finansial. Kandidat untuk program subsidi.'
        ),
        'bldg_ids' : [int(i) for i in pv_hi.nsmallest(5,'priority_rank')['bldg_id'].tolist()],
    })

    # Rekomendasi 3: High Env degradation
    env_hi = buildings[buildings['env_deg_index'] >= 0.70]
    recs.append({
        'fase'     : 'Fase 2 — Prioritas Menengah',
        'kategori' : 'Environmental Hero (Lingkungan terdegradasi)',
        'n_bangunan': int(len(env_hi)),
        'deskripsi': (
            f'{len(env_hi)} bangunan di area dengan degradasi lingkungan tinggi '
            f'(AOD↑, LST↑, NDVI↓). Transisi ke energi surya di sini memberikan '
            f'manfaat kesehatan dan iklim terbesar.'
        ),
        'bldg_ids' : [int(i) for i in env_hi.nsmallest(5,'priority_rank')['bldg_id'].tolist()],
    })

    recs.append({
        'fase'     : 'Fase 3 — Jangka Panjang',
        'kategori' : 'Semua bangunan Prioritas Rendah',
        'n_bangunan': int((buildings['priority_class']=='Rendah').sum()),
        'deskripsi': (
            'Bangunan prioritas rendah dapat dijadwalkan pada fase pengembangan '
            'jangka panjang setelah infrastruktur dan regulasi matur. '
            'Pertimbangkan skema insentif untuk meningkatkan adopsi.'
        ),
        'bldg_ids' : [],
    })

    return recs


def main(demo: bool = False) -> gpd.GeoDataFrame:
    t_start = datetime.now()
    log.info('=' * 70)
    log.info('  TASK 4 — INTEGRASI MULTI-KRITERIA: PRIORITAS SOLAR PV')
    log.info('  Technical Assessment · Geospatial Data Engineer')
    log.info(f'  Dimulai: {t_start:%Y-%m-%d %H:%M:%S}')
    log.info(f'  Mode: {"DEMO (data sintetis)" if demo else "PRODUCTION (output Task 1–3)"}')
    log.info('=' * 70)

    # ── [1] Load & Merge ──────────────────────────────────────────────────────
    buildings = load_and_merge(demo=demo)

    # ── [2] AHP ───────────────────────────────────────────────────────────────
    ahp_results = run_all_ahp(buildings)

    # ── [3] WLC + TOPSIS semua skenario ──────────────────────────────────────
    buildings = run_all_wlc(buildings)

    # ── [4] AHP score (skenario Balanced) ─────────────────────────────────────
    ahp_bln   = ahp_results.get('AHP_Balanced', {})
    if ahp_bln:
        w = ahp_bln.get('weights', {'PV':0.54, 'Econ':0.30, 'Env':0.16})
        buildings = wlc_score(buildings, w, 'score_ahp')

    # ── [5] ML Ensemble ───────────────────────────────────────────────────────
    buildings, ml_metrics = ml_ensemble(buildings, target_col='wlc_Baseline')

    # ── [6] Final Priority Score ──────────────────────────────────────────────
    buildings = calculate_final_priority(buildings, ml_metrics)

    # ── [7] Klasifikasi & Ranking ─────────────────────────────────────────────
    buildings = classify_and_rank(buildings)

    # ── [8] Monte Carlo ───────────────────────────────────────────────────────
    buildings, mc_meta = monte_carlo_sensitivity(buildings)

    # ── [9] Scenario Comparison ───────────────────────────────────────────────
    df_sens = scenario_comparison(buildings)

    # ── [10] Quadrant Analysis ────────────────────────────────────────────────
    quad_res = quadrant_analysis(buildings)

    # ── [11] Visualisasi ──────────────────────────────────────────────────────
    plot_priority_map(buildings)
    plot_interactive_map(buildings)
    plot_analysis_charts(buildings, ahp_results, ml_metrics, df_sens)
    plot_ahp_visualization(ahp_results)
    plot_monte_carlo(buildings, mc_meta)
    plot_tornado_chart(buildings)
    plot_quadrant_analysis(buildings, quad_res)

    # ── [12] Export ───────────────────────────────────────────────────────────
    export_all(buildings, ahp_results, ml_metrics, df_sens, mc_meta)

    elapsed = (datetime.now() - t_start).seconds
    log.info(f'\n✓ Task 4 selesai dalam {elapsed} detik')
    log.info(f'  Semua output tersimpan di: {OUT}')
    log.info(f'  Jalankan task4_report_generator.py untuk laporan teknis HTML.')
    return buildings


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Task 4 — Solar PV Priority Integration')
    parser.add_argument('--demo', action='store_true',
                        help='Gunakan data sintetis (jika output Task 1-3 belum ada)')
    args = parser.parse_args()
    result = main(demo=args.demo)
