import numpy as np
import geopandas as gpd
import torch
from pyproj import Transformer

from lib.paths import GEO

R_EARTH = 6371.0088
TO_LV95 = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)
TO_WGS = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
NE = "https://naciscdn.org/naturalearth/10m/cultural/"


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def haversine_torch(a, b):
    """a: (N, 2), b: (M, 2), [lat, lon] degrees -> (N, M) km."""
    a, b = torch.deg2rad(a), torch.deg2rad(b)
    dlat = a[:, None, 0] - b[None, :, 0]
    dlon = a[:, None, 1] - b[None, :, 1]
    h = torch.sin(dlat / 2) ** 2 + torch.cos(a[:, None, 0]) * torch.cos(b[None, :, 0]) * torch.sin(dlon / 2) ** 2
    return 2 * R_EARTH * torch.asin(torch.sqrt(h.clamp(0, 1)))


def load_boundaries():
    """Swiss border and cantons in LV95 (EPSG:2056)."""
    countries_path, cantons_path = GEO / "ne_admin0.gpkg", GEO / "ne_admin1_ch.gpkg"
    if not countries_path.exists() or not cantons_path.exists():
        countries = gpd.read_file(NE + "ne_10m_admin_0_countries.zip")
        countries[countries.ADM0_A3 == "CHE"].to_file(countries_path)
        admin1 = gpd.read_file(NE + "ne_10m_admin_1_states_provinces.zip")
        admin1[admin1.adm0_a3 == "CHE"][["name", "geometry"]].to_file(cantons_path)
    switzerland = gpd.read_file(countries_path).to_crs(2056).geometry.union_all()
    cantons = gpd.read_file(cantons_path).to_crs(2056)
    return switzerland, cantons


def to_lv95(lat, lon):
    easting, northing = TO_LV95.transform(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))
    return np.asarray(easting), np.asarray(northing)


def swiss_grid(step_m=1000, buffer_m=5000):
    """Regular LV95 grid inside Switzerland + buffer, returns (N, 2) [lat, lon]."""
    import shapely

    switzerland, _ = load_boundaries()
    poly = switzerland.buffer(buffer_m)
    x0, y0, x1, y1 = poly.bounds
    xs = np.arange(x0, x1, step_m)
    ys = np.arange(y0, y1, step_m)
    grid_x, grid_y = np.meshgrid(xs, ys)
    flat_x = grid_x.ravel()
    flat_y = grid_y.ravel()
    shapely.prepare(poly)
    keep = shapely.contains_xy(poly, flat_x, flat_y)
    lon, lat = TO_WGS.transform(flat_x[keep], flat_y[keep])
    return np.stack([lat, lon], axis=1).astype(np.float32)
