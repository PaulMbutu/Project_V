"""
Terrain Viewer - a 3D satellite-draped terrain model bounded by four coordinate pins.

Recreates the "Google Maps 3D satellite" look for a small patch of the planet, but
as a real numeric mesh (elevation grid + RGB texture) you can manipulate afterwards.

The 3D mesh and the flythrough are both built from ONE pre-fetched image pair that is
cropped exactly to the pins, so what you see never grows or shrinks as you zoom.

Resolution note: because we bake one fixed image instead of streaming tiles per view,
ground detail is bounded by (area / texture pixels). Imagery zoom is therefore chosen
as deep as the tile budget allows, not as shallow as a pixel target permits.

Data sources (all keyless, no subscription, no billing account):
  * Elevation : AWS Terrain Tiles ("terrarium" PNG), s3://elevation-tiles-prod.
                Public dataset on the AWS Open Data Registry. Global, z0-15
                (~5 m/px at best), built from SRTM / 3DEP / GMTED / ETOPO1 / NED.
  * Imagery   : Esri "World Imagery" - measured to z21 (0.05 m/px) over the Alps,
                z20 over Denver, z19 over Nairobi and London. Beyond coverage the
                service returns a flat "Map data not yet available" tile, which we
                detect and report rather than displaying as if it were imagery.

Google's own tiles are never scraped (that breaks the Maps ToS). Google is reached only
through sanctioned routes: keyless links out to Maps / Earth, a keyless embeddable 2D
satellite iframe, a KML export, and - if the user supplies their own Maps Platform key -
Photorealistic 3D Tiles rendered in CesiumJS inside the app.

Run:  streamlit run main.py
"""

from __future__ import annotations

import base64
import io
import json
import math
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
import pydeck as pdk
import requests
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw

try:  # optional - only needed for the "pick on a map" input mode
    import folium
    from streamlit_folium import st_folium

    HAS_MAP_PICKER = True
except ImportError:  # pragma: no cover
    HAS_MAP_PICKER = False


# --------------------------------------------------------------------------------------
# Tile sources
#
# max_zoom values below are measured, not guessed - see the module docstring. USGS in
# particular caches only to z16 in this tile scheme and returns 404 above it, which
# previously produced silent black holes in the texture.
# --------------------------------------------------------------------------------------

TILE_SIZE = 256

TERRAIN_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
TERRAIN_MAX_ZOOM = 15

IMAGERY_SOURCES = {
    "Esri World Imagery (global, deepest)": {
        "url": (
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        "max_zoom": 21,
        "attribution": "Esri, Maxar, Earthstar Geographics, and the GIS User Community",
    },
    "USGS Imagery (USA only, capped at z16)": {
        "url": (
            "https://basemap.nationalmap.gov/arcgis/rest/services/"
            "USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
        ),
        "max_zoom": 16,
        "attribution": "USGS The National Map",
    },
    "Esri World Topo (map, not satellite)": {
        "url": (
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Topo_Map/MapServer/tile/{z}/{y}/{x}"
        ),
        "max_zoom": 19,
        "attribution": "Esri",
    },
}

HEADERS = {"User-Agent": "terrain-viewer/1.0 (streamlit; python-requests)"}
MAX_MOSAIC_PX = 4096       # memory ceiling for a stitched image
DECK_TEXTURE_PX = 4096     # WebGL textures are >= 4096 on essentially all hardware
DECK_DEM_PX = 1024
PIN_LABELS = ("A", "B", "C", "D")
EARTH_CIRCUM = 40075016.686

# Bump whenever TerrainModel gains or loses a field. A saved model outlives an edit to
# this file: Streamlit re-runs the new code while session_state still holds an instance
# built by the old class, and the first access to a newly added field raises
# AttributeError. The stamp lets us spot and discard those stale models.
MODEL_SCHEMA = 3


# --------------------------------------------------------------------------------------
# Web-Mercator maths
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BBox:
    south: float
    west: float
    north: float
    east: float

    def valid(self) -> bool:
        return (
            -85 <= self.south < self.north <= 85
            and -180 <= self.west < self.east <= 180
        )

    @property
    def center(self) -> tuple[float, float]:
        return (self.south + self.north) / 2, (self.west + self.east) / 2

    def span_km(self) -> tuple[float, float]:
        lat_mid = self.center[0]
        w = (self.east - self.west) * 111.320 * math.cos(math.radians(lat_mid))
        h = (self.north - self.south) * 110.574
        return w, h

    def key(self) -> tuple:
        return (self.south, self.west, self.north, self.east)

    def padded(self, frac: float) -> BBox:
        dlat = (self.north - self.south) * frac
        dlon = (self.east - self.west) * frac
        return BBox(
            max(-85.0, self.south - dlat), max(-180.0, self.west - dlon),
            min(85.0, self.north + dlat), min(180.0, self.east + dlon),
        )


def merc_y(lat: float) -> float:
    """Normalised Web-Mercator y in [0, 1] (0 = north pole side)."""
    lat = max(min(lat, 85.05112878), -85.05112878)
    s = math.sin(math.radians(lat))
    return 0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)


def lon_to_px(lon: float, zoom: int) -> float:
    return (lon + 180.0) / 360.0 * TILE_SIZE * (2**zoom)


def lat_to_px(lat: float, zoom: int) -> float:
    return merc_y(lat) * TILE_SIZE * (2**zoom)


def ground_res(lat: float, zoom: int) -> float:
    """Native metres per pixel of a tile at this latitude and zoom."""
    return EARTH_CIRCUM * math.cos(math.radians(lat)) / (TILE_SIZE * 2**zoom)


def mosaic_px(bbox: BBox, zoom: int) -> tuple[int, int]:
    w = lon_to_px(bbox.east, zoom) - lon_to_px(bbox.west, zoom)
    h = lat_to_px(bbox.south, zoom) - lat_to_px(bbox.north, zoom)
    return max(2, int(round(w))), max(2, int(round(h)))


def tile_range(bbox: BBox, zoom: int) -> tuple[int, int, int, int]:
    x0 = int(math.floor(lon_to_px(bbox.west, zoom) / TILE_SIZE))
    x1 = int(math.floor((lon_to_px(bbox.east, zoom) - 1e-6) / TILE_SIZE))
    y0 = int(math.floor(lat_to_px(bbox.north, zoom) / TILE_SIZE))
    y1 = int(math.floor((lat_to_px(bbox.south, zoom) - 1e-6) / TILE_SIZE))
    n = 2**zoom
    return (max(0, x0), max(0, y0), min(n - 1, x1), min(n - 1, y1))


def tile_count(bbox: BBox, zoom: int) -> int:
    x0, y0, x1, y1 = tile_range(bbox, zoom)
    return (x1 - x0 + 1) * (y1 - y0 + 1)


def zoom_for_budget(bbox: BBox, budget: int, hard_max: int) -> int:
    """Deepest zoom whose tile count fits the budget."""
    best = 0
    for z in range(0, hard_max + 1):
        if tile_count(bbox, z) <= budget:
            best = z
        else:
            break
    return best


def pick_zoom(bbox: BBox, target_px: int, max_zoom: int) -> int:
    """Shallowest zoom whose mosaic is at least `target_px` across (used for the DEM,
    where more pixels than the mesh needs would just be thrown away)."""
    for z in range(0, max_zoom + 1):
        w, h = mosaic_px(bbox, z)
        if max(w, h) >= target_px:
            return z
    return max_zoom


# --------------------------------------------------------------------------------------
# Tile fetching / mosaicking
# --------------------------------------------------------------------------------------


def is_placeholder(arr: np.ndarray) -> bool:
    """Esri's "Map data not yet available" tile: a flat grey field with small text.
    Measured std ~5.4 versus ~40-65 for real imagery."""
    return float(arr.std()) < 12.0


def _fetch_one(session: requests.Session, url: str) -> tuple[Image.Image | None, bool]:
    try:
        r = session.get(url, timeout=20)
        if r.status_code != 200 or not r.content:
            return None, False
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        return img, is_placeholder(np.asarray(img))
    except Exception:
        return None, False


@st.cache_data(show_spinner=False, max_entries=16, ttl=3600)
def fetch_mosaic(url_template: str, bbox_key: tuple, zoom: int, budget: int
                 ) -> tuple[np.ndarray, dict]:
    """Download every tile covering the bbox, stitch, crop exactly to the bbox.

    Returns ((H, W, 3) uint8 with rows uniform in Mercator y, coverage stats).
    """
    bbox = BBox(*bbox_key)
    x0, y0, x1, y1 = tile_range(bbox, zoom)
    nx, ny = x1 - x0 + 1, y1 - y0 + 1
    if nx * ny > budget:
        raise ValueError(
            f"{nx * ny:,} tiles needed at zoom {zoom}, budget is {budget:,}. "
            "Raise the tile budget, lower the imagery detail, or shrink the area."
        )

    coords = [(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]
    mosaic = Image.new("RGB", (nx * TILE_SIZE, ny * TILE_SIZE), (0, 0, 0))

    with requests.Session() as session:
        session.headers.update(HEADERS)
        urls = [url_template.format(z=zoom, x=x, y=y) for x, y in coords]
        with ThreadPoolExecutor(max_workers=24) as pool:
            results = list(pool.map(lambda u: _fetch_one(session, u), urls))

    got = blank = 0
    for (x, y), (tile, placeholder) in zip(coords, results):
        if tile is None:
            continue
        got += 1
        blank += bool(placeholder)
        if tile.size != (TILE_SIZE, TILE_SIZE):
            tile = tile.resize((TILE_SIZE, TILE_SIZE))
        mosaic.paste(tile, ((x - x0) * TILE_SIZE, (y - y0) * TILE_SIZE))

    if got == 0:
        raise ValueError(
            f"No tiles could be downloaded at zoom {zoom} — the service may not cache "
            "this deep here. Try a lower imagery detail."
        )

    ox, oy = x0 * TILE_SIZE, y0 * TILE_SIZE
    left, top = max(0, int(round(lon_to_px(bbox.west, zoom) - ox))), \
                max(0, int(round(lat_to_px(bbox.north, zoom) - oy)))
    right = max(left + 2, min(mosaic.width, int(round(lon_to_px(bbox.east, zoom) - ox))))
    bottom = max(top + 2, min(mosaic.height, int(round(lat_to_px(bbox.south, zoom) - oy))))
    out = mosaic.crop((left, top, right, bottom))

    if max(out.size) > MAX_MOSAIC_PX:  # memory ceiling
        s = MAX_MOSAIC_PX / max(out.size)
        out = out.resize((max(2, int(out.width * s)), max(2, int(out.height * s))),
                         Image.LANCZOS)

    stats = {"tiles": len(coords), "fetched": got, "blank": blank,
             "missing": len(coords) - got, "zoom": zoom}
    return np.asarray(out), stats


@st.cache_data(show_spinner=False, max_entries=64, ttl=3600)
def deepest_real_zoom(url_template: str, lat: float, lon: float, hard_max: int) -> int:
    """Probe the centre tile downward until the service returns real imagery.

    Esri serves a 200 OK placeholder past its coverage rather than a 404, so asking
    for z21 everywhere would quietly produce a flat grey model. Coordinates are rounded
    by the caller so that nudging a pin does not re-probe the network on every keystroke.
    """
    with requests.Session() as session:
        session.headers.update(HEADERS)
        for z in range(hard_max, 9, -1):
            n = 2**z
            x = int((lon + 180.0) / 360.0 * n)
            y = int(merc_y(lat) * n)
            img, placeholder = _fetch_one(session, url_template.format(z=z, x=x, y=y))
            if img is not None and not placeholder:
                return z
    return 10


def decode_terrarium(rgb: np.ndarray) -> np.ndarray:
    """Terrarium encoding: elevation_m = R * 256 + G + B / 256 - 32768."""
    a = rgb.astype(np.float64)
    return a[:, :, 0] * 256.0 + a[:, :, 1] + a[:, :, 2] / 256.0 - 32768.0


def encode_terrarium(elev: np.ndarray) -> np.ndarray:
    """Inverse of decode_terrarium, so deck.gl can eat our own DEM as one image."""
    v = np.clip(elev + 32768.0, 0.0, 65535.999)
    r = np.floor(v / 256.0)
    rem = v - r * 256.0
    g = np.floor(rem)
    b = np.floor((rem - g) * 256.0)
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


# --------------------------------------------------------------------------------------
# Mercator -> equirectangular resampling
#
# Tile rows are uniform in Mercator y, but both renderers place the image on a grid that
# is uniform in LATITUDE. Remapping the rows is what makes the pinned bounds exact.
# --------------------------------------------------------------------------------------


def mercator_rows_to_latlinear(arr: np.ndarray, bbox: BBox) -> np.ndarray:
    h = arr.shape[0]
    yn, ys = merc_y(bbox.north), merc_y(bbox.south)
    if abs(ys - yn) < 1e-15:
        return arr.astype(np.float64)

    lats = bbox.north + (np.arange(h) + 0.5) * (bbox.south - bbox.north) / h
    ymerc = np.array([merc_y(v) for v in lats])
    src = np.clip((ymerc - yn) / (ys - yn) * h - 0.5, 0, h - 1)

    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, h - 1)
    frac = (src - lo).reshape((-1,) + (1,) * (arr.ndim - 1))

    a = arr.astype(np.float64)
    return a[lo] * (1 - frac) + a[hi] * frac


def resize_float(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(
        Image.fromarray(arr.astype(np.float32), mode="F").resize(size, Image.BILINEAR)
    ).astype(np.float64)


def resize_rgb(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(arr).resize(size, Image.LANCZOS))


def to_u8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------------------
# Model assembly
# --------------------------------------------------------------------------------------


@dataclass
class TerrainModel:
    elevation: np.ndarray      # (H, W) metres, row 0 = south, rows uniform in latitude
    texture: np.ndarray        # (H, W, 3) uint8, mesh-resolution drape
    tex_hi: np.ndarray         # (Ht, Wt, 3) uint8, full imagery resolution, same bounds
    lats: np.ndarray
    lons: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    bbox: BBox
    dem_zoom: int
    img_zoom: int
    img_stats: dict


@st.cache_data(show_spinner=False, max_entries=6, ttl=3600)
def build_model(bbox_key: tuple, imagery_url: str, mesh_px: int,
                dem_zoom: int, img_zoom: int, budget: int) -> TerrainModel:
    bbox = BBox(*bbox_key)
    dem_rgb, _ = fetch_mosaic(TERRAIN_URL, bbox_key, dem_zoom, budget)
    img_rgb, stats = fetch_mosaic(imagery_url, bbox_key, img_zoom, budget)

    w_km, h_km = bbox.span_km()
    aspect = max(w_km, 1e-6) / max(h_km, 1e-6)
    if aspect >= 1:
        gw, gh = mesh_px, max(8, int(round(mesh_px / aspect)))
    else:
        gh, gw = mesh_px, max(8, int(round(mesh_px * aspect)))

    # Full-resolution texture keeps whatever the imagery zoom actually delivered.
    th, tw = img_rgb.shape[:2]

    elevation = mercator_rows_to_latlinear(
        resize_float(decode_terrarium(dem_rgb), (gw, gh)), bbox
    )
    texture = to_u8(mercator_rows_to_latlinear(resize_rgb(img_rgb, (gw, gh)), bbox))
    tex_hi = to_u8(mercator_rows_to_latlinear(img_rgb, bbox))

    elevation = np.flipud(elevation)
    texture = np.flipud(texture)
    tex_hi = np.flipud(tex_hi)

    lats = bbox.south + (np.arange(gh) + 0.5) * (bbox.north - bbox.south) / gh
    lons = bbox.west + (np.arange(gw) + 0.5) * (bbox.east - bbox.west) / gw
    lat_mid = bbox.center[0]
    x_m = (lons - bbox.west) * 111_320.0 * math.cos(math.radians(lat_mid))
    y_m = (lats - bbox.south) * 110_574.0

    stats = dict(stats, tex_px=f"{tw} × {th}")
    return TerrainModel(elevation, texture, tex_hi, lats, lons, x_m, y_m,
                        bbox, dem_zoom, img_zoom, stats)


# --------------------------------------------------------------------------------------
# Renderer 1: Plotly Mesh3d with literal per-vertex RGB
#
# NOT go.Surface: Surface can only be coloured through a colorscale, and a colorscale
# with enough stops to carry a photo (>~250) is rejected outright by plotly.js
# ("map requires nshades to be at least size N"), which renders an empty grid.
# --------------------------------------------------------------------------------------


def make_mesh_figure(model: TerrainModel, exaggeration: float, shade: float) -> go.Figure:
    z = model.elevation * exaggeration
    h, w = z.shape
    xx, yy = np.meshgrid(model.x_m, model.y_m)

    ids = np.arange(h * w).reshape(h, w)
    v00, v01 = ids[:-1, :-1].ravel(), ids[:-1, 1:].ravel()
    v10, v11 = ids[1:, :-1].ravel(), ids[1:, 1:].ravel()

    mesh = go.Mesh3d(
        x=xx.ravel(), y=yy.ravel(), z=z.ravel(),
        i=np.concatenate([v00, v00]),
        j=np.concatenate([v10, v11]),
        k=np.concatenate([v11, v01]),
        vertexcolor=model.texture.reshape(-1, 3),
        flatshading=False,
        lighting=dict(ambient=1.0 - 0.6 * shade, diffuse=shade,
                      specular=0.03, roughness=0.95, fresnel=0.1),
        lightposition=dict(x=-1e4, y=1e4, z=1e4),
        hoverinfo="skip",
        name="terrain",
    )

    span_x = float(model.x_m[-1] - model.x_m[0]) or 1.0
    span_y = float(model.y_m[-1] - model.y_m[0]) or 1.0
    span_z = float(np.ptp(model.elevation) * exaggeration) or 1.0
    biggest = max(span_x, span_y)

    fig = go.Figure(mesh)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), height=680, showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            dragmode="orbit",  # one-finger orbit, no modifier key needed
            aspectmode="manual",
            aspectratio=dict(x=span_x / biggest, y=span_y / biggest,
                             z=max(0.05, min(1.0, span_z / biggest))),
            xaxis=dict(title="East (m)", showbackground=False),
            yaxis=dict(title="North (m)", showbackground=False),
            zaxis=dict(title="Elev (m ×exag)", showbackground=False),
            camera=dict(eye=dict(x=1.25, y=-1.25, z=0.85)),
        ),
    )
    return fig


# --------------------------------------------------------------------------------------
# Renderer 2: deck.gl TerrainLayer in SINGLE-IMAGE mode
#
# NOT tile-template mode: streaming tiles clipped by `extent` snaps the visible area to
# whole tile boundaries, which change at every zoom - that is a rectangle that grows and
# shrinks as you zoom. One DEM image + one texture with explicit `bounds` pins the
# surface to exactly the four coordinates at every zoom.
# --------------------------------------------------------------------------------------


class _Deck(pdk.Deck):
    """pydeck stamps `@@=` on string layer props, telling the deck.gl JSON converter to
    evaluate them as accessor expressions. That mangles our data: URLs and the terrain
    silently never loads. Strip it off the two props that are plain image URLs."""

    def to_json(self) -> str:
        obj = json.loads(super().to_json())
        for layer in obj.get("layers", []):
            for key in ("elevationData", "texture"):
                val = layer.get(key)
                if isinstance(val, str) and val.startswith("@@="):
                    layer[key] = val[3:]
        return json.dumps(obj)


def _data_url(img: Image.Image, fmt: str, **save_kw) -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, **save_kw)
    mime = "image/png" if fmt == "PNG" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode("ascii")


@st.cache_data(show_spinner=False, max_entries=6, ttl=3600)
def deck_images(_model: TerrainModel, cache_key: tuple) -> tuple[str, str, float]:
    """Terrarium DEM (PNG, lossless) + texture (JPEG) as data URLs, north-up."""
    dem = Image.fromarray(encode_terrarium(np.flipud(_model.elevation)))
    if max(dem.size) > DECK_DEM_PX:
        s = DECK_DEM_PX / max(dem.size)
        dem = dem.resize((max(2, int(dem.width * s)), max(2, int(dem.height * s))),
                         Image.NEAREST)

    tex = Image.fromarray(np.flipud(_model.tex_hi))
    if max(tex.size) > DECK_TEXTURE_PX:
        s = DECK_TEXTURE_PX / max(tex.size)
        tex = tex.resize((int(tex.width * s), int(tex.height * s)), Image.LANCZOS)

    dem_url = _data_url(dem, "PNG", optimize=True)
    tex_url = _data_url(tex, "JPEG", quality=88)
    return dem_url, tex_url, (len(dem_url) + len(tex_url)) / 1e6


def make_deck(model: TerrainModel, exaggeration: float,
              pins: list[tuple[float, float]]) -> tuple[pdk.Deck, float]:
    bbox = model.bbox
    dem_url, tex_url, mb = deck_images(model, (bbox.key(), model.tex_hi.shape))

    terrain = pdk.Layer(
        "TerrainLayer",
        elevation_data=dem_url,
        texture=tex_url,
        bounds=[bbox.west, bbox.south, bbox.east, bbox.north],
        elevation_decoder={
            "rScaler": 256 * exaggeration, "gScaler": 1 * exaggeration,
            "bScaler": exaggeration / 256, "offset": -32768 * exaggeration,
        },
        material={"ambient": 0.6, "diffuse": 0.5, "shininess": 8},
    )
    outline = pdk.Layer(
        "PathLayer",
        data=[{"path": [list(p[::-1]) for p in pins] + [list(pins[0][::-1])]}],
        get_path="path", get_color=[255, 90, 60],
        width_min_pixels=2, get_width=3,
    )

    lat, lon = bbox.center
    frac_x = (bbox.east - bbox.west) / 360.0
    frac_y = abs(merc_y(bbox.south) - merc_y(bbox.north))
    zoom = math.log2(0.78 * 700 / (TILE_SIZE * max(frac_x, frac_y, 1e-9)))

    deck = _Deck(
        layers=[terrain, outline],
        initial_view_state=pdk.ViewState(
            latitude=lat, longitude=lon,
            zoom=max(1.0, min(20.0, zoom)), pitch=58, bearing=15,
        ),
        map_provider=None, map_style=None,
    )
    # deck.gl ships with touchRotate OFF, so there is no way to tilt on a phone.
    deck.controller = {"dragRotate": True, "touchRotate": True,
                       "touchZoom": True, "keyboard": True}
    return deck, mb


# --------------------------------------------------------------------------------------
# Crop & rotate (imagery only)
# --------------------------------------------------------------------------------------


def rotated_crop(padded: np.ndarray, pad_frac: float, angle_deg: float, scale: float
                 ) -> tuple[Image.Image, Image.Image]:
    """Rotate a crop box about the centre of the pinned area and cut it out."""
    img = Image.fromarray(padded)
    pw, ph = img.size
    inner_w = pw / (1 + 2 * pad_frac) * scale
    inner_h = ph / (1 + 2 * pad_frac) * scale
    cx, cy = pw / 2.0, ph / 2.0

    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    corners = []
    for dx, dy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        ox, oy = dx * inner_w / 2, dy * inner_h / 2
        corners.append((cx + ox * ca - oy * sa, cy + ox * sa + oy * ca))

    preview = img.copy().convert("RGB")
    draw = ImageDraw.Draw(preview, "RGBA")
    draw.polygon(corners, outline=(255, 90, 60, 255), width=max(2, pw // 300))
    draw.line([corners[0], corners[1]], fill=(80, 220, 255, 255), width=max(3, pw // 250))

    rot = img.rotate(angle_deg, resample=Image.BICUBIC, center=(cx, cy))
    left, top = cx - inner_w / 2, cy - inner_h / 2
    crop = rot.crop((int(round(left)), int(round(top)),
                     int(round(left + inner_w)), int(round(top + inner_h))))
    return preview, crop


# --------------------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------------------


def export_npz(model: TerrainModel) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(
        buf, elevation_m=model.elevation, texture_rgb=model.texture,
        texture_hires=model.tex_hi, lats=model.lats, lons=model.lons,
        x_m=model.x_m, y_m=model.y_m, bbox=np.array(model.bbox.key()),
    )
    return buf.getvalue()


def export_obj_zip(model: TerrainModel, exaggeration: float) -> bytes:
    """Textured OBJ + MTL + PNG, ready for Blender / MeshLab / three.js."""
    h, w = model.elevation.shape
    xs = np.tile(model.x_m, h)
    ys = np.repeat(model.y_m, w)
    zs = (model.elevation * exaggeration).ravel()
    us = np.tile(np.linspace(0, 1, w), h)
    vs = np.repeat(np.linspace(0, 1, h), w)

    obj = io.StringIO()
    obj.write("# terrain-viewer export\nmtllib terrain.mtl\no terrain\n")
    for x, y, z in zip(xs, ys, zs):
        obj.write(f"v {x:.3f} {z:.3f} {-y:.3f}\n")   # OBJ is Y-up
    for u, v in zip(us, vs):
        obj.write(f"vt {u:.6f} {v:.6f}\n")
    obj.write("usemtl terrain\n")
    for r in range(h - 1):
        base = r * w
        for c in range(w - 1):
            a, b = base + c + 1, base + c + 2
            d, e = base + w + c + 1, base + w + c + 2
            obj.write(f"f {a}/{a} {d}/{d} {e}/{e}\nf {a}/{a} {e}/{e} {b}/{b}\n")

    png = io.BytesIO()
    Image.fromarray(np.flipud(model.tex_hi)).save(png, format="PNG")

    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("terrain.obj", obj.getvalue())
        zf.writestr("terrain.mtl",
                    "newmtl terrain\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nd 1\nillum 1\n"
                    "map_Kd terrain.png\n")
        zf.writestr("terrain.png", png.getvalue())
    return zbuf.getvalue()


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def pins_to_bbox(pins: list[tuple[float, float]]) -> BBox:
    lats = [p[0] for p in pins]
    lons = [p[1] for p in pins]
    return BBox(min(lats), min(lons), max(lats), max(lons))


# --------------------------------------------------------------------------------------
# Google Maps / Earth hand-off
#
# Measured behaviour (response headers, checked against the live services):
#   www.google.com/maps/@...      X-Frame-Options: SAMEORIGIN  -> new tab only
#   earth.google.com/web/@...     X-Frame-Options: SAMEORIGIN  -> new tab only
#   maps.google.com/...&output=embed   no XFO                  -> embeddable, keyless, 2D
#   www.google.com/maps/embed/v1/*     401 without a key       -> needs Maps Platform key
#   tile.googleapis.com/v1/3dtiles/*   403 without a key       -> needs Maps Platform key
#
# So: the real 3D view can be linked to for free, but only embedded with a key.
# --------------------------------------------------------------------------------------


def camera_range(bbox: BBox) -> float:
    """A viewing distance that frames the pinned box."""
    w_km, h_km = bbox.span_km()
    return max(250.0, math.hypot(w_km, h_km) * 1000 * 1.3)


def google_maps_3d_url(bbox: BBox, heading: float, tilt: float) -> str:
    """Tilted satellite view. This `@lat,lon,<a>a,<y>y,<h>h,<t>t/data=!3m1!1e3` form is
    undocumented (it is what the Maps UI itself puts in the address bar), so treat it as
    best-effort; google_maps_satellite_url below is the documented, stable fallback."""
    lat, lon = bbox.center
    return (f"https://www.google.com/maps/@{lat:.6f},{lon:.6f},"
            f"{camera_range(bbox):.0f}a,35y,{heading:.1f}h,{tilt:.1f}t/data=!3m1!1e3")


def google_maps_satellite_url(bbox: BBox) -> str:
    """Documented Maps URLs API - no key, no billing, stable contract, but 2D only."""
    lat, lon = bbox.center
    frac = max((bbox.east - bbox.west) / 360.0,
               abs(merc_y(bbox.south) - merc_y(bbox.north)), 1e-9)
    zoom = max(1, min(21, round(math.log2(0.9 * 800 / (TILE_SIZE * frac)))))
    return ("https://www.google.com/maps/@?api=1&map_action=map"
            f"&center={lat:.6f},{lon:.6f}&zoom={zoom}&basemap=satellite")


def google_earth_url(bbox: BBox, heading: float, tilt: float) -> str:
    lat, lon = bbox.center
    return (f"https://earth.google.com/web/@{lat:.6f},{lon:.6f},0a,"
            f"{camera_range(bbox):.0f}d,35y,{heading:.1f}h,{tilt:.1f}t,0r")


def google_embed_url(bbox: BBox) -> str:
    """Keyless embeddable satellite map (2D). Legacy `output=embed` endpoint - it sends
    no X-Frame-Options, unlike every /maps/ page."""
    lat, lon = bbox.center
    frac = max((bbox.east - bbox.west) / 360.0,
               abs(merc_y(bbox.south) - merc_y(bbox.north)), 1e-9)
    zoom = max(1, min(21, round(math.log2(0.9 * 800 / (TILE_SIZE * frac)))))
    return (f"https://maps.google.com/maps?q={lat:.6f},{lon:.6f}"
            f"&t=k&z={zoom}&output=embed")


def kml_bytes(pins: list[tuple[float, float]], bbox: BBox,
              heading: float, tilt: float) -> bytes:
    """Four placemarks plus the enclosing box, with a tilted camera. Opens in Google
    Earth Web (earth.google.com -> File -> Import) or Google Earth Pro."""
    ring = list(pins) + [pins[0]]
    coords = " ".join(f"{lon:.7f},{lat:.7f},0" for lat, lon in ring)
    lat_c, lon_c = bbox.center
    marks = "\n".join(
        f"""  <Placemark><name>{lab}</name>
    <Point><coordinates>{lon:.7f},{lat:.7f},0</coordinates></Point></Placemark>"""
        for lab, (lat, lon) in zip(PIN_LABELS, pins)
    )
    doc = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
 <Document>
  <name>Terrain Viewer - points A-D</name>
  <LookAt>
    <longitude>{lon_c:.7f}</longitude><latitude>{lat_c:.7f}</latitude>
    <altitude>0</altitude><heading>{heading:.1f}</heading><tilt>{tilt:.1f}</tilt>
    <range>{camera_range(bbox):.0f}</range>
    <altitudeMode>relativeToGround</altitudeMode>
  </LookAt>
  <Style id="box">
    <LineStyle><color>ff3c5aff</color><width>3</width></LineStyle>
    <PolyStyle><fill>0</fill></PolyStyle>
  </Style>
{marks}
  <Placemark><name>Bounds</name><styleUrl>#box</styleUrl>
    <LineString><tessellate>1</tessellate>
      <coordinates>{coords}</coordinates>
    </LineString></Placemark>
 </Document>
</kml>
"""
    return doc.encode("utf-8")


def cesium_3d_html(bbox: BBox, pins: list[tuple[float, float]], api_key: str,
                   heading: float, tilt: float, height: int = 640) -> str:
    """Google's Photorealistic 3D Tiles rendered in CesiumJS, inside the app.

    This is the sanctioned way to put Google's actual 3D imagery in your own page.
    tile.googleapis.com answers 403 without a key, so a Maps Platform key with billing
    enabled is required - there is no keyless route to this data.
    """
    lat, lon = bbox.center
    rng = camera_range(bbox)
    ring = json.dumps([c for lat_, lon_ in list(pins) + [pins[0]] for c in (lon_, lat_)])
    ver = "1.115"
    base = f"https://cesium.com/downloads/cesiumjs/releases/{ver}/Build/Cesium/"
    return f"""
<link href="{base}Widgets/widgets.css" rel="stylesheet">
<style>
  html,body,#cesium {{ margin:0; padding:0; height:{height}px; width:100%;
                       background:#0e1117; overflow:hidden; }}
  .cesium-viewer-bottom {{ font-size:11px; }}
  #err {{ color:#ffb4a8; font:13px system-ui; padding:12px; }}
</style>
<div id="cesium"></div>
<div id="err"></div>
<script>window.CESIUM_BASE_URL = "{base}";</script>
<script src="{base}Cesium.js"></script>
<script>
(async function () {{
  const show = m => document.getElementById("err").textContent = m;
  try {{
    Cesium.Ion.defaultAccessToken = undefined;      // no Cesium ion account needed
    const viewer = new Cesium.Viewer("cesium", {{
      baseLayer: false, baseLayerPicker: false, geocoder: false, homeButton: false,
      sceneModePicker: false, navigationHelpButton: false, animation: false,
      timeline: false, infoBox: false, selectionIndicator: false, fullscreenButton: false,
      globe: false                                   // Google tiles supply the ground
    }});
    viewer.scene.skyAtmosphere.show = false;
    viewer.scene.screenSpaceCameraController.enableCollisionDetection = false;

    viewer.camera.setView({{
      destination: Cesium.Cartesian3.fromDegrees({lon:.6f}, {lat:.6f}, {rng:.0f}),
      orientation: {{ heading: Cesium.Math.toRadians({heading:.1f}),
                      pitch: Cesium.Math.toRadians({-(90 - tilt):.1f}), roll: 0 }}
    }});

    viewer.entities.add({{
      polyline: {{
        positions: Cesium.Cartesian3.fromDegreesArray({ring}),
        width: 3, clampToGround: true,
        material: Cesium.Color.fromCssColorString("#ff5a3c")
      }}
    }});

    const tileset = await Cesium.Cesium3DTileset.fromUrl(
      "https://tile.googleapis.com/v1/3dtiles/root.json?key={api_key}",
      {{ showCreditsOnScreen: true }}
    );
    viewer.scene.primitives.add(tileset);
  }} catch (e) {{
    show("Could not load Google 3D Tiles: " + e
         + "  —  check that the key is valid, that the Map Tiles API is enabled, "
         + "and that the key's HTTP referrer restrictions allow this page.");
  }}
}})();
</script>
"""


# --------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------

st.set_page_config(page_title="3D Terrain Viewer", page_icon="🏔️", layout="wide")

# Mobile: browsers claim touch gestures for page scroll before the canvas sees them, so
# a drag never reaches Plotly or deck.gl. Releasing touch-action enables one-finger
# orbit and two-finger pinch/rotate on a phone.
st.markdown(
    """
    <style>
      .stPlotlyChart, .stPlotlyChart *, .js-plotly-plot, .js-plotly-plot canvas,
      [data-testid="stDeckGlJsonChart"], [data-testid="stDeckGlJsonChart"] canvas {
          touch-action: none !important;
          -ms-touch-action: none !important;
      }
      [data-testid="stDeckGlJsonChart"] > div { background: #0e1117; border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULT_PINS = {"A": (-3.772448, 38.425414), "B": (-3.760115, 38.415829),
                "C": (-3.767310, 38.403935), "D": (-3.782554,38.417596)}

for _lab, (_lat, _lon) in DEFAULT_PINS.items():
    st.session_state.setdefault(f"pin_{_lab}_lat", _lat)
    st.session_state.setdefault(f"pin_{_lab}_lon", _lon)
st.session_state.setdefault("active_pin", "A")

# A map click arrives on the run AFTER the number inputs already exist, and Streamlit
# forbids writing a widget's key once that widget is instantiated. So the click handler
# parks the result here and reruns; we apply it at the top, before any widget is built.
_pending = st.session_state.pop("_pending_pin", None)
if _pending:
    _plab, _plat, _plon = _pending
    st.session_state[f"pin_{_plab}_lat"] = _plat
    st.session_state[f"pin_{_plab}_lon"] = _plon
    st.session_state["active_pin"] = PIN_LABELS[
        (PIN_LABELS.index(_plab) + 1) % len(PIN_LABELS)
    ]

st.title("🏔️ 3D Terrain Viewer")
st.caption(
    "Satellite-draped elevation model bounded by four pins — built from keyless, "
    "open tile services."
)

with st.sidebar:
    st.header("Points")

    for lab in PIN_LABELS:
        c0, c1, c2 = st.columns([0.6, 2.4, 2.4], vertical_alignment="center")
        marker = "🔴" if st.session_state.active_pin == lab else "　"
        c0.markdown(f"**{lab}**{marker}")
        c1.number_input(
            f"{lab} latitude", min_value=-85.0, max_value=85.0,
            key=f"pin_{lab}_lat", step=0.0010, format="%.6f",
            label_visibility="collapsed",
        )
        c2.number_input(
            f"{lab} longitude", min_value=-180.0, max_value=180.0,
            key=f"pin_{lab}_lon", step=0.0010, format="%.6f",
            label_visibility="collapsed",
        )
    st.caption("Latitude, longitude — one row per point.")

    use_map = st.toggle("Pick points on a map", value=False)
    if use_map:
        if not HAS_MAP_PICKER:
            st.warning("Install the picker:  `pip install streamlit-folium folium`")
        else:
            # No widget key here: `active_pin` is ours to write, and a widget key
            # would refuse the programmatic advance after each click.
            st.session_state.active_pin = st.radio(
                "Point to place next", PIN_LABELS,
                index=PIN_LABELS.index(st.session_state.active_pin), horizontal=True,
            )
            st.caption("Click the map to set the highlighted point; it then advances "
                       "to the next one.")

            cur = [(st.session_state[f"pin_{l}_lat"], st.session_state[f"pin_{l}_lon"])
                   for l in PIN_LABELS]
            cbb = pins_to_bbox(cur) if pins_to_bbox(cur).valid() else None
            centre = cbb.center if cbb else cur[0]

            fmap = folium.Map(location=list(centre), zoom_start=11, tiles=None)
            folium.TileLayer(
                tiles=IMAGERY_SOURCES["Esri World Imagery (global, deepest)"]["url"],
                attr="Esri", name="Satellite", max_zoom=21,
            ).add_to(fmap)
            folium.Polygon(locations=cur, color="#00e0ff", weight=2, fill=False).add_to(fmap)
            for lab, (la, lo) in zip(PIN_LABELS, cur):
                folium.Marker(
                    [la, lo],
                    icon=folium.DivIcon(html=(
                        '<div style="background:#ff3b30;color:#fff;border-radius:50%;'
                        'width:22px;height:22px;line-height:22px;text-align:center;'
                        f'font-weight:700;font-family:sans-serif">{lab}</div>')),
                ).add_to(fmap)

            out = st_folium(fmap, height=360, use_container_width=True,
                            returned_objects=["last_clicked"], key="ptpicker")
            click = (out or {}).get("last_clicked")
            if click:
                sig = (round(float(click["lat"]), 7), round(float(click["lng"]), 7))
                if st.session_state.get("_last_click") != sig:
                    st.session_state["_last_click"] = sig
                    st.session_state["_pending_pin"] = (
                        st.session_state.active_pin, sig[0], sig[1]
                    )
                    st.rerun()

    pins = [(st.session_state[f"pin_{l}_lat"], st.session_state[f"pin_{l}_lon"])
            for l in PIN_LABELS]
    bbox = pins_to_bbox(pins)

    if not bbox.valid():
        st.error("The four points enclose no area — give them differing latitudes "
                 "and longitudes.")
        st.stop()

    w_km, h_km = bbox.span_km()
    st.info(f"Bounds: **{w_km:.3f} km × {h_km:.3f} km**")

    st.header("Imagery")
    imagery_name = st.selectbox("Source", list(IMAGERY_SOURCES))
    imagery = IMAGERY_SOURCES[imagery_name]

    budget = st.select_slider(
        "Tile budget", options=[100, 250, 500, 1000, 2000, 3000], value=500,
        help="Tiles are ~20 KB each and fetched 24 at a time. This is the main "
             "control over how sharp the ground can get.",
    )

    _clat, _clon = bbox.center
    with st.spinner("Checking how deep this service covers the area…"):
        z_available = deepest_real_zoom(
            imagery["url"], round(_clat, 3), round(_clon, 3), imagery["max_zoom"]
        )
    z_budget = zoom_for_budget(bbox, budget, z_available)

    auto = st.checkbox("Use the deepest zoom the budget allows", value=True)
    if auto:
        img_zoom = max(1, z_budget)
    else:
        img_zoom = st.slider("Imagery zoom", 8, z_available, max(8, z_budget))

    lat_mid = bbox.center[0]
    tex_w, tex_h = mosaic_px(bbox, img_zoom)
    shrink = min(1.0, MAX_MOSAIC_PX / max(tex_w, tex_h))
    eff_w = max(1.0, tex_w * shrink)
    gsd = w_km * 1000 / eff_w
    capped = shrink < 1.0
    n_tiles = tile_count(bbox, img_zoom)

    st.markdown(
        f"**z{img_zoom}** · ground **{gsd:.2f} m/px** · texture {int(eff_w):,} px · "
        f"**{n_tiles:,} tiles** (~{n_tiles * 20 / 1024:.1f} MB)"
    )
    if z_available < imagery["max_zoom"]:
        st.caption(f"This service only has real imagery to z{z_available} here "
                   f"(native {ground_res(lat_mid, z_available):.2f} m/px).")
    if img_zoom < z_available:
        need = tile_count(bbox, z_available)
        st.caption(f"z{z_available} is available ({ground_res(lat_mid, z_available):.2f} "
                   f"m/px) but needs {need:,} tiles — raise the budget or shrink the area.")
    if capped:
        st.caption(f"Texture capped at {MAX_MOSAIC_PX:,} px for memory.")
    if n_tiles > 800:
        st.warning(f"{n_tiles:,} tiles is a slow first fetch (cached afterwards).")

    st.header("Model")
    detail = st.select_slider(
        "Mesh resolution", options=[96, 128, 160, 200, 256, 320], value=200,
        help="Terrain grid cells across the long axis. Elevation data itself stops at "
             "z15, so this is about mesh smoothness, not new ground detail.",
    )
    exaggeration = st.slider("Vertical exaggeration", 0.5, 8.0, 1.5, 0.1)
    shade = st.slider("Hill shading", 0.0, 1.0, 0.30, 0.05)

    dem_zoom = pick_zoom(bbox, detail, TERRAIN_MAX_ZOOM)
    st.caption(
        f"Elevation z{dem_zoom} — {ground_res(lat_mid, dem_zoom):.1f} m/px "
        f"(terrarium tops out at z{TERRAIN_MAX_ZOOM})."
    )

    render = st.button("Render 3D model", type="primary", width="stretch")


if render:
    st.session_state.pop("model", None)
    try:
        with st.spinner(f"Fetching {tile_count(bbox, img_zoom):,} imagery tiles "
                        f"and the elevation grid…"):
            st.session_state.model = build_model(
                bbox.key(), imagery["url"], detail, dem_zoom, img_zoom, budget
            )
            st.session_state.model_pins = pins
            st.session_state.model_schema = MODEL_SCHEMA
    except ValueError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Failed to build the model: {exc}")

with st.expander("🌍 Open these points in Google Maps / Earth", expanded=False):
    g1, g2 = st.columns(2)
    g_heading = g1.slider("Camera heading (°)", 0.0, 360.0, 0.0, 5.0, key="g_head")
    g_tilt = g2.slider("Camera tilt (°)", 0.0, 85.0, 60.0, 5.0, key="g_tilt")

    b1, b2, b3, b4 = st.columns(4)
    b1.link_button("Maps — 3D satellite", google_maps_3d_url(bbox, g_heading, g_tilt),
                   width="stretch", type="primary")
    b2.link_button("Google Earth — 3D", google_earth_url(bbox, g_heading, g_tilt),
                   width="stretch")
    b3.link_button("Maps — satellite (2D)", google_maps_satellite_url(bbox),
                   width="stretch")
    b4.download_button("⬇ Points as KML", kml_bytes(pins, bbox, g_heading, g_tilt),
                       file_name="terrain_points.kml",
                       mime="application/vnd.google-earth.kml+xml", width="stretch")
    st.caption(
        "These open in a new tab: google.com/maps and earth.google.com both send "
        "`X-Frame-Options: SAMEORIGIN`, so neither can be displayed inside this page. "
        "The KML drops all four points plus the box into Google Earth."
    )

    st.divider()
    st.markdown("**Show a Google map inside the app**")
    embed_mode = st.radio(
        "Embed mode",
        ["Off", "Satellite (2D, no key needed)", "Photorealistic 3D (needs a key)"],
        horizontal=True, label_visibility="collapsed",
    )

    if embed_mode.startswith("Satellite"):
        st.iframe(google_embed_url(bbox), height=460)
        st.caption(
            "Keyless `output=embed` endpoint — the one Google's own Share ▸ Embed "
            "dialog produces. It is the only Google map that can be framed without a "
            "key, and it is 2D only: this endpoint has no tilt parameter."
        )

    elif embed_mode.startswith("Photorealistic"):
        key = st.text_input(
            "Google Maps Platform API key", type="password",
            help="Kept in this session only — never written to disk. The key needs the "
                 "Map Tiles API enabled.",
        )
        if not key:
            st.info(
                "This renders Google's real Photorealistic 3D Tiles in CesiumJS, inside "
                "the app. `tile.googleapis.com` returns **403** without a key, so there "
                "is no keyless route to this data — see the notes below on what the key "
                "costs."
            )
        else:
            # components.html (not st.iframe/st.html): Cesium needs its own document
            # for workers and stylesheet, and st.iframe takes only a src URL.
            components.html(
                cesium_3d_html(bbox, pins, key, g_heading, g_tilt, height=620),
                height=650,
            )
            st.caption("Google Photorealistic 3D Tiles via CesiumJS. Attribution is "
                       "shown in the viewer and must stay visible.")

model: TerrainModel | None = st.session_state.get("model")

if model is not None and st.session_state.get("model_schema") != MODEL_SCHEMA:
    st.session_state.pop("model", None)
    st.session_state.pop("model_schema", None)
    model = None
    st.info("`main.py` changed since this model was built — press **Render 3D model** "
            "to rebuild it. (Tiles are cached, so this is quick.)")

if model is None:
    st.info("Set your four points in the sidebar, then hit **Render 3D model**.")
    st.stop()

model_pins = st.session_state.get("model_pins", pins)
stats = model.img_stats
if stats.get("blank"):
    st.warning(
        f"{stats['blank']} of {stats['tiles']} imagery tiles came back as "
        f"“no data yet” placeholders at z{stats['zoom']} — drop the imagery zoom by one."
    )
if stats.get("missing"):
    st.warning(f"{stats['missing']} of {stats['tiles']} imagery tiles failed to download.")

tab_mesh, tab_fly, tab_data = st.tabs(["3D mesh", "Flythrough", "Data, crop & export"])

with tab_mesh:
    st.plotly_chart(
        make_mesh_figure(model, exaggeration, shade), width="stretch",
        config={"scrollZoom": True, "displaylogo": False, "doubleClick": "reset"},
    )
    tri = 2 * (model.elevation.shape[0] - 1) * (model.elevation.shape[1] - 1)
    st.caption(
        f"{model.elevation.shape[1]}×{model.elevation.shape[0]} grid · {tri:,} triangles · "
        f"elevation {model.elevation.min():.0f}–{model.elevation.max():.0f} m · "
        f"texture {stats.get('tex_px')} at z{model.img_zoom} · {imagery['attribution']}"
    )
    st.caption("Drag to orbit · scroll or pinch to zoom · two fingers to rotate on mobile.")

with tab_fly:
    deck, deck_mb = make_deck(model, exaggeration, model_pins)
    st.pydeck_chart(deck, width="stretch", height=680)
    st.caption(
        f"One pre-fetched DEM + texture pinned to the exact bounds ({deck_mb:.1f} MB "
        "embedded) — the surface does not resize with zoom. Drag to pan · two fingers "
        "to tilt and rotate."
    )

with tab_data:
    st.subheader("Crop & rotate the satellite image")
    st.caption("The crop box defaults to your four points; the surrounding context is "
               "free world. Imagery only — this does not affect the 3D views.")

    c1, c2, c3 = st.columns(3)
    angle = c1.slider("Rotation (°)", -180.0, 180.0, 0.0, 0.5)
    scale = c2.slider("Crop size (× pin box)", 0.25, 1.50, 1.00, 0.05)
    context = c3.slider("Surrounding context", 0.10, 1.00, 0.45, 0.05)

    a = math.radians(abs(angle))
    need = 0.5 * scale * (abs(math.cos(a)) + abs(math.sin(a))) - 0.5
    pad_frac = max(context, need + 0.02)

    pad_bbox = bbox.padded(pad_frac)
    pad_zoom = min(model.img_zoom, zoom_for_budget(pad_bbox, budget, model.img_zoom))
    try:
        padded_raw, _ = fetch_mosaic(imagery["url"], pad_bbox.key(), pad_zoom, budget)
        padded = to_u8(mercator_rows_to_latlinear(padded_raw, pad_bbox))
        preview, crop = rotated_crop(padded, pad_frac, angle, scale)

        p1, p2 = st.columns([3, 2])
        p1.image(preview, width="stretch",
                 caption=f"Context {pad_frac:.2f}× at z{pad_zoom} · box in red, top edge in blue")
        p2.image(crop, width="stretch",
                 caption=f"Extracted crop — {crop.width}×{crop.height} px")
        st.download_button("⬇ Cropped image (PNG)", png_bytes(crop),
                           file_name=f"crop_{angle:+.1f}deg.png", mime="image/png")
    except ValueError as exc:
        st.error(str(exc))

    st.divider()
    st.subheader("Model data")

    d1, d2 = st.columns([2, 3])
    with d1:
        st.write({
            "points": {lab: f"{la:.6f}, {lo:.6f}"
                       for lab, (la, lo) in zip(PIN_LABELS, model_pins)},
            "imagery zoom": model.img_zoom,
            "texture": stats.get("tex_px"),
            "ground sampling": f"{model.bbox.span_km()[0] * 1000 / model.tex_hi.shape[1]:.2f} m/px",
            "elevation zoom": model.dem_zoom,
            "min / max elevation (m)": f"{model.elevation.min():.1f} / "
                                       f"{model.elevation.max():.1f}",
            "relief (m)": round(float(np.ptp(model.elevation)), 1),
            "mesh grid": f"{model.elevation.shape[1]} × {model.elevation.shape[0]}",
        })
    with d2:
        st.image(model.tex_hi[::-1], caption="Pinned texture (north up)", width="stretch")

    st.divider()
    if st.checkbox("Prepare downloads", help="Building these is slow at high texture "
                                             "resolutions, so it is opt-in."):
        e1, e2 = st.columns(2)
        e1.download_button("⬇ NumPy arrays (.npz)", export_npz(model),
                           file_name="terrain.npz", width="stretch")
        e2.download_button("⬇ Textured mesh (OBJ + MTL + PNG)",
                           export_obj_zip(model, exaggeration),
                           file_name="terrain_obj.zip", width="stretch")
        st.code(
            "import numpy as np\n"
            "d = np.load('terrain.npz')\n"
            "z = d['elevation_m']       # (H, W) metres, row 0 = south, rows uniform in lat\n"
            "rgb = d['texture_hires']   # full-resolution imagery over the same bounds\n"
            "x, y = d['x_m'], d['y_m']  # local metric axes\n",
            language="python",
        )
