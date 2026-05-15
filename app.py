from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Tuple
from PIL import Image
import base64
import io
import colorsys
import statistics

app = FastAPI(title="Over 2.5 Grade Analyzer", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class BBox(BaseModel):
    # Coordenadas normalizadas de 0 a 1 sobre a imagem inteira.
    # x1/y1 = canto superior esquerdo da grade
    # x2/y2 = canto inferior direito da grade
    x1: float = Field(..., ge=0, le=1)
    y1: float = Field(..., ge=0, le=1)
    x2: float = Field(..., ge=0, le=1)
    y2: float = Field(..., ge=0, le=1)


class AnalyzeRequest(BaseModel):
    image_base64: str
    bbox: Optional[BBox] = None
    rows: int = Field(12, ge=1, le=40)
    cols: int = Field(20, ge=1, le=40)
    market: str = "Over 2.5"
    league: str = ""


class CellResult(BaseModel):
    row: int
    col: int
    status: str
    score: float
    rgb: List[int]


def _decode_image(image_base64: str) -> Image.Image:
    try:
        raw = image_base64
        if "," in raw:
            raw = raw.split(",", 1)[1]
        data = base64.b64decode(raw)
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return img
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Imagem inválida: {exc}")


def _crop_by_bbox(img: Image.Image, bbox: Optional[BBox]) -> Image.Image:
    if bbox is None:
        return img
    w, h = img.size
    x1 = int(min(bbox.x1, bbox.x2) * w)
    y1 = int(min(bbox.y1, bbox.y2) * h)
    x2 = int(max(bbox.x1, bbox.x2) * w)
    y2 = int(max(bbox.y1, bbox.y2) * h)
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))
    return img.crop((x1, y1, x2, y2))


def _median_rgb(pixels: List[Tuple[int, int, int]]) -> Tuple[int, int, int]:
    if not pixels:
        return (0, 0, 0)
    rs = [p[0] for p in pixels]
    gs = [p[1] for p in pixels]
    bs = [p[2] for p in pixels]
    return (int(statistics.median(rs)), int(statistics.median(gs)), int(statistics.median(bs)))


def _sample_cell(img: Image.Image, row: int, col: int, rows: int, cols: int) -> Tuple[int, int, int]:
    w, h = img.size
    cell_w = w / cols
    cell_h = h / rows

    x_start = int(col * cell_w + cell_w * 0.25)
    x_end = int((col + 1) * cell_w - cell_w * 0.25)
    y_start = int(row * cell_h + cell_h * 0.25)
    y_end = int((row + 1) * cell_h - cell_h * 0.25)

    x_start = max(0, min(w - 1, x_start))
    x_end = max(x_start + 1, min(w, x_end))
    y_start = max(0, min(h - 1, y_start))
    y_end = max(y_start + 1, min(h, y_end))

    pixels = []
    step_x = max(1, (x_end - x_start) // 6)
    step_y = max(1, (y_end - y_start) // 6)
    pix = img.load()
    for y in range(y_start, y_end, step_y):
        for x in range(x_start, x_end, step_x):
            pixels.append(pix[x, y])
    return _median_rgb(pixels)


def _classify_rgb(rgb: Tuple[int, int, int]) -> Tuple[str, float]:
    r, g, b = rgb
    rf, gf, bf = r / 255.0, g / 255.0, b / 255.0
    h, s, v = colorsys.rgb_to_hsv(rf, gf, bf)
    hue = h * 360

    # Fundo/ célula sem resultado: muito escuro/cinza/baixo contraste
    if v < 0.18 or s < 0.18:
        return "empty", 0.0

    # Verde da grade geralmente fica entre 70 e 170 graus.
    green_score = 0.0
    if 70 <= hue <= 170 and g > r * 1.10 and g > b * 1.05:
        green_score = min(1.0, (s * 0.55) + (v * 0.45))

    # Vermelho/laranja da grade: hue perto de 0/360 ou 345/25.
    red_score = 0.0
    if (hue <= 30 or hue >= 340) and r > g * 1.05 and r > b * 1.05:
        red_score = min(1.0, (s * 0.55) + (v * 0.45))

    if green_score < 0.32 and red_score < 0.32:
        return "unknown", max(green_score, red_score)
    if green_score >= red_score:
        return "over", green_score
    return "under", red_score


def _detect_patterns(matrix: List[List[str]]) -> Dict[str, Any]:
    rows = len(matrix)
    cols = len(matrix[0]) if rows else 0

    overs_total = sum(1 for row in matrix for c in row if c == "over")
    unders_total = sum(1 for row in matrix for c in row if c == "under")
    known_total = overs_total + unders_total

    row_stats = []
    for r, row in enumerate(matrix):
        ov = row.count("over")
        un = row.count("under")
        row_stats.append({"row": r, "over": ov, "under": un, "known": ov + un})

    col_stats = []
    for c in range(cols):
        vals = [matrix[r][c] for r in range(rows)]
        ov = vals.count("over")
        un = vals.count("under")
        col_stats.append({"col": c, "over": ov, "under": un, "known": ov + un})

    hot_cols = [x for x in col_stats if x["known"] >= 3 and x["over"] >= max(2, x["under"])]
    cold_cols = [x for x in col_stats if x["known"] >= 3 and x["under"] > x["over"]]

    # Escada: sequência diagonal de overs.
    stairs = []
    for r in range(rows - 2):
        for c in range(cols - 2):
            if matrix[r][c] == "over" and matrix[r + 1][c + 1] == "over" and matrix[r + 2][c + 2] == "over":
                stairs.append({"direction": "down_right", "start": [r, c], "length": 3})
    for r in range(2, rows):
        for c in range(cols - 2):
            if matrix[r][c] == "over" and matrix[r - 1][c + 1] == "over" and matrix[r - 2][c + 2] == "over":
                stairs.append({"direction": "up_right", "start": [r, c], "length": 3})

    # X: verdes nas duas diagonais em volta de um centro.
    x_patterns = []
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            diag1 = matrix[r - 1][c - 1] == "over" and matrix[r + 1][c + 1] == "over"
            diag2 = matrix[r + 1][c - 1] == "over" and matrix[r - 1][c + 1] == "over"
            if diag1 and diag2:
                x_patterns.append({"center": [r, c]})

    # Pé de galinha: centro/âncora com pelo menos 3 overs ao redor.
    chicken_feet = []
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            count = 0
            for dr, dc in neighbors:
                if matrix[r + dr][c + dc] == "over":
                    count += 1
            if count >= 4:
                chicken_feet.append({"center": [r, c], "around_over": count})

    # Estado da primeira linha detectável como linha atual, assumindo grade de cima para baixo.
    current = row_stats[0] if row_stats else {"over": 0, "under": 0, "known": 0}
    current_overs = current["over"]
    if current_overs <= 4:
        state = "atrasada"
    elif current_overs <= 8:
        state = "normal"
    else:
        state = "saturada"

    return {
        "totals": {"over": overs_total, "under": unders_total, "known": known_total},
        "current_line": {**current, "state": state},
        "row_stats": row_stats,
        "col_stats": col_stats,
        "hot_cols": hot_cols[:8],
        "cold_cols": cold_cols[:8],
        "patterns": {
            "stairs": stairs[:20],
            "x": x_patterns[:20],
            "chicken_feet": chicken_feet[:20],
            "stairs_count": len(stairs),
            "x_count": len(x_patterns),
            "chicken_feet_count": len(chicken_feet),
        },
    }


@app.get("/health")
def health():
    return {"ok": True, "service": "over25-grade-analyzer"}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    img = _decode_image(req.image_base64)
    cropped = _crop_by_bbox(img, req.bbox)

    cells: List[CellResult] = []
    matrix: List[List[str]] = []

    for r in range(req.rows):
        row_vals = []
        for c in range(req.cols):
            rgb = _sample_cell(cropped, r, c, req.rows, req.cols)
            status, score = _classify_rgb(rgb)
            row_vals.append(status)
            cells.append(CellResult(row=r, col=c, status=status, score=round(score, 3), rgb=list(rgb)))
        matrix.append(row_vals)

    analysis = _detect_patterns(matrix)

    return {
        "ok": True,
        "league": req.league,
        "market": req.market,
        "image_size": {"width": img.size[0], "height": img.size[1]},
        "cropped_size": {"width": cropped.size[0], "height": cropped.size[1]},
        "rows": req.rows,
        "cols": req.cols,
        "matrix": matrix,
        "cells": [c.dict() for c in cells],
        "analysis": analysis,
        "summary": _make_summary(analysis),
    }


def _make_summary(analysis: Dict[str, Any]) -> List[str]:
    lines = []
    cur = analysis.get("current_line", {})
    totals = analysis.get("totals", {})
    patterns = analysis.get("patterns", {})

    lines.append(f"Linha atual: {cur.get('state', 'indefinida')} — Over {cur.get('over', 0)} / Under {cur.get('under', 0)}")
    lines.append(f"Grade lida: Over {totals.get('over', 0)} / Under {totals.get('under', 0)}")
    lines.append(f"Escadas: {patterns.get('stairs_count', 0)} | X: {patterns.get('x_count', 0)} | Pé de galinha: {patterns.get('chicken_feet_count', 0)}")

    hot_cols = analysis.get("hot_cols", [])
    if hot_cols:
        cols = ", ".join(str(x["col"]) for x in hot_cols[:5])
        lines.append(f"Colunas com força de Over: {cols}")
    else:
        lines.append("Sem colunas fortes detectadas ainda.")

    return lines
