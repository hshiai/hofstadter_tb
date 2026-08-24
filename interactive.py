"""Write a self-contained linked Hofstadter/Wannier HTML figure."""

from __future__ import annotations

import json
from pathlib import Path


_FRAGMENT_START = "<!-- CODEX_FRAGMENT_START -->"
_FRAGMENT_END = "<!-- CODEX_FRAGMENT_END -->"


def save_linked_figure(
    path: Path,
    *,
    name: str,
    n_orbitals: int,
    chi: int,
    flux_min: float,
    flux_max: float,
    energy_min: float,
    energy_max: float,
    spectrum_width: int,
    spectrum_height: int,
    spectrum_mask: str,
    gap_threshold: float,
    groups: list[list[object]],
) -> Path:
    """Save linked energy-spectrum and density-flux panels."""

    payload = json.dumps(
        {
            "name": name,
            "nOrbitals": n_orbitals,
            "chi": chi,
            "fluxMin": flux_min,
            "fluxMax": flux_max,
            "energyMin": energy_min,
            "energyMax": energy_max,
            "spectrumWidth": spectrum_width,
            "spectrumHeight": spectrum_height,
            "spectrumMask": spectrum_mask,
            "gapThreshold": gap_threshold,
            "groups": groups,
        },
        separators=(",", ":"),
    ).replace("</", "<\\/")
    fragment = _fragment(payload)
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{name}: linked Hofstadter and Wannier diagrams</title>
<style>
:root {{
  color-scheme: light dark;
  --background: #ffffff;
  --foreground: #17202a;
  --muted-foreground: #65717e;
  --border: #c7ced6;
  --viz-series-1: #1769aa;
  --viz-series-2: #d1495b;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --background: #11161c;
    --foreground: #edf2f7;
    --muted-foreground: #9ca8b4;
    --border: #46515d;
    --viz-series-1: #70b7ed;
    --viz-series-2: #ff8290;
  }}
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  padding: 20px;
  background: var(--background);
  color: var(--foreground);
  font-family: ui-sans-serif, system-ui, -apple-system, sans-serif;
}}
.text-small {{ font-size: 13px; }}
.text-muted {{ color: var(--muted-foreground); }}
.tooltip {{
  background: var(--foreground);
  color: var(--background);
  padding: 6px 8px;
  border-radius: 4px;
  font-size: 12px;
  white-space: nowrap;
}}
</style>
</head>
<body>
{_FRAGMENT_START}
{fragment}
{_FRAGMENT_END}
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")
    return path


def extract_fragment(source: Path, destination: Path) -> Path:
    """Extract the conversation fragment from a generated standalone page."""

    text = source.read_text(encoding="utf-8")
    start = text.index(_FRAGMENT_START) + len(_FRAGMENT_START)
    stop = text.index(_FRAGMENT_END)
    destination.write_text(text[start:stop].strip() + "\n", encoding="utf-8")
    return destination


def _fragment(payload: str) -> str:
    return f"""<div id="linked-hofstadter-wannier-plot" class="linked-hw">
<style>
#linked-hofstadter-wannier-plot {{
  width: 100%;
  color: var(--foreground);
}}
#linked-hofstadter-wannier-plot .linked-hw-detail {{
  min-height: 1.7em;
  margin-bottom: 8px;
  color: var(--foreground);
  font-variant-numeric: tabular-nums;
}}
#linked-hofstadter-wannier-plot .linked-hw-grid {{
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 18px;
}}
#linked-hofstadter-wannier-plot .linked-hw-panel {{
  min-width: 0;
  position: relative;
}}
#linked-hofstadter-wannier-plot .linked-hw-panel h3 {{
  margin: 0 0 4px;
  color: var(--foreground);
  font-weight: 500;
}}
#linked-hofstadter-wannier-plot canvas {{
  display: block;
  width: 100%;
  height: 360px;
  cursor: crosshair;
}}
#linked-hofstadter-wannier-plot .linked-hw-legend {{
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 4px 10px 0 48px;
  color: var(--muted-foreground);
}}
#linked-hofstadter-wannier-plot .linked-hw-gradient {{
  height: 8px;
  flex: 1;
  background: linear-gradient(90deg, var(--background), var(--viz-series-1));
  border: 1px solid var(--border);
}}
#linked-hofstadter-wannier-plot .linked-hw-tooltip {{
  position: absolute;
  z-index: 2;
  visibility: hidden;
  pointer-events: none;
}}
@media (max-width: 680px) {{
  #linked-hofstadter-wannier-plot .linked-hw-grid {{
    grid-template-columns: minmax(0, 1fr);
  }}
  #linked-hofstadter-wannier-plot canvas {{ height: 330px; }}
}}
</style>
<div id="linked-hw-detail" class="linked-hw-detail text-small" aria-live="polite"></div>
<div class="linked-hw-grid">
  <section id="linked-hw-spectrum-panel" class="linked-hw-panel">
    <h3>Hofstadter spectrum</h3>
    <canvas id="linked-hw-spectrum" role="img" aria-label="Hofstadter energy spectrum; click near a gap to select it"></canvas>
    <div id="linked-hw-spectrum-tip" class="tooltip linked-hw-tooltip"></div>
  </section>
  <section id="linked-hw-wannier-panel" class="linked-hw-panel">
    <h3>Wannier diagram</h3>
    <canvas id="linked-hw-wannier" role="img" aria-label="Wannier density-flux diagram linked to the selected energy gap"></canvas>
    <div id="linked-hw-wannier-tip" class="tooltip linked-hw-tooltip"></div>
    <div class="linked-hw-legend text-small">
      <span>0</span><span class="linked-hw-gradient" aria-hidden="true"></span><span id="linked-hw-gap-max"></span><span>ΔE</span>
    </div>
  </section>
</div>
<script>
(() => {{
  const root = document.getElementById("linked-hofstadter-wannier-plot");
  const payload = {payload};
  const spectrumCanvas = document.getElementById("linked-hw-spectrum");
  const wannierCanvas = document.getElementById("linked-hw-wannier");
  const detail = document.getElementById("linked-hw-detail");
  const spectrumTip = document.getElementById("linked-hw-spectrum-tip");
  const wannierTip = document.getElementById("linked-hw-wannier-tip");
  const style = getComputedStyle(root);
  const colors = () => ({{
    foreground: style.getPropertyValue("--foreground").trim(),
    muted: style.getPropertyValue("--muted-foreground").trim(),
    border: style.getPropertyValue("--border").trim(),
    series: style.getPropertyValue("--viz-series-1").trim(),
    selected: style.getPropertyValue("--viz-series-2").trim(),
    background: style.getPropertyValue("--background").trim(),
  }});
  const spectrumMaskCanvas = document.createElement("canvas");
  spectrumMaskCanvas.width = payload.spectrumWidth;
  spectrumMaskCanvas.height = payload.spectrumHeight;
  const spectrumMaskContext = spectrumMaskCanvas.getContext("2d");
  const spectrumMaskImage = spectrumMaskContext.createImageData(
    payload.spectrumWidth,
    payload.spectrumHeight,
  );
  const spectrumMaskPixels = new Uint32Array(spectrumMaskImage.data.buffer);
  const spectrumMaskBytes = Uint8Array.from(
    atob(payload.spectrumMask),
    character => character.charCodeAt(0),
  );
  for (let index = 0; index < spectrumMaskPixels.length; index += 1) {{
    if (spectrumMaskBytes[index >> 3] & (128 >> (index & 7))) {{
      spectrumMaskPixels[index] = 0xffffffff;
    }}
  }}
  spectrumMaskContext.putImageData(spectrumMaskImage, 0, 0);
  const spectrumTintCanvas = document.createElement("canvas");
  spectrumTintCanvas.width = payload.spectrumWidth;
  spectrumTintCanvas.height = payload.spectrumHeight;
  const spectrumTintContext = spectrumTintCanvas.getContext("2d");
  const groups = payload.groups.map((raw, index) => {{
    const gaps = [];
    for (let i = 0; i < raw[2].length; i += 3) {{
      gaps.push({{
        groupIndex: index,
        r: raw[2][i],
        lower: raw[2][i + 1],
        upper: raw[2][i + 2],
      }});
    }}
    return {{
      p: raw[0],
      q: raw[1],
      flux: payload.chi * raw[0] / raw[1],
      gaps,
    }};
  }});
  const allGaps = groups.flatMap(group => group.gaps);
  const energyMin = payload.energyMin;
  const energyMax = payload.energyMax;
  let gapMax = 0;
  groups.forEach(group => {{
    group.gaps.forEach(gap => {{
      gapMax = Math.max(gapMax, gap.upper - gap.lower);
    }});
  }});
  document.getElementById("linked-hw-gap-max").textContent = format(gapMax);
  let selected = allGaps.reduce((best, gap) =>
    !best || gap.upper - gap.lower > best.upper - best.lower ? gap : best,
  null);

  function format(value) {{
    const magnitude = Math.abs(value);
    if (magnitude && (magnitude < 1e-3 || magnitude >= 1e3)) return value.toExponential(2);
    return value.toFixed(magnitude < 0.1 ? 4 : 3).replace(/0+$/, "").replace(/[.]$/, "");
  }}

  function canvasState(canvas) {{
    const rect = canvas.getBoundingClientRect();
    const ratio = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = Math.max(1, Math.round(rect.width * ratio));
    canvas.height = Math.max(1, Math.round(rect.height * ratio));
    const context = canvas.getContext("2d");
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    return {{ context, width: rect.width, height: rect.height }};
  }}

  function geometry(width, height) {{
    const pad = {{ left: 50, right: 12, top: 10, bottom: 42 }};
    return {{
      pad,
      plotWidth: Math.max(1, width - pad.left - pad.right),
      plotHeight: Math.max(1, height - pad.top - pad.bottom),
    }};
  }}

  function xPixel(flux, geom) {{
    return geom.pad.left + (flux - payload.fluxMin) /
      (payload.fluxMax - payload.fluxMin) * geom.plotWidth;
  }}

  function yPixel(value, minimum, maximum, geom) {{
    return geom.pad.top + (maximum - value) / (maximum - minimum) * geom.plotHeight;
  }}

  function drawAxes(context, width, height, yMin, yMax, yLabel) {{
    const geom = geometry(width, height);
    const color = colors();
    const fontFamily = getComputedStyle(root).fontFamily;
    context.save();
    context.strokeStyle = color.border;
    context.fillStyle = color.muted;
    context.lineWidth = 1;
    context.font = `12px ${{fontFamily}}`;
    context.textAlign = "center";
    context.textBaseline = "top";
    context.beginPath();
    context.rect(geom.pad.left, geom.pad.top, geom.plotWidth, geom.plotHeight);
    context.stroke();
    for (let i = 0; i <= 4; i += 1) {{
      const fraction = i / 4;
      const x = geom.pad.left + fraction * geom.plotWidth;
      context.beginPath();
      context.moveTo(x, geom.pad.top + geom.plotHeight);
      context.lineTo(x, geom.pad.top + geom.plotHeight + 4);
      context.stroke();
      context.fillText(format(payload.fluxMin + fraction * (payload.fluxMax - payload.fluxMin)), x, geom.pad.top + geom.plotHeight + 7);
    }}
    context.fillStyle = color.foreground;
    context.fillText("Φ/Φ₀", geom.pad.left + geom.plotWidth / 2, height - 16);
    context.fillStyle = color.muted;
    context.textAlign = "right";
    context.textBaseline = "middle";
    for (let i = 0; i <= 4; i += 1) {{
      const fraction = i / 4;
      const y = geom.pad.top + (1 - fraction) * geom.plotHeight;
      context.beginPath();
      context.moveTo(geom.pad.left - 4, y);
      context.lineTo(geom.pad.left, y);
      context.stroke();
      context.fillText(format(yMin + fraction * (yMax - yMin)), geom.pad.left - 7, y);
    }}
    context.fillStyle = color.foreground;
    context.save();
    context.translate(13, geom.pad.top + geom.plotHeight / 2);
    context.rotate(-Math.PI / 2);
    context.textAlign = "center";
    context.fillText(yLabel, 0, 0);
    context.restore();
    context.restore();
    return geom;
  }}

  function drawSpectrum() {{
    const {{ context, width, height }} = canvasState(spectrumCanvas);
    const geom = drawAxes(context, width, height, energyMin, energyMax, "E");
    const color = colors();
    spectrumTintContext.clearRect(
      0,
      0,
      payload.spectrumWidth,
      payload.spectrumHeight,
    );
    spectrumTintContext.globalCompositeOperation = "source-over";
    spectrumTintContext.fillStyle = color.foreground;
    spectrumTintContext.fillRect(
      0,
      0,
      payload.spectrumWidth,
      payload.spectrumHeight,
    );
    spectrumTintContext.globalCompositeOperation = "destination-in";
    spectrumTintContext.drawImage(spectrumMaskCanvas, 0, 0);
    spectrumTintContext.globalCompositeOperation = "source-over";
    context.save();
    context.imageSmoothingEnabled = true;
    context.drawImage(
      spectrumTintCanvas,
      geom.pad.left,
      geom.pad.top,
      geom.plotWidth,
      geom.plotHeight,
    );
    context.restore();
    if (selected) {{
      const group = groups[selected.groupIndex];
      const x = xPixel(group.flux, geom);
      const y1 = yPixel(selected.lower, energyMin, energyMax, geom);
      const y2 = yPixel(selected.upper, energyMin, energyMax, geom);
      context.save();
      context.strokeStyle = color.selected;
      context.lineWidth = 3;
      context.beginPath();
      context.moveTo(x, y1);
      context.lineTo(x, y2);
      context.moveTo(x - 4, y1);
      context.lineTo(x + 4, y1);
      context.moveTo(x - 4, y2);
      context.lineTo(x + 4, y2);
      context.stroke();
      context.restore();
    }}
  }}

  function drawWannier() {{
    const {{ context, width, height }} = canvasState(wannierCanvas);
    const geom = drawAxes(context, width, height, 0, payload.nOrbitals, "n per cell");
    const color = colors();
    context.save();
    context.fillStyle = color.series;
    allGaps.forEach(gap => {{
      const group = groups[gap.groupIndex];
      const x = xPixel(group.flux, geom);
      const y = yPixel(gap.r / group.q, 0, payload.nOrbitals, geom);
      context.globalAlpha = Math.max(0.035, Math.sqrt((gap.upper - gap.lower) / gapMax));
      context.beginPath();
      context.arc(x, y, 1.25, 0, 2 * Math.PI);
      context.fill();
    }});
    context.restore();
    if (selected) {{
      const group = groups[selected.groupIndex];
      const x = xPixel(group.flux, geom);
      const y = yPixel(selected.r / group.q, 0, payload.nOrbitals, geom);
      context.save();
      context.strokeStyle = color.selected;
      context.lineWidth = 2;
      context.beginPath();
      context.arc(x, y, 6, 0, 2 * Math.PI);
      context.moveTo(x - 9, y);
      context.lineTo(x + 9, y);
      context.moveTo(x, y - 9);
      context.lineTo(x, y + 9);
      context.stroke();
      context.restore();
    }}
  }}

  function nearestGroup(flux) {{
    let low = 0;
    let high = groups.length - 1;
    while (low < high) {{
      const middle = Math.floor((low + high) / 2);
      if (groups[middle].flux < flux) low = middle + 1;
      else high = middle;
    }}
    if (low > 0 && Math.abs(groups[low - 1].flux - flux) < Math.abs(groups[low].flux - flux)) return groups[low - 1];
    return groups[low];
  }}

  function gapNearEvent(event, panel) {{
    const canvas = panel === "spectrum" ? spectrumCanvas : wannierCanvas;
    const rect = canvas.getBoundingClientRect();
    const geom = geometry(rect.width, rect.height);
    const x = Math.max(geom.pad.left, Math.min(geom.pad.left + geom.plotWidth, event.clientX - rect.left));
    const y = Math.max(geom.pad.top, Math.min(geom.pad.top + geom.plotHeight, event.clientY - rect.top));
    const flux = payload.fluxMin + (x - geom.pad.left) / geom.plotWidth * (payload.fluxMax - payload.fluxMin);
    const group = nearestGroup(flux);
    if (!group || !group.gaps.length) return null;
    const value = panel === "spectrum"
      ? energyMax - (y - geom.pad.top) / geom.plotHeight * (energyMax - energyMin)
      : payload.nOrbitals - (y - geom.pad.top) / geom.plotHeight * payload.nOrbitals;
    return group.gaps.reduce((best, gap) => {{
      const location = panel === "spectrum" ? 0.5 * (gap.lower + gap.upper) : gap.r / group.q;
      return !best || Math.abs(location - value) < Math.abs((panel === "spectrum" ? 0.5 * (best.lower + best.upper) : best.r / group.q) - value) ? gap : best;
    }}, null);
  }}

  function updateDetail() {{
    if (!selected) return;
    const group = groups[selected.groupIndex];
    const size = selected.upper - selected.lower;
    const middle = 0.5 * (selected.lower + selected.upper);
    detail.textContent = `Selected gap: Φ/Φ₀ = ${{payload.chi * group.p}}/${{group.q}} = ${{format(group.flux)}} · r = ${{selected.r}} · n = ${{format(selected.r / group.q)}} · ΔE = ${{format(size)}} · E_mid = ${{format(middle)}}`;
  }}

  function selectGap(gap) {{
    if (!gap) return;
    selected = gap;
    updateDetail();
    drawSpectrum();
    drawWannier();
  }}

  function showTooltip(event, panel, tooltip) {{
    const gap = gapNearEvent(event, panel);
    if (!gap) {{ tooltip.style.visibility = "hidden"; return; }}
    const group = groups[gap.groupIndex];
    tooltip.textContent = `${{payload.chi * group.p}}/${{group.q}} · n=${{format(gap.r / group.q)}} · ΔE=${{format(gap.upper - gap.lower)}}`;
    tooltip.style.visibility = "visible";
    const parent = tooltip.parentElement.getBoundingClientRect();
    const canvas = panel === "spectrum" ? spectrumCanvas : wannierCanvas;
    const rect = canvas.getBoundingClientRect();
    const geom = geometry(rect.width, rect.height);
    const markX = xPixel(group.flux, geom) + rect.left - parent.left;
    const markValue = panel === "spectrum" ? 0.5 * (gap.lower + gap.upper) : gap.r / group.q;
    const markY = yPixel(markValue, panel === "spectrum" ? energyMin : 0, panel === "spectrum" ? energyMax : payload.nOrbitals, geom) + rect.top - parent.top;
    const left = Math.max(4, Math.min(parent.width - tooltip.offsetWidth - 4, markX + 8));
    const top = Math.max(4, Math.min(parent.height - tooltip.offsetHeight - 4, markY - tooltip.offsetHeight - 7));
    tooltip.style.left = `${{left}}px`;
    tooltip.style.top = `${{top}}px`;
  }}

  spectrumCanvas.addEventListener("click", event => selectGap(gapNearEvent(event, "spectrum")));
  wannierCanvas.addEventListener("click", event => selectGap(gapNearEvent(event, "wannier")));
  spectrumCanvas.addEventListener("mousemove", event => showTooltip(event, "spectrum", spectrumTip));
  wannierCanvas.addEventListener("mousemove", event => showTooltip(event, "wannier", wannierTip));
  spectrumCanvas.addEventListener("mouseleave", () => {{ spectrumTip.style.visibility = "hidden"; }});
  wannierCanvas.addEventListener("mouseleave", () => {{ wannierTip.style.visibility = "hidden"; }});
  const observer = new ResizeObserver(() => {{ drawSpectrum(); drawWannier(); }});
  observer.observe(root);
  updateDetail();
  drawSpectrum();
  drawWannier();
}})();
</script>
</div>"""
