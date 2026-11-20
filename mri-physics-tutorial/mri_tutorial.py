#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MRI physics, step by step: a 3-screen, 3D tutorial.

Screen A  patient and scanner (3D, zoomable)     what the hardware and the spins are doing
Screen B  what the machine sees                  coil voltage, ADC samples, k-space, pulse sequence
Screen C  what the researcher sees               console numbers, voxel grid, the image
Strip     the decoder between B and C            demodulate -> digitize -> place in k-space -> 2D FFT -> image

Run:   python mri_tutorial.py            (interactive window, autoplay, pauses after every phase)
       python mri_tutorial.py --auto     (never pauses between phases)
       python mri_tutorial.py --export out_dir [--only 3.2]   (headless: write PNG frames of every step)

Keys:  Space pause/play   Right next step   Shift+Right next phase   Left previous step
       Shift+Left previous phase   R restart step   Home whole machine   1 whole  2 patient  3 head  4 voxels
       Mouse: left-drag rotate, wheel zoom, middle/shift-drag pan (always available).
"""
import sys, os, math, time, argparse
import numpy as np

# ----------------------------------------------------------------------------------------------
# constants and small physics helpers
# ----------------------------------------------------------------------------------------------
GAMMA_MHZ_T = 42.577                       # gyromagnetic ratio of 1H / (2 pi), MHz per tesla
H_PLANCK, K_B = 6.626e-34, 1.381e-23

def larmor_mhz(B):                         # f = gamma_bar * B
    return GAMMA_MHZ_T * B

def boltzmann_excess(B, T=310.0):          # (N_parallel - N_anti) / N  at body temperature
    return H_PLANCK * GAMMA_MHZ_T * 1e6 * B / (2 * K_B * T)

def smooth(x):                             # smoothstep on [0,1]
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)

# ----------------------------------------------------------------------------------------------
# Biot-Savart for stacks of circular loops (axisymmetric) -> B0 field lines
# ----------------------------------------------------------------------------------------------
def _ellip(m):
    m = np.minimum(m, 0.999999)
    a = np.ones_like(m); b = np.sqrt(1 - m); s = m / 2.0; p = 1.0
    for _ in range(12):
        an = (a + b) / 2; cn = (a - b) / 2; b = np.sqrt(a * b); a = an; s = s + p * cn * cn; p *= 2
    K = np.pi / (2 * a)
    return K, K * (1 - s)

def loops_field(x, r, z0, I):
    """Field of unit-radius loops at axial positions z0 (array) with currents I, at (x, r)."""
    dx = x - z0
    if r < 1e-3:
        return float(np.sum(I * np.pi / (1 + dx * dx) ** 1.5)), 0.0
    s1 = (1 + r) ** 2 + dx * dx
    s2 = np.maximum((1 - r) ** 2 + dx * dx, 2e-3)
    K, E = _ellip(4 * r / s1)
    q = 1 / np.sqrt(s1)
    bx = q * (K + ((1 - r * r - dx * dx) / s2) * E)
    br = q * dx / r * (-K + ((1 + r * r + dx * dx) / s2) * E)
    return float(np.sum(I * bx)), float(np.sum(I * br))

def trace_line(x0, r0, z0, I, h=0.04, maxn=1600, xmax=4.5, rmax=3.2):
    pts = [(x0, r0)]; x, r = x0, r0; closed = False
    for i in range(maxn):
        bx, br = loops_field(x, r, z0, I); m = math.hypot(bx, br) or 1.0
        kx, kr = bx / m, br / m
        bx2, br2 = loops_field(x + kx * h / 2, max(r + kr * h / 2, 1e-3), z0, I); m2 = math.hypot(bx2, br2) or 1.0
        x += bx2 / m2 * h; r = max(r + br2 / m2 * h, 1e-3); pts.append((x, r))
        if abs(x) > xmax or r > rmax: break
        if i > 20 and math.hypot(x - x0, r - r0) < 0.05: closed = True; break
    return np.array(pts), closed

def b0_field_lines(a=0.62, zs=(-0.6, -0.45, -0.15, 0.15, 0.45, 0.6), n_seed=6, n_az=6):
    """3D B0 streamlines for a solenoid of radius a (m). Returns list of (n,3) arrays in metres (z = bore axis)."""
    z0 = np.array(zs) / a; I = np.ones(len(zs))
    lines2d = []
    for k in range(1, n_seed + 1):
        r0 = 0.5 * math.sqrt(k / n_seed)
        fwd, closed = trace_line(0.0, r0, z0, I)
        if not closed:
            bwd, _ = trace_line(0.0, r0, z0, I, h=-0.04)
            fwd = np.vstack([bwd[::-1], fwd[1:]])
        lines2d.append(fwd)
    out = []
    for li, l in enumerate(lines2d):
        for j in range(n_az):
            ph = (j + 0.5 * (li % 2)) * 2 * np.pi / n_az
            out.append(np.c_[l[:, 1] * np.cos(ph) * a, l[:, 1] * np.sin(ph) * a, l[:, 0] * a])
    axis = np.c_[np.zeros(60), np.zeros(60), np.linspace(-2.6, 2.6, 60)]
    out.append(axis)
    return out

def axis_b0_profile(zs, a=0.62, zc=(-0.6, -0.45, -0.15, 0.15, 0.45, 0.6), B0=3.0):
    z0 = np.array(zc) / a; I = np.ones(len(zc))
    b = np.array([loops_field(z / a, 0.0, z0, I)[0] for z in zs]); return B0 * b / b[len(b) // 2]

# ----------------------------------------------------------------------------------------------
# head phantom with tissue parameters (3 T-like values, ms) -> weighted images, k-space
# ----------------------------------------------------------------------------------------------
TISSUES = {   # name: (PD, T1 ms, T2 ms, ADC 1e-3 mm^2/s, colour)
    'CSF':   (1.00, 4000., 2000., 3.0, '#6cc4ff'),
    'GM':    (0.80, 1330.,  110., 0.80, '#b0a0ff'),
    'WM':    (0.70,  830.,   80., 0.70, '#ffd27a'),
    'Fat':   (0.90,  380.,   70., 0.10, '#ff8a5c'),
    'Lesion':(0.90, 1500.,  150., 0.40, '#ff5c8a'),
    'Bone':  (0.05,  500.,    3., 0.10, '#888888'),
}

def make_phantom(n=128):
    y, x = np.mgrid[-1:1:n * 1j, -1:1:n * 1j]
    lab = np.zeros((n, n), dtype=int)   # 0 air, 1 CSF, 2 GM, 3 WM, 4 Fat, 5 Lesion, 6 Bone
    def ell(cx, cy, rx, ry): return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1
    lab[ell(0, 0, 0.88, 0.98)] = 4          # scalp fat
    lab[ell(0, 0, 0.80, 0.90)] = 6          # skull
    lab[ell(0, 0, 0.74, 0.84)] = 1          # CSF layer
    lab[ell(0, 0, 0.70, 0.80)] = 2          # grey matter
    lab[ell(0, 0, 0.50, 0.60)] = 3          # white matter
    lab[ell(-0.11, 0.04, 0.07, 0.22)] = 1   # ventricles
    lab[ell(0.11, 0.04, 0.07, 0.22)] = 1
    lab[ell(0.33, -0.30, 0.10, 0.10)] = 5   # a lesion
    rng = np.random.default_rng(3)
    sulci = (np.sin(22 * x + 4 * np.cos(9 * y)) * np.cos(19 * y) > 0.55) & (lab == 2)
    lab[sulci] = 1
    names = [None, 'CSF', 'GM', 'WM', 'Fat', 'Lesion', 'Bone']
    pd = np.zeros((n, n)); t1 = np.ones((n, n)); t2 = np.ones((n, n)); adc = np.zeros((n, n))
    for i in range(1, 7):
        p, a1, a2, ad, _ = TISSUES[names[i]]; m = lab == i
        pd[m], t1[m], t2[m], adc[m] = p, a1, a2, ad
    return dict(n=n, lab=lab, pd=pd, t1=t1, t2=t2, adc=adc, x=x, y=y)

PH = make_phantom(128)

def weighted(TR, TE, ph=PH):
    img = ph['pd'] * (1 - np.exp(-TR / ph['t1'])) * np.exp(-TE / ph['t2'])
    return img / 0.75

def to_kspace(img):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))

def from_kspace(k):
    return np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k))))
# ----------------------------------------------------------------------------------------------
# 3D world: scanner, table, patient, electronics  (units: metres; z = bore axis, y = up, x = lateral)
# ----------------------------------------------------------------------------------------------
import pyvista as pv

BG = '#0b0f14'
COL = dict(shell='#e8edf2', cryo='#7fd4ff', b0='#3b82f6', b0s='#93c5fd', shim='#34d399',
           gx='#f59e0b', gy='#a3e635', gz='#e879f9', rf='#facc15', rx='#22d3ee', table='#9aa4af',
           skin='#e9b99a', preamp='#f87171', adc='#c084fc', pc='#94a3b8', room='#64748b', spin='#fb923c')
BORE_R, BORE_L = 0.35, 1.6
HEAD_Y = -0.045                             # head centre height
Z_HOME = -2.7                               # head z when the patient is outside

def tube(points, r=0.006, n_sides=8):
    pts = np.asarray(points, float)
    if len(pts) < 3:
        pts = np.vstack([pts[0], (pts[0] + pts[-1]) / 2, pts[-1]])
    return pv.Spline(pts, len(pts)).tube(radius=r, n_sides=n_sides)

def ring_pts(z, R, n=72, a0=0.0, a1=2 * np.pi, y0=0.0, x0=0.0):
    a = np.linspace(a0, a1, n)
    return np.c_[x0 + R * np.cos(a), y0 + R * np.sin(a), np.full(n, z)]

def ring(z, R, r=0.006, n=72):
    return tube(ring_pts(z, R, n), r)

def box(c, s):
    return pv.Box(bounds=(c[0] - s[0] / 2, c[0] + s[0] / 2, c[1] - s[1] / 2, c[1] + s[1] / 2, c[2] - s[2] / 2, c[2] + s[2] / 2))

def saddle(center_deg, width_deg, z0, z1, R, r=0.006, n=24):
    a0, a1 = np.radians(center_deg - width_deg / 2), np.radians(center_deg + width_deg / 2)
    top = ring_pts(z1, R, n, a0, a1); bot = ring_pts(z0, R, n, a1, a0)
    pts = np.vstack([top, bot, top[:1]])
    return tube(pts, r)

class World:
    def __init__(self, plotter):
        self.p = plotter; self.parts = {}; self.base_op = {}; self.target_op = {}; self.table_z = Z_HOME
        self.moving = []     # names that follow the table
        self._build()

    # -- helper to register a part
    def add(self, name, mesh, color, opacity=1.0, follows=False, **kw):
        act = self.p.add_mesh(mesh, color=color, opacity=opacity, name=name, pickable=False, **kw)
        self.parts[name] = act; self.base_op[name] = opacity; self.target_op[name] = opacity
        if follows: self.moving.append(name)
        return act

    def _build(self):
        clip = lambda m: m.clip(normal=(1, 0, 0), origin=(0.02, 0, 0))      # cutaway: remove x>0 half
        # outer cover and end plates
        cover = pv.Cylinder(radius=1.02, height=BORE_L, direction=(0, 0, 1), resolution=120, capping=False)
        self.add('cover', clip(cover), COL['shell'], 0.28, smooth_shading=True)
        for s, z in (('front', BORE_L / 2), ('back', -BORE_L / 2)):
            d = pv.Disc(center=(0, 0, z), inner=BORE_R + 0.01, outer=1.02, normal=(0, 0, 1), r_res=2, c_res=120)
            self.add('plate_' + s, clip(d), COL['shell'], 0.55, smooth_shading=True)
        # bore tube (patient bore)
        bore = pv.Cylinder(radius=BORE_R, height=BORE_L, direction=(0, 0, 1), resolution=96, capping=False)
        self.add('bore', clip(bore), '#dfe7ee', 0.22, smooth_shading=True)
        # cryostat: helium vessel (shell) and cold head
        he = pv.Cylinder(radius=0.90, height=1.45, direction=(0, 0, 1), resolution=96, capping=False)
        self.add('cryo', clip(he), COL['cryo'], 0.16, smooth_shading=True)
        self.add('coldhead', pv.Cylinder(center=(0, 1.2, -0.3), direction=(0, 1, 0), radius=0.09, height=0.5, resolution=32), '#cbd5e1', 0.9, smooth_shading=True)
        # B0 superconducting coils (main) and shield coils
        z_main = (-0.6, -0.45, -0.15, 0.15, 0.45, 0.6)
        self.add('b0', pv.merge([ring(z, 0.62, 0.022) for z in z_main]), COL['b0'], 0.95, smooth_shading=True)
        self.add('b0shield', pv.merge([ring(z, 0.82, 0.016) for z in (-0.5, 0.5)]), COL['b0s'], 0.8, smooth_shading=True)
        # shim: small iron plates in trays around r = 0.52
        shims = []
        for k in range(12):
            a = k * np.pi / 6
            for z in (-0.35, 0.0, 0.35):
                b = box((0.52 * math.cos(a), 0.52 * math.sin(a), z), (0.05, 0.05, 0.1)); shims.append(b)
        self.add('shim', pv.merge(shims), COL['shim'], 0.95, smooth_shading=True)
        # gradient assembly: shell + Z (Maxwell pair) + X, Y (Golay saddles)
        gshell = pv.Cylinder(radius=0.47, height=1.3, direction=(0, 0, 1), resolution=96, capping=False)
        self.add('gshell', clip(gshell), '#8b95a1', 0.12, smooth_shading=True)
        zl = []
        for zc in (-0.30, 0.30):
            for dz, R in ((0.0, 0.44), (0.035, 0.44), (0.07, 0.44)):
                zl.append(ring(zc + math.copysign(dz, zc), R, 0.008))
        self.add('gz', pv.merge(zl), COL['gz'], 1.0, smooth_shading=True)
        gx = [saddle(c, 110, z0, z1, 0.45, 0.008) for c in (0, 180) for (z0, z1) in ((0.05, 0.5), (-0.5, -0.05))]
        gy = [saddle(c, 110, z0, z1, 0.43, 0.008) for c in (90, 270) for (z0, z1) in ((0.05, 0.5), (-0.5, -0.05))]
        self.add('gx', pv.merge(gx), COL['gx'], 1.0, smooth_shading=True)
        self.add('gy', pv.merge(gy), COL['gy'], 1.0, smooth_shading=True)
        # body RF transmit coil (birdcage) inside the gradient assembly
        bc = [ring(z, 0.385, 0.007) for z in (-0.32, 0.32)]
        for k in range(16):
            a = k * 2 * np.pi / 16; bc.append(tube([(0.385 * math.cos(a), 0.385 * math.sin(a), -0.32), (0.385 * math.cos(a), 0.385 * math.sin(a), 0.32)], 0.005))
        self.add('bodycoil', pv.merge(bc), COL['rf'], 1.0, smooth_shading=True)
        # table
        self.add('table', box((0, -0.17, -0.5), (0.52, 0.04, 2.7)), COL['table'], 0.95, follows=True, smooth_shading=True)
        self.add('pedestal', box((0, -0.62, -1.05), (0.55, 0.9, 0.9)), '#59636e', 0.95, smooth_shading=True)
        # patient (supine, head toward +z)
        self.add('patient', self._patient(), COL['skin'], 0.5, follows=True, smooth_shading=True)
        # head receive coil (phased array cage) and its preamp
        hc = [ring(z, 0.135, 0.004, 48) for z in (-0.12, 0.0, 0.12)]
        for k in range(14):
            a = k * 2 * np.pi / 14 + 0.1
            hc.append(tube([(0.135 * math.cos(a), 0.135 * math.sin(a) + HEAD_Y, -0.12), (0.135 * math.cos(a), 0.135 * math.sin(a) + HEAD_Y, 0.12)], 0.003))
        for m in hc:                       # rings were built around y=0; lift to head height
            pass
        hcm = pv.merge(hc[:3]); hcm.points = hcm.points + np.array([0, HEAD_Y, 0])
        self.add('headcoil', pv.merge([hcm, pv.merge(hc[3:])]), COL['rx'], 1.0, follows=True, smooth_shading=True)
        self.add('preamp', box((0, -0.12, -0.17), (0.08, 0.04, 0.06)), COL['preamp'], 1.0, follows=True, smooth_shading=True)
        # electronics: receiver / ADC cabinet, computer and monitor, RF-shielded room
        self.add('adc', box((2.3, -0.45, 0.6), (0.55, 1.3, 0.55)), COL['adc'], 0.9, smooth_shading=True)
        self.add('pc', box((3.0, -0.55, -0.7), (0.5, 1.0, 0.6)), COL['pc'], 0.9, smooth_shading=True)
        self.add('monitor', box((2.74, 0.10, -0.7), (0.05, 0.38, 0.55)), '#0ea5e9', 0.95, smooth_shading=True)
        room = pv.Box(bounds=(-3.2, 4.0, -1.1, 1.8, -5.6, 2.4)).extract_all_edges()
        self.add('room', room, COL['room'], 0.35, line_width=1.5)
        floor = pv.Plane(center=(0.4, -1.1, -1.6), direction=(0, 1, 0), i_size=7.2, j_size=8.0)
        self.add('floor', floor, '#18202a', 1.0)
        self.set_table(Z_HOME)

    def _patient(self):
        parts = []
        parts.append(pv.ParametricEllipsoid(0.075, 0.095, 0.105, u_res=40, v_res=40, w_res=40).translate((0, HEAD_Y, 0.0)))
        parts.append(pv.Cylinder(center=(0, HEAD_Y - 0.03, -0.14), direction=(0, 0, 1), radius=0.05, height=0.12, resolution=24))
        parts.append(pv.ParametricEllipsoid(0.19, 0.1, 0.34, u_res=40, v_res=40, w_res=40).translate((0, -0.065, -0.46)))
        parts.append(pv.ParametricEllipsoid(0.17, 0.09, 0.15, u_res=40, v_res=40, w_res=40).translate((0, -0.075, -0.88)))
        for sx in (-0.085, 0.085):
            parts.append(pv.Cylinder(center=(sx, -0.095, -1.45), direction=(0, 0, 1), radius=0.07, height=0.95, resolution=24))
            parts.append(pv.Cylinder(center=(sx * 2.7, -0.09, -0.58), direction=(0, 0, 1), radius=0.045, height=0.8, resolution=20))
        return pv.merge(parts)

    def set_table(self, z):
        self.table_z = z
        for n in self.moving:
            self.parts[n].position = (0, 0, z)
        self._cable()

    def head_pos(self):
        return np.array([0.0, HEAD_Y, self.table_z])

    def _cable(self):
        z = self.table_z
        path = np.array([[0, -0.12, z - 0.17], [0.0, -0.20, z - 0.6], [0.5, -0.9, min(z - 0.9, -0.9)], [1.4, -1.02, 0.0], [2.3, -1.0, 0.6], [2.3, -0.75, 0.6]])
        self.cable_path = path
        m = tube(path, 0.008); old = self.target_op.get('cable', 1.0)
        if 'cable' in self.parts: self.p.remove_actor(self.parts['cable'])
        self.parts['cable'] = self.p.add_mesh(m, color='#fca5a5', name='cable', pickable=False, opacity=old); self.base_op['cable'] = 1.0; self.target_op['cable'] = old
        self.parts['cable'].SetVisibility(old > 0.01)
        # second cable ADC -> computer
        p2 = np.array([[2.3, -0.75, 0.6], [2.5, -1.02, 0.3], [3.0, -1.02, -0.5], [3.0, -0.9, -0.7]])
        self.cable2_path = p2
        if 'cable2' not in self.parts:
            self.parts['cable2'] = self.p.add_mesh(tube(p2, 0.008), color='#d8b4fe', name='cable2', pickable=False); self.base_op['cable2'] = 1.0; self.target_op['cable2'] = 1.0

    # -- opacity management ------------------------------------------------------------------
    def show_only(self, spec, default=0.0):
        """spec: dict name -> opacity (None = base).  Others fade to `default`."""
        for n in self.parts:
            v = spec.get(n, default)
            self.target_op[n] = self.base_op[n] if v is None else v

    def step_opacity(self, k=0.18):
        for n, a in self.parts.items():
            cur = a.prop.opacity; tgt = self.target_op[n]
            if abs(cur - tgt) > 0.004:
                a.prop.opacity = cur + (tgt - cur) * k
            elif cur != tgt:
                a.prop.opacity = tgt
            a.SetVisibility(a.prop.opacity > 0.01)

    def snap_opacity(self):
        for n, a in self.parts.items():
            a.prop.opacity = self.target_op[n]; a.SetVisibility(a.prop.opacity > 0.01)

    def tooth(self):
        return None

# camera presets: (position, focal point)
CAMS = {
    'whole':   ((6.8, 3.2, -7.4), (0.9, -0.45, -1.0)),
    'scanner': ((4.3, 1.9, -3.6), (0.0, 0.0, 0.0)),
    'patient': ((2.3, 1.0, -1.9), (0.0, -0.05, -0.8)),
    'head':    ((0.95, 0.50, -1.45), (0.0, HEAD_Y, 0.0)),
    'voxels':  ((0.20, 0.10, -0.34), (0.0, HEAD_Y, 0.0)),
    'coils':   ((1.5, 0.75, -1.6), (0.0, 0.0, 0.0)),
    'cabinet': ((5.0, 1.6, 2.6), (2.5, -0.4, 0.0)),
    'front':   ((0.0, 0.2, -3.2), (0.0, 0.0, 0.0)),
    'side':    ((3.4, 0.2, 0.0), (0.0, 0.0, 0.0)),
}
# ----------------------------------------------------------------------------------------------
# components (3D scene pieces that steps switch on and drive)
# ----------------------------------------------------------------------------------------------
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import FancyBboxPatch
import matplotlib.cm as mcm
import matplotlib

def hex2rgb(h):
    h = h.lstrip('#'); return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], np.uint8)

def hsv_rgb(h, s=0.75, v=1.0):
    h = np.asarray(h, float) % 1.0
    i = (h * 6).astype(int) % 6; f = h * 6 - np.floor(h * 6)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    r = np.choose(i, [v + 0 * h, q, p, p, t, v + 0 * h]); g = np.choose(i, [t, v + 0 * h, v + 0 * h, q, p, p]); b = np.choose(i, [p, p, t, v + 0 * h, v + 0 * h, q])
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)

def rot_x_to(v):
    v = np.asarray(v, float); n = np.linalg.norm(v)
    if n < 1e-12: return np.eye(3)
    v = v / n; x = np.array([1., 0, 0]); c = float(x @ v)
    if c > 0.999999: return np.eye(3)
    if c < -0.999999: return np.diag([-1., 1, -1])
    ax = np.cross(x, v); s = np.linalg.norm(ax); ax /= s
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + K * s + K @ K * (1 - c)

class Comp:
    def __init__(self, app): self.app = app; self.actors = []; self.on = False
    def show(self, v):
        self.on = v
        for a in self.actors: a.SetVisibility(bool(v))
    def kill(self):
        for a in self.actors: self.app.p.remove_actor(a)

class Spins(Comp):
    """~150 spin arrows inside the head; coloured per spin."""
    N = 150
    def __init__(self, app):
        super().__init__(app); rng = np.random.default_rng(5); pts = []
        while len(pts) < self.N:
            q = rng.uniform(-1, 1, 3)
            if q @ q <= 1: pts.append(q * np.array([0.058, 0.078, 0.082]))
        self.rel = np.array(pts); N = self.N; self.L = 0.036
        lines = np.hstack([np.full((N, 1), 2), np.arange(N)[:, None], (np.arange(N) + N)[:, None]]).ravel()
        self.m = pv.PolyData(np.zeros((2 * N, 3)), lines=lines); self.m.point_data['rgb'] = np.zeros((2 * N, 3), np.uint8)
        self.t = pv.PolyData(np.zeros((N, 3))); self.t.point_data['rgb'] = np.zeros((N, 3), np.uint8)
        a1 = app.p.add_mesh(self.m, scalars='rgb', rgb=True, line_width=3, name='spin_l', pickable=False)
        a2 = app.p.add_mesh(self.t, scalars='rgb', rgb=True, point_size=7, render_points_as_spheres=True, name='spin_t', pickable=False)
        self.actors = [a1, a2]; self.show(False)
    def set(self, vecs, rgb, scale=1.0):
        N = self.N; v = np.asarray(vecs, float); c = self.app.w.head_pos() + np.array([0, 0, 0])
        n = np.linalg.norm(v, axis=1, keepdims=True); vn = v / np.maximum(n, 1e-9) * np.minimum(n, 1.0) * scale
        base = c + self.rel; P = np.vstack([base - 0.5 * self.L * vn, base + 0.5 * self.L * vn])
        rgb = np.asarray(rgb, np.uint8)
        if rgb.ndim == 1: rgb = np.tile(rgb, (N, 1))
        self.m.points = P; self.m.point_data['rgb'] = np.vstack([rgb, rgb]); self.t.points = P[N:]; self.t.point_data['rgb'] = rgb

class MVec(Comp):
    """one big arrow (net magnetization) anchored at a point."""
    def __init__(self, app, color='#fbbf24', ref=0.17):
        super().__init__(app); self.ref = ref
        a = pv.Arrow(start=(0, 0, 0), direction=(1, 0, 0), tip_length=0.28, tip_radius=0.1, shaft_radius=0.035, scale=1.0)
        self.actor = app.p.add_mesh(a, color=color, smooth_shading=True, name='mvec_' + color, pickable=False); self.actors = [self.actor]; self.show(False)
    def set(self, origin, vec):
        n = np.linalg.norm(vec); M = np.eye(4)
        if n < 1e-6: self.actor.SetVisibility(False); return
        self.actor.SetVisibility(self.on)
        M[:3, :3] = rot_x_to(vec) * (n * self.ref); M[:3, 3] = origin; self.actor.user_matrix = M

class Flow(Comp):
    """dots running along polylines (current in wires, signal in cables, B-field streamlines)."""
    def __init__(self, app, paths, color, n_res=240, per=3, size=7, closed=False, tubes=None, tube_color=None, tube_op=0.5, name='flow'):
        super().__init__(app); self.paths = []
        for p in paths:
            p = np.asarray(p, float)
            if closed: p = np.vstack([p, p[:1]])
            d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]; s = np.linspace(0, d[-1], n_res)
            self.paths.append(np.c_[[np.interp(s, d, p[:, k]) for k in range(3)]].T)
        self.n = n_res; self.per = per; self.color = hex2rgb(color) if isinstance(color, str) else color
        P = len(self.paths) * per
        self.dots = pv.PolyData(np.zeros((P, 3))); self.dots.point_data['rgb'] = np.tile(self.color, (P, 1))
        self.actors = [app.p.add_mesh(self.dots, scalars='rgb', rgb=True, point_size=size, render_points_as_spheres=True, name=name + '_d', pickable=False)]
        if tubes:
            m = pv.merge([tube(p, tubes) for p in self.paths])
            self.actors.append(app.p.add_mesh(m, color=tube_color or color, opacity=tube_op, name=name + '_t', pickable=False))
        self.show(False)
    def set(self, phase, sign=1.0, signs=None, rgb=None, offset=None):
        pts = []
        for i, p in enumerate(self.paths):
            sg = sign if signs is None else signs[i]
            for k in range(self.per):
                idx = int(((phase * sg + k / self.per) % 1.0) * (self.n - 1)); pts.append(p[idx])
        P = np.array(pts)
        if offset is not None: P = P + offset
        self.dots.points = P
        if rgb is not None: self.dots.point_data['rgb'] = np.asarray(rgb, np.uint8)

class Probe(Comp):
    """a row of coloured beads along an axis through the head (shows B, frequency or phase versus position)."""
    def __init__(self, app, axis=2, half=0.35, n=41, size=10):
        super().__init__(app); self.axis = axis; self.n = n; self.half = half
        self.u = np.linspace(-half, half, n); self.m = pv.PolyData(np.zeros((n, 3))); self.m.point_data['rgb'] = np.zeros((n, 3), np.uint8)
        self.actors = [app.p.add_mesh(self.m, scalars='rgb', rgb=True, point_size=size, render_points_as_spheres=True, name='probe%d' % axis, pickable=False)]; self.show(False)
    def set(self, vals, lo, hi, cmap='coolwarm', lift=None):
        c = self.app.w.head_pos() if self.axis != 2 else np.array([0, HEAD_Y, 0.0])
        P = np.tile(c, (self.n, 1)); P[:, self.axis] += self.u
        if lift is not None: P += lift
        self.m.points = P
        t = np.clip((np.asarray(vals, float) - lo) / max(hi - lo, 1e-12), 0, 1)
        self.m.point_data['rgb'] = (matplotlib.colormaps[cmap](t)[:, :3] * 255).astype(np.uint8)

class Ring3(Comp):
    """expanding RF wave fronts around the head."""
    def __init__(self, app, k=3):
        super().__init__(app); a = np.linspace(0, 2 * np.pi, 49); self.k = k; self.unit = np.c_[np.cos(a), np.sin(a), np.zeros_like(a)]
        n = len(a); pts = np.zeros((k * n, 3)); cells = []
        for j in range(k): cells += [n] + list(range(j * n, (j + 1) * n))
        self.m = pv.PolyData(pts, lines=np.array(cells)); self.actors = [app.p.add_mesh(self.m, color='#facc15', line_width=4, name='rfrings', pickable=False)]; self.show(False)
    def set(self, phase, rmax=0.3, center=None):
        c = self.app.w.head_pos() if center is None else center; P = []
        for j in range(self.k):
            f = (phase + j / self.k) % 1.0; P.append(c + self.unit * (0.06 + rmax * f))
        self.m.points = np.vstack(P)

class VoxelBlock(Comp):
    """pixel grid extruded into voxels (a thin slab of the head phantom)."""
    def __init__(self, app):
        super().__init__(app); self.actors = []; self.key = None
    def build(self, n, thick, size=0.17, mode='gray'):
        key = (n, round(thick, 4), mode)
        if key == self.key: return
        for a in self.actors: self.app.p.remove_actor(a)
        self.key = key; g = pv.ImageData(dimensions=(n + 1, n + 1, 2), spacing=(size / n, size / n, thick), origin=(-size / 2, -size / 2, -thick / 2))
        lab = PH['pd'][::PH['n'] // n, ::PH['n'] // n][:n, :n]; lab = lab[::-1]
        g.cell_data['v'] = lab.T.ravel(order='F') if False else np.tile(lab.ravel(order='C'), 1)
        self.grid = g
        self.actors = [self.app.p.add_mesh(g, scalars='v', cmap='gray', show_edges=True, edge_color='#475569', line_width=1, show_scalar_bar=False, name='voxblock', pickable=False)]
        self.set_pose(); self.show(self.on)
    def set_pose(self):
        c = self.app.w.head_pos()
        for a in self.actors: a.position = tuple(c)

class Slab(Comp):
    def __init__(self, app):
        super().__init__(app); b = pv.Box(bounds=(-0.5, 0.5, -0.5, 0.5, -0.5, 0.5))
        self.actors = [app.p.add_mesh(b, color='#fde047', opacity=0.35, name='slab', pickable=False, smooth_shading=True)]
        e = b.extract_all_edges(); self.actors.append(app.p.add_mesh(e, color='#fde047', line_width=2, name='slab_e', pickable=False)); self.show(False)
    def set(self, z, thick, w=0.26, h=0.30):
        for a in self.actors:
            a.scale = (w, h, max(thick, 1e-4)); a.position = (0, HEAD_Y, z)

class Wave(Comp):
    """a plane-wave pattern over the slice: the spatial-frequency 'ingredient' of one k-space point."""
    def __init__(self, app, n=60):
        super().__init__(app); self.n = n; self.w, self.h = 0.22, 0.26
        xs = np.linspace(-self.w / 2, self.w / 2, n); ys = np.linspace(-self.h / 2, self.h / 2, n)
        X, Y = np.meshgrid(xs, ys); self.X, self.Y = X, Y
        self.g = pv.StructuredGrid(X, Y, np.zeros_like(X)); self.g['w'] = np.zeros(X.size)
        self.actors = [app.p.add_mesh(self.g, scalars='w', cmap='gray', clim=(-1, 1), show_scalar_bar=False, name='wave', pickable=False, smooth_shading=True)]; self.show(False)
    def set(self, kx, ky, ph=0.0, amp=0.012, z=None):
        z0 = self.app.w.table_z if z is None else z
        v = np.cos(2 * np.pi * (kx * self.X / self.w + ky * self.Y / self.h) + ph)
        P = np.c_[self.X.ravel(), self.Y.ravel() + HEAD_Y, (z0 + 0.0) + amp * v.ravel()]
        self.g.points = P; self.g['w'] = v.ravel()

class Fibers(Comp):
    """white-matter bundle + diffusion tensors (for DTI)."""
    def __init__(self, app):
        super().__init__(app); rng = np.random.default_rng(2); self.paths = []; meshes = []
        for k in range(14):
            t = np.linspace(-1, 1, 30); off = rng.normal(0, 0.012, 3)
            p = np.c_[0.12 * t + off[0] * 0 + 0.0, 0.03 * np.sin(2.2 * t) + off[1] + HEAD_Y, 0.07 * t * t - 0.02 + off[2]]
            p[:, 2] = 0.075 * t + off[2]; p[:, 0] = 0.01 * np.sin(2.2 * t) + off[0]; p[:, 1] = HEAD_Y + 0.03 * np.cos(1.8 * t) - 0.02 + off[1]
            self.paths.append(p); meshes.append(tube(p, 0.0016, 6))
        self.rel = np.array([p[15] for p in self.paths]) - np.array([0, 0, 0])
        self.actors = [app.p.add_mesh(pv.merge(meshes), color='#fcd34d', opacity=0.9, name='fibers', pickable=False, smooth_shading=True)]
        ell = pv.Sphere(radius=1.0, theta_resolution=24, phi_resolution=24)
        self.ell = []
        for i in range(0, 14, 2):
            p = self.paths[i]; tdir = np.gradient(p, axis=0)[15]; tdir /= np.linalg.norm(tdir)
            e = ell.copy(); e.points = e.points * np.array([0.018, 0.006, 0.006])
            R = rot_x_to(tdir); e.points = e.points @ R.T + p[15]
            col = np.abs(tdir); self.ell.append(app.p.add_mesh(e, color=tuple(col / col.max()), opacity=0.95, name='ell%d' % i, pickable=False, smooth_shading=True))
        self.actors += self.ell; self.show(False)
    def paths_pos(self): return self.paths

class Labels:
    def __init__(self, app): self.app = app
    def set(self, items, size=14):
        self.clear()
        if not items: return
        pts = np.array([i[1] for i in items], float); txt = [i[0] for i in items]
        self.app.p.add_point_labels(pts, txt, font_size=size, text_color='white', shape_color='#0b0f14', shape_opacity=0.65, point_size=1, always_visible=True, name='labels', shape='rounded_rect', margin=3, show_points=False, pickable=False)
    def clear(self):
        try: self.app.p.remove_actor('labels')
        except Exception: pass

# ----------------------------------------------------------------------------------------------
# steps registry
# ----------------------------------------------------------------------------------------------
STEPS = []
PHASES = {}

def phase_info(n, title, summary):
    PHASES[n] = dict(title=title, summary=summary)

def step(phase, title, text, dur=12.0, cam='scanner', show=None, uses=(), table=None, labels=None):
    def deco(fn):
        STEPS.append(dict(phase=phase, title=title, text=text, dur=dur, cam=cam, show=show, uses=tuple(uses), table=table, labels=labels, fn=fn))
        return fn
    return deco

# ----------------------------------------------------------------------------------------------
# dark-themed matplotlib screens
# ----------------------------------------------------------------------------------------------
FG, MUTED, PANEL, GRID = '#e2e8f0', '#94a3b8', '#0f1720', '#243244'

def sty(ax, title=None, xl=None, yl=None):
    ax.set_facecolor(PANEL)
    for s in ax.spines.values(): s.set_color('#334155')
    ax.tick_params(colors=MUTED, labelsize=7, length=2)
    if title: ax.set_title(title, color=FG, fontsize=8.5, loc='left', pad=3)
    if xl: ax.set_xlabel(xl, color=MUTED, fontsize=7.5, labelpad=1)
    if yl: ax.set_ylabel(yl, color=MUTED, fontsize=7.5, labelpad=1)
    ax.grid(True, color=GRID, lw=0.4)
    return ax

def mk_fig(w, h):
    f = Figure(figsize=(w, h), dpi=100, facecolor='#0b0f14'); FigureCanvasAgg(f) if False else None
    return f

DECODE = ['1 Demodulate', '2 Digitize', '3 Fill k-space', '4 Inverse 2D FT', '5 Magnitude', '6 Image']
DECODE_NOTE = ['remove the 64/128 MHz carrier, keep the slow complex signal', 'ADC samples the voltage; each sample is one k-space point',
               'each readout fills one line; the phase-encode step picks the row', 'every pixel = a weighted sum of all k-space points (sinusoids)',
               'take |value| of the complex result for each pixel', 'gray level = signal at TE; the researcher reads this']


class Arrows(Comp):
    """a few big labelled arrows (voxel magnetizations)."""
    def __init__(self, app, n, L=0.05, lw=6, size=13, name='arr'):
        super().__init__(app); self.n = n; self.L = L
        lines = np.hstack([np.full((n, 1), 2), np.arange(n)[:, None], (np.arange(n) + n)[:, None]]).ravel()
        self.m = pv.PolyData(np.zeros((2 * n, 3)), lines=lines); self.m.point_data['rgb'] = np.zeros((2 * n, 3), np.uint8)
        self.t = pv.PolyData(np.zeros((n, 3))); self.t.point_data['rgb'] = np.zeros((n, 3), np.uint8)
        a1 = app.p.add_mesh(self.m, scalars='rgb', rgb=True, line_width=lw, name=name + str(n) + '_l', pickable=False)
        a2 = app.p.add_mesh(self.t, scalars='rgb', rgb=True, point_size=size, render_points_as_spheres=True, name=name + str(n) + '_t', pickable=False)
        self.actors = [a1, a2]; self.show(False)
    def set(self, origins, vecs, rgb, L=None):
        L = self.L if L is None else L; o = np.asarray(origins, float); v = np.asarray(vecs, float)
        P = np.vstack([o, o + v * L]); rgb = np.asarray(rgb, np.uint8)
        if rgb.ndim == 1: rgb = np.tile(rgb, (self.n, 1))
        self.m.points = P; self.m.point_data['rgb'] = np.vstack([rgb, rgb]); self.t.points = P[self.n:]; self.t.point_data['rgb'] = rgb
# ----------------------------------------------------------------------------------------------
# the application: three screens + decoder strip + step controller
# ----------------------------------------------------------------------------------------------
class App:
    def __init__(self, plotter, figs=None, headless=False, auto=False):
        self.p = plotter; self.headless = headless; self.auto = auto
        self.w = World(plotter); self.lab = Labels(self)
        self.figB, self.figD, self.figC = figs if figs else (Figure(figsize=(6.4, 3.6), dpi=100), Figure(figsize=(6.4, 0.95), dpi=100), Figure(figsize=(6.4, 3.4), dpi=100))
        if figs is None:
            for f in (self.figB, self.figD, self.figC): FigureCanvasAgg(f)
        for f in (self.figB, self.figD, self.figC): f.set_facecolor('#0b0f14')
        self.comps = {}; self.i = -1; self.t = 0.0; self.clock = 0.0; self.state = 'play'; self.S = {}
        self.dirty = dict(B=True, C=True, D=True); self._cam_active = False; self._last_cam = None
        self._dkey = None; self._tab = None; self._build_decoder(); self.cam_now('whole')

    # -- component access --------------------------------------------------------------------------
    def get(self, name):
        if name not in self.comps:
            mk = dict(spins=lambda: Spins(self), mvec=lambda: MVec(self), mvec2=lambda: MVec(self, '#38bdf8'), mvec3=lambda: MVec(self, '#f472b6'),
                      slab=lambda: Slab(self), rf=lambda: Ring3(self), vox=lambda: VoxelBlock(self), wave=lambda: Wave(self),
                      fib=lambda: Fibers(self), probez=lambda: Probe(self, 2, 0.36), probex=lambda: Probe(self, 0, 0.30, 31), probey=lambda: Probe(self, 1, 0.30, 31))
            if name in mk: self.comps[name] = mk[name]()
            elif name.startswith('arr'): self.comps[name] = Arrows(self, int(name[3:]))
            elif name.startswith('rings'):
                zs = {1: (0.0,), 2: (-0.15, 0.15), 3: (-0.3, 0.0, 0.3), 6: (-0.6, -0.45, -0.15, 0.15, 0.45, 0.6)}[int(name[5:])]
                self.comps[name] = Flow(self, [ring_pts(z, 0.62, 100) for z in zs], '#fde047', per=10, size=9, name=name)
            elif name == 'field':
                self.fl_cache = getattr(self, 'fl_cache', {})
                self.comps[name] = self._field(6)
            elif name == 'chain':
                path = np.vstack([self.w.cable_path, self.w.cable2_path[1:]])
                self.comps[name] = Flow(self, [path], '#fbbf24', n_res=300, per=6, size=9, name='chain')
            elif name == 'curz':      # current in the two Z-gradient (Maxwell) loops
                self.comps[name] = Flow(self, [ring_pts(-0.30, 0.44, 80), ring_pts(0.30, 0.44, 80)], '#fde047', per=8, size=9, closed=False, name='curz')
            elif name == 'curb0':
                self.comps[name] = Flow(self, [ring_pts(z, 0.62, 100) for z in (-0.6, -0.45, -0.15, 0.15, 0.45, 0.6)], '#93c5fd', per=10, size=8, name='curb0')
            elif name == 'cur_rf':
                self.comps[name] = Flow(self, [ring_pts(z, 0.385, 80) for z in (-0.32, 0.32)], '#fef08a', per=8, size=8, name='currf')
        return self.comps[name]

    def _field(self, nrings):
        zs = {1: (0.0,), 2: (-0.15, 0.15), 3: (-0.3, 0.0, 0.3), 6: (-0.6, -0.45, -0.15, 0.15, 0.45, 0.6)}[nrings]
        lines = b0_field_lines(0.62, zs, 5, 6)
        return Flow(self, lines, '#7dd3fc', n_res=260, per=3, size=6, tubes=0.0018, tube_color='#3b82f6', tube_op=0.55, name='fld%d' % nrings)

    def field_for(self, n):
        key = 'field%d' % n
        if key not in self.comps: self.comps[key] = self._field(n)
        return self.comps[key]

    # -- layouts for the three screens ----------------------------------------------------------------
    def axB(self, kind='one'):
        f = self.figB; f.clf(); L = {
            'one': dict(a=[0.09, 0.13, 0.88, 0.76]),
            'two': dict(a=[0.09, 0.58, 0.88, 0.34], b=[0.09, 0.11, 0.88, 0.34]),
            'seq1': dict(s=[0.09, 0.74, 0.88, 0.22], a=[0.09, 0.12, 0.88, 0.52]),
            'seq2': dict(s=[0.09, 0.74, 0.88, 0.22], a=[0.09, 0.12, 0.50, 0.52], b=[0.68, 0.12, 0.29, 0.52]),
            'lr': dict(a=[0.09, 0.13, 0.50, 0.76], b=[0.68, 0.13, 0.29, 0.76])}[kind]
        d = {k: sty(f.add_axes(v)) for k, v in L.items()}; self.dirty['B'] = True; return d

    def axC(self, kind='one'):
        f = self.figC; f.clf(); L = {
            'one': dict(a=[0.09, 0.12, 0.88, 0.78]),
            'lr': dict(a=[0.04, 0.08, 0.46, 0.84], b=[0.56, 0.12, 0.42, 0.78]),
            'three': dict(a=[0.02, 0.08, 0.31, 0.78], b=[0.355, 0.08, 0.31, 0.78], c=[0.69, 0.08, 0.31, 0.78]),
            'text': dict(a=[0.02, 0.02, 0.96, 0.94]),
            'img_txt': dict(a=[0.03, 0.06, 0.47, 0.86], t=[0.53, 0.02, 0.46, 0.94])}[kind]
        d = {k: sty(f.add_axes(v)) for k, v in L.items()}
        for k, a in d.items():
            if kind in ('text',) or k == 't': a.axis('off')
        self.dirty['C'] = True; return d

    def mark(self, *ks):
        for k in ks: self.dirty[k] = True

    # -- text on an axis (list of (string, colour) or string)  --------------------------------------------
    @staticmethod
    def put(ax, lines, x=0.02, y=0.96, size=8.2, dy=0.085, color=FG):
        ax.axis('off'); ts = []
        for k, ln in enumerate(lines):
            s, c = (ln, color) if isinstance(ln, str) else ln
            ts.append(ax.text(x, y - k * dy, s, transform=ax.transAxes, color=c, fontsize=size, va='top', family='DejaVu Sans'))
        return ts

    # -- decoder strip ----------------------------------------------------------------------------------
    def _build_decoder(self):
        f = self.figD; f.clf(); ax = f.add_axes([0.005, 0.0, 0.99, 1.0]); ax.set_xlim(0, 6); ax.set_ylim(0, 1); ax.axis('off'); ax.set_facecolor('#0b0f14')
        self.dbox = []
        for k, name in enumerate(DECODE):
            p = FancyBboxPatch((k + 0.04, 0.38), 0.88, 0.5, boxstyle='round,pad=0.01,rounding_size=0.08', fc='#1e293b', ec='#334155', lw=1)
            ax.add_patch(p); t = ax.text(k + 0.48, 0.63, name, color=MUTED, ha='center', va='center', fontsize=7.4)
            self.dbox.append((p, t))
            if k < 5: ax.annotate('', xy=(k + 1.04, 0.63), xytext=(k + 0.93, 0.63), arrowprops=dict(arrowstyle='-|>', color='#64748b', lw=1))
        self.dnote = ax.text(0.05, 0.12, 'decoder idle', color=MUTED, fontsize=7.8, va='center'); self.dax = ax

    def decode(self, active=None, note=None):
        if active is None: act = []
        else: act = [active] if isinstance(active, (int, np.integer)) else list(active)
        key = (tuple(act), note)
        if key == self._dkey: return
        self._dkey = key; lo = min(act) if act else 99
        for k, (p, t) in enumerate(self.dbox):
            if k in act: p.set_facecolor('#f59e0b'); p.set_edgecolor('#fde68a'); t.set_color('#111827')
            elif k < lo and act: p.set_facecolor('#164e63'); p.set_edgecolor('#22d3ee'); t.set_color('#a5f3fc')
            else: p.set_facecolor('#1e293b'); p.set_edgecolor('#334155'); t.set_color(MUTED)
        self.dnote.set_text(note if note else (DECODE_NOTE[act[0]] if act else 'decoder idle: nothing to decode yet'))
        self.dirty['D'] = True

    # -- camera --------------------------------------------------------------------------------------------
    def cam_now(self, name):
        pos, foc = CAMS[name] if isinstance(name, str) else name
        self.p.camera_position = [pos, foc, (0, 1, 0)]; self._last_cam = np.array(pos, float); self._cam_active = False

    def cam_to(self, name, dur=2.0):
        pos, foc = CAMS[name] if isinstance(name, str) else name
        cp = self.p.camera_position
        self._cf = (np.array(cp[0], float), np.array(cp[1], float)); self._ct = (np.array(pos, float), np.array(foc, float))
        self._ctime = 0.0; self._cdur = dur; self._cam_active = True; self._last_cam = None

    def _cam_step(self, dt):
        if not self._cam_active: return
        cur = np.array(self.p.camera_position[0], float)
        if self._last_cam is not None and np.linalg.norm(cur - self._last_cam) > 2e-3: self._cam_active = False; return   # user took over
        self._ctime += dt; k = float(smooth(self._ctime / self._cdur))
        pos = self._cf[0] + (self._ct[0] - self._cf[0]) * k; foc = self._cf[1] + (self._ct[1] - self._cf[1]) * k
        self.p.camera_position = [tuple(pos), tuple(foc), (0, 1, 0)]; self._last_cam = pos
        if self._ctime >= self._cdur: self._cam_active = False

    # -- table ----------------------------------------------------------------------------------------------
    def table(self, z):
        if abs(z - self.w.table_z) > 0.02 or z in (0.0, Z_HOME): self.w.set_table(float(z))
        for n in ('spins', 'vox'):
            if n in self.comps and n == 'vox': self.comps[n].set_pose()

    # -- step control -----------------------------------------------------------------------------------------
    def go(self, i, snap=False):
        i = int(np.clip(i, 0, len(STEPS) - 1)); self.i = i; self.t = 0.0; self.S = {}; self._tab = None; self.state = 'play' if self.state != 'pause' else 'pause'
        sp = STEPS[i]; self.lab.clear()
        spec = {'floor': 1.0}; show = sp['show'] or {}
        dflt = show.get('*', 0.0)
        for n in self.w.parts:
            if n in show: spec[n] = show[n]
            elif n not in spec: spec[n] = dflt
        for k in ('cable', 'cable2'): spec.setdefault(k, 0.0)
        self.w.show_only(spec, default=0.0)
        if snap: self.w.snap_opacity()
        for n, c in self.comps.items():
            if isinstance(c, Comp): c.show(n in sp['uses'])
        for u in sp['uses']:
            if u.startswith('field'): self.field_for(int(u[5:])).show(True)
            elif u not in self.comps: self.get(u).show(True)
            else: self.comps[u].show(True)
        for n, c in list(self.comps.items()):
            if isinstance(c, Comp): c.show(n in sp['uses'])
        if sp['table'] is not None:
            if snap: self.w.set_table(sp['table'])
            else: self._tab = (self.w.table_z, sp['table'])
        if snap: self.cam_now(sp['cam'])
        else: self.cam_to(sp['cam'])
        if sp['labels']: self.lab.set(sp['labels']() if callable(sp['labels']) else sp['labels'])
        self.decode(None); self.axB('one'); self.axC('one'); self.figB.clf(); self.figC.clf()
        self.dirty = dict(B=True, C=True, D=True)

    def is_last_in_phase(self):
        return self.i >= len(STEPS) - 1 or STEPS[self.i + 1]['phase'] != STEPS[self.i]['phase']

    def next_step(self):
        if self.state == 'phase_end': self.state = 'play'
        if self.i < len(STEPS) - 1: self.go(self.i + 1)

    def prev_step(self):
        self.state = 'play' if self.state == 'phase_end' else self.state; self.go(max(0, self.i - 1))

    def next_phase(self):
        ph = STEPS[self.i]['phase']; j = self.i
        while j < len(STEPS) - 1 and STEPS[j]['phase'] == ph: j += 1
        self.state = 'play' if self.state == 'phase_end' else self.state; self.go(j)

    def prev_phase(self):
        ph = STEPS[self.i]['phase']; j = self.i
        while j > 0 and STEPS[j]['phase'] == ph: j -= 1
        ph2 = STEPS[j]['phase']
        while j > 0 and STEPS[j - 1]['phase'] == ph2: j -= 1
        self.state = 'play' if self.state == 'phase_end' else self.state; self.go(j)

    def toggle(self):
        if self.state == 'phase_end': self.next_step(); return
        self.state = 'pause' if self.state == 'play' else 'play'

    def caption(self):
        sp = STEPS[self.i]; ph = PHASES.get(sp['phase'], {})
        head = 'Phase %d.%d  %s' % (sp['phase'], 1 + sum(1 for s in STEPS[:self.i] if s['phase'] == sp['phase']), sp['title'])
        body = sp['text']
        if self.state == 'phase_end':
            head = 'Phase %d complete: %s' % (sp['phase'], ph.get('title', ''))
            body = ph.get('summary', '') + '     [press Right to continue, Left to review]'
        return head, body

    def status(self):
        sp = STEPS[self.i]; return '%s   step %d/%d   %s' % (self.state.upper(), self.i + 1, len(STEPS), '%.1f / %.1f s' % (min(self.t, sp['dur']), sp['dur']))

    # -- frame update -----------------------------------------------------------------------------------------------
    def update(self, dt):
        sp = STEPS[self.i]
        if self.state == 'play':
            self.t += dt; self.clock += dt
            if self.t >= sp['dur']:
                if not self.is_last_in_phase(): self.go(self.i + 1); sp = STEPS[self.i]
                elif self.auto and self.i < len(STEPS) - 1: self.go(self.i + 1); sp = STEPS[self.i]
                else: self.t = sp['dur']; self.state = 'phase_end'
        elif self.state == 'phase_end': self.clock += dt
        self._cam_step(dt)
        tb = self._tab
        if tb:
            k = float(smooth(min(self.t, 3.0) / 3.0)); self.w.set_table(tb[0] + (tb[1] - tb[0]) * k) if abs(tb[0] - tb[1]) > 1e-6 else None
        self.w.step_opacity()
        sp['fn'](self, min(self.t, sp['dur']), min(self.t / sp['dur'], 1.0), self.S)
        for n, c in self.comps.items():
            if isinstance(c, VoxelBlock) and c.on: c.set_pose()

    # -- rendering to pixels (headless) ---------------------------------------------------------------------------------
    @staticmethod
    def fig_rgb(fig):
        fig.canvas.draw(); return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()

    def compose(self, a_img):
        from PIL import Image, ImageDraw, ImageFont
        import matplotlib
        b, d, c = self.fig_rgb(self.figB), self.fig_rgb(self.figD), self.fig_rgb(self.figC)
        h = a_img.shape[0]; wA = a_img.shape[1]; wR = b.shape[1]
        W_ = wA + wR; H_ = h + 170
        im = Image.new('RGB', (W_, H_), (11, 15, 20)); im.paste(Image.fromarray(a_img), (0, 0))
        y = 0
        for part in (b, d, c):
            im.paste(Image.fromarray(part), (wA, y)); y += part.shape[0]
        dr = ImageDraw.Draw(im); fp = os.path.join(matplotlib.get_data_path(), 'fonts/ttf/DejaVuSans.ttf'); fb = os.path.join(matplotlib.get_data_path(), 'fonts/ttf/DejaVuSans-Bold.ttf')
        f1 = ImageFont.truetype(fb, 19); f2 = ImageFont.truetype(fp, 16)
        head, body = self.caption(); dr.text((16, h + 10), head, fill=(251, 191, 36), font=f1)
        words, line, yy = body.split(), '', h + 42
        for w_ in words:
            if dr.textlength(line + ' ' + w_, font=f2) > W_ - 32: dr.text((16, yy), line, fill=(226, 232, 240), font=f2); yy += 22; line = w_
            else: line = (line + ' ' + w_).strip()
        dr.text((16, yy), line, fill=(226, 232, 240), font=f2)
        dr.text((16, H_ - 24), self.status() + '     A: patient and scanner     B: what the machine sees     C: what the researcher sees', fill=(148, 163, 184), font=ImageFont.truetype(fp, 13))
        dr.text((6, 6), 'A  patient and scanner (3D)', fill=(148, 163, 184), font=ImageFont.truetype(fp, 13))
        dr.text((wA + 6, 4), 'B  machine', fill=(148, 163, 184), font=ImageFont.truetype(fp, 12)); dr.text((wA + 6, b.shape[0] + d.shape[0] + 3), 'C  researcher', fill=(148, 163, 184), font=ImageFont.truetype(fp, 12))
        return np.asarray(im)

    def snapshot(self):
        self.p.render(); a = self.p.screenshot(return_img=True); return self.compose(a)

def _smoke():
    pass
# ----------------------------------------------------------------------------------------------
# shared spin-ensemble helpers
# ----------------------------------------------------------------------------------------------
from matplotlib.patches import Circle, Rectangle
NSP = Spins.N
_r = np.random.default_rng(1)
RP = _r.uniform(0, 2 * np.pi, NSP); RT = np.arccos(_r.uniform(-1, 1, NSP)); W1 = _r.uniform(.6, 1.4, NSP); W2 = _r.uniform(.6, 1.4, NSP)
OFFR = _r.normal(0, 1, NSP); UPR = _r.uniform(0, 1, NSP)
OM = 2 * np.pi * 0.8                                   # precession rate shown on screen (real: 64 or 128 MHz, slowed down)
C_OR, C_TEAL, C_COR, C_GREY, C_BLUE = [hex2rgb(c) for c in ('#fb923c', '#2dd4bf', '#f87171', '#94a3b8', '#60a5fa')]

def v_random(t):
    th = RT + 0.7 * W1 * t; ph = RP + 1.1 * W2 * t
    return np.c_[np.cos(th), np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph)]

def v_cone(t, a, om=OM):
    up = UPR < 0.5 + 0.3 * a; th = np.where(up, 0.9553, 2.1863); ph = RP + om * t
    return np.c_[np.cos(th), np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph)], up

def v_iso(mz, mxy, ph):
    ph = np.asarray(ph, float) * np.ones(NSP); return np.c_[mz * np.ones(NSP), mxy * np.cos(ph), mxy * np.sin(ph)]

def phase_rgb(ph): return hsv_rgb(np.asarray(ph) / (2 * np.pi), 0.8, 1.0)
def hp(app, dx=0, dy=0, dz=0): return app.w.head_pos() + np.array([dx, dy, dz])
def pick_field(app, n):
    for k in (1, 2, 3, 6): app.field_for(k).show(k == n)
def pick_rings(app, n):
    for k in (1, 2, 3, 6):
        app.get('rings%d' % k).show(k == n)
def dtc(app, S):
    d = app.clock - S.get('_lc', app.clock); S['_lc'] = app.clock; return d
def cursor(ax, x, S, key='cur', **kw):
    if key in S: S[key].set_xdata([x, x])
    else: S[key] = ax.axvline(x, color='#fbbf24', lw=1, **kw)
def col(name): return COL[name]
def swatch(c, text): return ('■ ' + text, c)
LAB = lambda s, p: (s, p)

def axis_profile(zs, zc, a):
    z0 = np.array(zc) / a; I = np.ones(len(zc)); return np.array([loops_field(z / a, 0.0, z0, I)[0] for z in zs])

def maxwell_profile(zs, opposite=True, G_mT_m=10.0, a=0.44, z0=0.30):
    I = np.array([1.0, -1.0 if opposite else 1.0]); zc = np.array([-z0 / a, z0 / a])
    f = lambda z: loops_field(z / a, 0.0, zc, I)[0]
    v = np.array([f(z) for z in zs]); h = 1e-3; slope = (f(h) - f(-h)) / (2 * h) / a
    k = G_mT_m / abs(slope) if abs(slope) > 1e-9 else 1.0
    return v * k, np.sign(slope) if slope != 0 else 1.0

# ----------------------------------------------------------------------------------------------
# PHASE 0  scanner anatomy
# ----------------------------------------------------------------------------------------------
phase_info(0, 'Scanner anatomy', 'The scanner is a stack of layers around the patient: RF coil, gradient coils, shim, superconducting B0 magnet in a cryostat. The coil closest to the head is also the antenna that collects the signal.')

def draw_layers(ax):
    ax.set_aspect('equal'); ax.set_xlim(-1.25, 2.55); ax.set_ylim(-1.15, 1.15); ax.axis('off')
    L = [('Cover', 1.02, COL['shell']), ('Helium vessel (cryostat)', 0.90, COL['cryo']), ('Shield coil', 0.82, COL['b0s']), ('Main B0 coil', 0.62, COL['b0']),
         ('Shim plates', 0.52, COL['shim']), ('Gradient coils X, Y, Z', 0.45, COL['gx']), ('RF body coil (transmit)', 0.385, COL['rf']), ('Patient bore', 0.35, '#dfe7ee')]
    for k, (name, r, c) in enumerate(L):
        ax.add_patch(Circle((0, 0), r, fill=False, ec=c, lw=2.2))
        y = 0.95 - k * 0.27; x = np.sqrt(max(r * r - min(y * y, (0.97 * r) ** 2), 0.0))
        ax.plot([x, 1.15], [np.sign(y) * min(abs(y), 0.97 * r) if abs(y) > 0.97 * r else y, y], color=c, lw=0.8); ax.text(1.18, y, name, color=c, fontsize=7.6, va='center')
    ax.add_patch(Circle((0, -0.06), 0.16, fc=COL['skin'], ec='none', alpha=0.8)); ax.text(0, -0.30, 'head', color=COL['skin'], fontsize=7, ha='center')
    ax.set_title('Cross-section, looking down the bore', color=FG, fontsize=8.5, loc='left')

@step(0, 'The whole machine', 'The scanner is a set of cylinders nested around the patient. From the inside out: RF coil, gradient coils, shim plates, the superconducting B0 magnet in its cryostat, and a cover. Electronics sit beside it. Scroll to zoom, drag to rotate.',
      dur=18, cam='whole', show={'*': None, 'cable': 0, 'cable2': 0, 'room': 0.18}, table=Z_HOME,
      labels=lambda: [('Superconducting B0 magnet', (0, 0.62, 0.6)), ('Gradient + RF coils', (0, 0.4, -0.6)), ('Cold head', (0, 1.5, -0.3)), ('Patient table', (0, -0.15, -3.4)),
                      ('Receiver + ADC', (2.3, 0.3, 0.6)), ('Computer', (3.0, 0.1, -0.7))])
def s0_1(app, t, u, S):
    if not S:
        d = app.axB('one'); draw_layers(d['a']); c = app.axC('text')
        app.put(c['a'], [('What each layer does', FG), swatch(COL['b0'], 'B0 magnet: strong, steady field along z, 1.5 or 3 T'), swatch(COL['cryo'], 'Cryostat: keeps the wire near 4 K so it has no resistance'),
                         swatch(COL['shim'], 'Shim: small corrections so B0 is uniform over the head'), swatch(COL['gx'], 'Gradients X, Y, Z: add a position-dependent field'),
                         swatch(COL['rf'], 'RF body coil: transmits the pulse that tips M'), swatch(COL['rx'], 'Receive coil: antenna that picks up the signal'),
                         swatch(COL['adc'], 'Receiver + ADC: amplify, demodulate, digitize'), swatch(COL['pc'], 'Computer: reconstructs the image')], size=8.4, dy=0.105)
        S['i'] = 1
    app.mark('B', 'C')

def b0_profile_plot(app, S):
    d = app.axB('one'); a = d['a']; zs = np.linspace(-1.3, 1.3, 260); B = axis_b0_profile(zs, 0.62) if False else None
    zc = (-0.6, -0.45, -0.15, 0.15, 0.45, 0.6); prof = axis_profile(zs, zc, 0.62); prof = 3.0 * prof / prof[len(prof) // 2]
    a.plot(zs, prof, color=COL['b0'], lw=2); a.axvspan(-0.25, 0.25, color='#22d3ee', alpha=0.15); a.text(0, 0.35, 'imaging sphere', color='#67e8f9', fontsize=7.5, ha='center')
    a.set_ylim(0, 3.6); sty(a, 'B0 along the bore axis (computed from the coil layout)', 'z (m)', 'B (T)'); return d

@step(0, 'B0 magnet and cryostat', 'The main magnet is a set of superconducting loops in a vessel of liquid helium near 4 K. Current in these loops never stops, so the field is always on. A cold head keeps the helium cold. Opposite-current shield loops pull the field back in.',
      dur=16, cam='scanner', show={'b0': None, 'b0shield': None, 'cryo': 0.25, 'cover': 0.10, 'bore': 0.12, 'coldhead': None, 'plate_front': 0.15, 'plate_back': 0.15, 'patient': 0.25, 'table': 0.4, 'pedestal': 0.5}, uses=('curb0',), table=0.0,
      labels=lambda: [('Main B0 loops', (0, 0.66, 0.15)), ('Shield loops', (0, 0.84, 0.5)), ('Helium vessel, ~4 K', (0, -0.9, -0.2)), ('Cold head', (0, 1.5, -0.3))])
def s0_2(app, t, u, S):
    if not S:
        b0_profile_plot(app, S); c = app.axC('text')
        app.put(c['a'], [('Facts', FG), '• Field strength: 1.5 T or 3 T (Earth: about 0.00005 T)', '• Larmor frequency: f = 42.58 MHz/T x B0', '   1.5 T -> %.1f MHz     3 T -> %.1f MHz' % (larmor_mhz(1.5), larmor_mhz(3)),
                         '• The wire is superconducting, so the current circulates for years', '• Helium vessel sits inside a vacuum jacket to limit heat leak',
                         '• Field must be uniform to a few parts per million in the head', '• Link to your sheet: each loop gives B = mu0 I / 2r at its center;', '   many loops add into one straight field (see phase 1)'], size=8.4, dy=0.105)
        S['i'] = 1
    app.get('curb0').set(app.clock * 0.18)
    app.mark('B', 'C')

@step(0, 'Gradient coils and shim', 'Three gradient coils each add a small field that changes linearly along one axis. Z uses a Maxwell pair, X and Y use saddle (Golay) coils. Shim plates sit outside them to flatten B0. These coils switch on and off, which causes the knocking sound.',
      dur=18, cam='coils', show={'gx': None, 'gy': None, 'gz': None, 'gshell': 0.12, 'shim': None, 'bore': 0.12, 'patient': 0.25, 'table': 0.4, 'b0': 0.18, 'bodycoil': 0.35}, uses=('curz',), table=0.0,
      labels=lambda: [('Z gradient (Maxwell pair)', (0, 0.44, 0.3)), ('X gradient (saddles)', (0.45, 0.0, -0.3)), ('Y gradient (saddles)', (0, 0.43, -0.28)), ('Shim plates', (0.52, 0.0, 0.0))])
def s0_3(app, t, u, S):
    if not S:
        d = app.axB('two'); zs = np.linspace(-0.45, 0.45, 120); S['zs'] = zs
        prof, sg = maxwell_profile(zs); S['prof'] = prof
        S['l1'], = d['a'].plot(zs, prof * 0, color=COL['gz'], lw=2); d['a'].set_ylim(-5, 5); sty(d['a'], 'Z gradient field added to B0, along z (opposite currents)', 'z (m)', 'dB (mT)')
        tt = np.linspace(0, 1, 200); S['tt'] = tt; g = np.clip(np.minimum(tt / 0.2, 1) * np.minimum((1 - tt) / 0.2, 1), 0, 1); S['g'] = g
        d['b'].plot(tt, g * 10, color=COL['gx'], lw=1.5); S['cur'] = d['b'].axvline(0, color='#fbbf24'); sty(d['b'], 'Gradient waveform: ramp up, hold, ramp down', 'time', 'G (mT/m)')
        c = app.axC('text')
        app.put(c['a'], [('Gradient facts', FG), '• Strength: tens of mT/m, so the field changes by only a few mT', '   across the body, compared with 1500 or 3000 mT of B0', '• Slew rate: how fast G can change (limited by coil inductance)',
                         '• Isocenter: the point where the gradient adds zero, so B = B0', '• Coils on your sheet: Maxwell pair = two loops with opposite current', '• F = I L B sin(theta) on the wires makes the knocking sound'], size=8.4, dy=0.11)
    g = S['g'][int(np.clip(u, 0, 0.999) * 199)]; S['l1'].set_ydata(S['prof'] * g); S['cur'].set_xdata([u, u])
    app.get('curz').set(app.clock * 0.2, signs=[1, -1]); app.mark('B', 'C')

@step(0, 'RF transmit and receive coils', 'The body coil in the bore wall transmits the RF pulse. A smaller receive coil sits close to the head and acts as the antenna. Both are tuned to the Larmor frequency. Closer coils see a stronger signal, so a head coil beats the body coil for brain scans.',
      dur=16, cam='head', show={'bodycoil': None, 'headcoil': None, 'preamp': None, 'patient': 0.45, 'table': 0.6, 'bore': 0.1}, uses=('cur_rf',), table=0.0,
      labels=lambda: [('Head receive coil', (0.0, 0.14, 0.0)), ('Preamp', (0.0, -0.07, -0.17)), ('Body coil (transmit)', (0.0, 0.40, 0.3))])
def s0_4(app, t, u, S):
    if not S:
        d = app.axB('one'); tt = np.linspace(-3, 3, 900); env = np.sinc(tt); car = np.cos(2 * np.pi * 9 * tt)
        d['a'].plot(tt, env * car, color=COL['rf'], lw=0.7); d['a'].plot(tt, env, color='#fde68a', lw=1.5); S['cur'] = d['a'].axvline(-3, color='#fbbf24')
        sty(d['a'], 'Transmit: RF pulse = slow envelope x carrier at the Larmor frequency (carrier slowed for display)', 'time', 'B1')
        c = app.axC('text')
        app.put(c['a'], [('Two jobs, two coils', FG), swatch(COL['rf'], 'Transmit (body coil): sends a B1 field turning at f = %.1f MHz at 1.5 T' % larmor_mhz(1.5)),
                         '   or %.1f MHz at 3 T. This field is perpendicular to B0.' % larmor_mhz(3), swatch(COL['rx'], 'Receive (head coil): a loop near the head. Precessing M'),
                         '   induces a tiny voltage in it (Faraday). It sits as close as possible.', swatch(COL['preamp'], 'Preamp: mounted on the coil so the signal is boosted at once'),
                         '', 'The transmit coil is placed so its B1 is perpendicular to B0.'], size=8.4, dy=0.11)
    S['cur'].set_xdata([-3 + 6 * u] * 2); app.get('cur_rf').set(app.clock * 0.9); app.mark('B', 'C')

@step(0, 'Where the data is collected', 'The receive coil picks up a voltage of microvolt scale. The preamp boosts it, a cable carries it to the receiver, the ADC turns it into numbers, and the computer builds the image. This chain is the only place the machine reads out the patient.',
      dur=18, cam='whole', show={'headcoil': None, 'preamp': None, 'cable': None, 'cable2': None, 'adc': None, 'pc': None, 'monitor': None, 'patient': 0.35, 'table': 0.5, 'bore': 0.08, 'cover': 0.1, 'pedestal': 0.4, 'room': 0.15}, uses=('chain',), table=0.0,
      labels=lambda: [('Head coil', (0, 0.15, 0.0)), ('Preamp', (0, -0.07, -0.17)), ('Receiver + ADC', (2.3, 0.3, 0.6)), ('Computer', (3.0, 0.1, -0.7))])
def s0_5(app, t, u, S):
    if not S:
        d = app.axB('one'); a = d['a']; a.axis('off'); a.set_xlim(0, 10); a.set_ylim(0, 4)
        boxes = [('Head coil', 'EMF ~ microvolts', COL['rx']), ('Preamp', 'low noise, x gain', COL['preamp']), ('Receiver', 'demodulate, filter', COL['adc']), ('ADC', 'samples -> numbers', COL['adc']), ('Computer', 'k-space -> image', COL['pc'])]
        S['bx'] = []
        for k, (n_, s_, c_) in enumerate(boxes):
            x = 0.2 + k * 1.95; p = FancyBboxPatch((x, 1.2), 1.7, 1.5, boxstyle='round,pad=0.02,rounding_size=0.15', fc='#1e293b', ec=c_, lw=1.6); a.add_patch(p)
            a.text(x + 0.85, 2.2, n_, color=c_, ha='center', fontsize=8, va='center'); a.text(x + 0.85, 1.65, s_, color=MUTED, ha='center', fontsize=6.6, va='center'); S['bx'].append(p)
            if k < 4: a.annotate('', xy=(x + 1.93, 1.95), xytext=(x + 1.72, 1.95), arrowprops=dict(arrowstyle='-|>', color='#94a3b8'))
        a.text(0.2, 3.5, 'Receive chain: from patient to numbers', color=FG, fontsize=9); S['dot'], = a.plot([0.2], [0.7], 'o', color='#fbbf24')
        c = app.axC('text')
        app.put(c['a'], [('Why this chain is built this way', FG), '• The coil voltage is tiny, so the first amplifier must add very little noise', '• During transmit, coil ports can see more than 100 V, so receivers are protected', '• The receiver brings the signal to about 1 V for the ADC',
                         '• Each coil element has its own preamp and receiver channel', '• The signal is strongest when all spins are in phase (center of k-space)'], size=8.4, dy=0.11)
    app.get('chain').set(app.clock * 0.25); S['dot'].set_xdata([0.2 + 9.6 * ((app.clock * 0.25) % 1.0)])
    for k, p in enumerate(S['bx']): p.set_facecolor('#334155' if abs(((app.clock * 0.25) % 1.0) * 5 - k - 0.5) < 0.5 else '#1e293b')
    app.decode(None, 'the decoder starts at the receiver: the next steps show each stage'); app.mark('B', 'C', 'D')

# ----------------------------------------------------------------------------------------------
# PHASE 1  B0 and net magnetization
# ----------------------------------------------------------------------------------------------
phase_info(1, 'B0 and net magnetization', 'Rings of current add into one straight field. In that field slightly more hydrogen spins point with B0 than against it, which builds a net magnetization M. M precesses at the Larmor frequency f = 42.58 MHz/T x B0.')

@step(1, 'Rings of current add into a straight field', 'Each ring makes its own circular field. Inside the coil, the rings point the same way and add. Outside, neighbouring rings oppose and mostly cancel. More rings give a straighter, stronger field through the middle.',
      dur=20, cam='scanner', show={'b0': 0.45, 'b0shield': 0.0, 'cryo': 0.0, 'cover': 0.06, 'bore': 0.08, 'patient': 0.12, 'table': 0.3, 'pedestal': 0.4}, uses=('rings1', 'rings2', 'rings3', 'rings6', 'field1', 'field2', 'field3', 'field6'), table=0.0)
def s1_1(app, t, u, S):
    n = 1 if u < 0.2 else 2 if u < 0.42 else 3 if u < 0.65 else 6
    if S.get('n') != n:
        S['n'] = n; d = app.axB('one'); a = d['a']; zs = np.linspace(-1.2, 1.2, 200); zc = {1: (0.0,), 2: (-0.15, 0.15), 3: (-0.3, 0.0, 0.3), 6: (-0.6, -0.45, -0.15, 0.15, 0.45, 0.6)}[n]
        p = axis_profile(zs, zc, 0.62); S['ref'] = S.get('ref') or axis_profile(np.array([0.0]), (0.0,), 0.62)[0]
        a.plot(zs, p, color=COL['b0'], lw=2); a.set_ylim(0, 22); a.set_xlim(-1.2, 1.2); sty(a, 'Field on the axis with %d ring%s' % (n, '' if n == 1 else 's'), 'z (m)', 'B (arbitrary units)')
        S['ctr'] = p[len(p) // 2]; c = app.axC('text')
        app.put(c['a'], [('%d ring%s' % (n, '' if n == 1 else 's'), '#fbbf24'), 'Field at the center: %.1f x the single-ring value' % (S['ctr'] / S['ref']),
                         '', 'One ring (your sheet): B = mu0 I / 2r at its center', 'and it falls off quickly along the axis.', '', 'Adding rings makes the curve flatter in the middle:', 'a uniform field, which MRI needs.'], size=8.6, dy=0.115)
    pick_field(app, n); pick_rings(app, n); app.field_for(n).set(app.clock * 0.12); app.get('rings%d' % n).set(app.clock * 0.2); app.mark('B', 'C')

@step(1, 'The patient enters and M builds', 'The patient slides into B0. Hydrogen spins start pointing in random directions, so their sum is zero. Once B0 acts, each spin precesses on a cone, and slightly more point along B0 than against it. That tiny excess adds up to a net magnetization M.',
      dur=26, cam='patient', show={'patient': 0.28, 'table': 0.6, 'bore': 0.1, 'cover': 0.06, 'b0': 0.4, 'pedestal': 0.5, 'bodycoil': 0.2, 'plate_back': 0.1, 'plate_front': 0.1}, uses=('spins', 'mvec', 'curb0'), table=0.0,
      labels=lambda: [('B0 points along +z', (0, 0.5, 0.45))])
def s1_2(app, t, u, S):
    if not S:
        d = app.axB('two'); S['d'] = d; S['bars'] = d['a'].bar(['parallel', 'antiparallel'], [0.5, 0.5], color=[COL['rx'], '#f87171']); d['a'].set_ylim(0, 1); sty(d['a'], 'Spin populations (excess exaggerated so you can see it)', None, 'fraction')
        S['ml'], = d['b'].plot([], [], color='#fbbf24', lw=2); d['b'].set_xlim(0, 1); d['b'].set_ylim(0, 1.05); sty(d['b'], 'M along B0 builds with time', 'time in the field', 'M / M0')
        c = app.axC('text'); S['txt'] = c['a']; S['xs'] = []; S['ys'] = []
        app.put(c['a'], [('Real numbers', FG), 'Excess = hf / 2kT (body temperature, 310 K)', '   1.5 T:  %.1f spins per million' % (boltzmann_excess(1.5) * 1e6), '   3 T:    %.1f spins per million' % (boltzmann_excess(3) * 1e6),
                         '', 'A voxel of 1 mm^3 holds about 3 x 10^19 water molecules,', 'so even a few ppm is about 10^14 net spins.', 'We are building a magnet inside the body.'], size=8.6, dy=0.115)
    a = float(smooth((u - 0.38) / 0.42)); vr = v_random(app.clock); vc, up = v_cone(app.clock, a)
    v = vr * (1 - a) + vc * a; v /= np.linalg.norm(v, axis=1, keepdims=True)
    rgb = np.where((a > 0.25) & up[:, None], C_TEAL, np.where(a > 0.25, C_COR, C_GREY)).astype(np.uint8)
    app.get('spins').set(v, rgb); M = v.mean(0)
    app.get('mvec').set(hp(app, 0, 0.12, 0), M * 3.0)
    for b, h in zip(S['bars'], (0.5 + 0.3 * a, 0.5 - 0.3 * a)): b.set_height(h)
    S['xs'].append(u); S['ys'].append(min(1, a)); S['ml'].set_data(S['xs'], S['ys']); app.get('curb0').set(app.clock * 0.15); app.mark('B', 'C')

@step(1, 'Precession at the Larmor frequency', 'In B0 each spin precesses about the field like a spinning top. The rate is set only by the field: f = 42.58 MHz/T x B0. Raise B0 and every spin turns faster. A 3 T scanner precesses at about 128 MHz, a 1.5 T scanner at about 64 MHz.',
      dur=18, cam='head', show={'patient': 0.2, 'table': 0.3, 'headcoil': 0.3}, uses=('spins', 'mvec'), table=0.0)
def s1_3(app, t, u, S):
    if not S:
        d = app.axB('one'); Bs = np.linspace(0, 7, 50); d['a'].plot(Bs, GAMMA_MHZ_T * Bs, color=COL['b0'], lw=2); S['pt'], = d['a'].plot([1.5], [64], 'o', color='#fbbf24', ms=8)
        for b_ in (1.5, 3.0, 7.0): d['a'].plot([b_], [GAMMA_MHZ_T * b_], 'o', color='#93c5fd', ms=4); d['a'].text(b_, GAMMA_MHZ_T * b_ + 8, '%.1f T' % b_, color=MUTED, fontsize=7, ha='center')
        sty(d['a'], 'Larmor frequency versus field strength', 'B0 (T)', 'f (MHz)'); S['txt'] = app.axC('text')['a']; S['tt'] = None; S['ph'] = 0.0
    B = 1.5 + 1.5 * float(smooth((u - 0.2) / 0.5)); S['ph'] += dtc(app, S) * OM * (B / 1.5); S['pt'].set_data([B], [larmor_mhz(B)])
    vc, up = v_cone(0, 1.0); th = np.where(up, 0.9553, 2.1863); ph = RP + S['ph']
    v = np.c_[np.cos(th), np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph)]; rgb = np.where(up[:, None], C_TEAL, C_COR).astype(np.uint8)
    app.get('spins').set(v, rgb); app.get('mvec').set(hp(app, 0, 0.12, 0), v.mean(0) * 3.0)
    S['txt'].clear(); app.put(S['txt'], [('f = gamma x B0', FG), 'B0 = %.2f T     f = %.2f MHz' % (B, larmor_mhz(B)), '', 'Spin rate on screen is slowed by about 10^8.', '', 'Torque on a magnetic moment (like tau = pE sin(theta)', 'on your sheet) is perpendicular to the spin, so it', 'turns the spin instead of aligning it: precession.'], size=8.8, dy=0.12)
    app.mark('B', 'C')

# ----------------------------------------------------------------------------------------------
# PHASE 2  RF excitation and relaxation
# ----------------------------------------------------------------------------------------------
phase_info(2, 'RF pulse, T2 decay, T1 recovery', 'An RF pulse at the Larmor frequency tips M into the transverse plane. There it precesses, induces the signal, and fades by T2 decay (spins dephase). Separately and much more slowly, M regrows along B0 by T1 recovery. The machine only sees the fast T2 process.')
T1S = dict(WM=830., GM=1330., CSF=4000.); T2S = dict(WM=80., GM=110., CSF=2000.)

@step(2, 'The RF pulse tips M', 'The body coil sends a B1 field turning at the Larmor frequency. In the frame of the spins, B1 looks static, so M rotates about it and leaves the z axis. A 90 degree pulse lays M flat in the xy plane. The pulse stops when the flip angle is reached.',
      dur=18, cam='head', show={'bodycoil': 0.8, 'patient': 0.2, 'table': 0.3, 'headcoil': 0.3, 'bore': 0.05}, uses=('spins', 'mvec', 'rf', 'cur_rf'), table=0.0)
def s2_1(app, t, u, S):
    if not S:
        d = app.axB('two'); tt = np.linspace(0, 1, 400); env = np.where((tt > 0.2) & (tt < 0.5), np.sinc((tt - 0.35) / 0.15 * 2), 0)
        d['a'].plot(tt, env, color=COL['rf'], lw=1.6); S['c1'] = d['a'].axvline(0, color='#fbbf24'); sty(d['a'], 'B1 envelope (RF pulse)', None, 'B1'); d['a'].set_xticks([])
        S['f'], = d['b'].plot([], [], color='#fbbf24', lw=2); d['b'].set_xlim(0, 1); d['b'].set_ylim(0, 100); sty(d['b'], 'Flip angle', 'time', 'degrees')
        c = app.axC('text'); app.put(c['a'], [('Flip angle', FG), 'alpha = 2 pi x 42.58 MHz/T x B1 x duration', 'A 90 degree pulse lasting 1 ms needs B1 of about %.1f microtesla' % (0.25 / (GAMMA_MHZ_T * 1e6 * 1e-3) * 1e6), '', 'Stronger B1 or longer pulse -> larger angle.', '180 degrees flips M to -z (used for spin echo, phase 9).'], size=8.6, dy=0.13)
        S['x'] = []; S['y'] = []
    pulse = 0.2 < u < 0.5; k = float(np.clip((u - 0.2) / 0.3, 0, 1)); al = (np.pi / 2) * k
    ph = OM * app.clock + 0.25 * np.sin(RP)
    app.get('spins').set(v_iso(np.cos(al), np.sin(al), ph), C_OR); app.get('mvec').set(hp(app, 0, 0.12, 0), np.array([np.cos(al), np.sin(al) * np.cos(OM * app.clock), np.sin(al) * np.sin(OM * app.clock)]) * 1.2)
    app.get('rf').show(pulse); app.get('rf').set(app.clock * 0.7); app.get('cur_rf').show(pulse); app.get('cur_rf').set(app.clock * 0.9)
    S['c1'].set_xdata([u, u]); S['x'].append(u); S['y'].append(np.degrees(al)); S['f'].set_data(S['x'], S['y']); app.mark('B', 'C')

@step(2, 'The signal: free induction decay', 'After the pulse, M precesses in the xy plane. The head coil is parallel to z, so it sees only the transverse part of M. Its precession induces an alternating voltage, the free induction decay. M along z induces nothing.',
      dur=18, cam='head', show={'headcoil': 1.0, 'patient': 0.2, 'table': 0.3, 'preamp': None, 'bore': 0.05}, uses=('spins', 'mvec'), table=0.0)
def s2_2(app, t, u, S):
    if not S:
        d = app.axB('one'); S['ln'], = d['a'].plot([], [], color=COL['rx'], lw=1.2); d['a'].set_xlim(0, 1); d['a'].set_ylim(-1.1, 1.1); sty(d['a'], 'Voltage in the receive coil (carrier slowed for display)', 'time after the pulse', 'EMF')
        c = app.axC('text'); app.put(c['a'], [('What the coil sees', FG), swatch(COL['rx'], 'transverse M (in the xy plane): induces a voltage'), swatch('#94a3b8', 'longitudinal M (along z): induces nothing'), '',
                                             'The envelope shrinks because spins fall out of step.', 'The machine records this curve; it never sees T1 recovery.'], size=8.8, dy=0.13); S['x'] = []; S['y'] = []
    tt = t / 18.0 * 1.0; dec = np.exp(-t / 5.0); ph = OM * app.clock + OFFR * 0.12 * t
    app.get('spins').set(v_iso(0.0, dec, ph), C_OR); M = v_iso(0.0, dec, ph).mean(0); app.get('mvec').set(hp(app, 0, 0.12, 0), M * 1.2)
    S['x'].append(tt); S['y'].append(M[1]); S['ln'].set_data(S['x'], S['y']); app.mark('B', 'C')

@step(2, 'T2 decay: the spins fall out of step', 'Each spin sees a slightly different field (spin-spin interactions and field imperfections), so each precesses at a slightly different rate. Starting in step, they spread around the circle. The vector sum shrinks, and the signal decays with time constant T2.',
      dur=18, cam='head', show={'patient': 0.2, 'table': 0.3, 'headcoil': 0.3}, uses=('spins', 'mvec'), table=0.0)
def s2_3(app, t, u, S):
    if not S:
        d = app.axB('two'); S['wheel'] = d['a']; d['a'].set_xlim(-1.2, 1.2); d['a'].set_ylim(-1.2, 1.2); d['a'].set_aspect('equal'); d['a'].add_patch(Circle((0, 0), 1, fill=False, ec='#475569'))
        S['sc'] = d['a'].scatter(np.cos(RP), np.sin(RP), s=8, c='#fb923c'); S['ar'], = d['a'].plot([0, 0], [0, 0], color='#fbbf24', lw=3); sty(d['a'], 'Looking down the bore: phase of each spin', None, None); d['a'].set_xticks([]); d['a'].set_yticks([])
        S['l'], = d['b'].plot([], [], color='#fbbf24', lw=2); tt = np.linspace(0, 1, 200); d['b'].plot(tt, np.exp(-tt * 14 * 15 / 80), '--', color='#93c5fd', lw=1); d['b'].set_xlim(0, 1); d['b'].set_ylim(0, 1.05); sty(d['b'], 'Net transverse M (dashed: pure T2 = 80 ms)', 'time (ms)', '|Mxy|')
        d['b'].set_xticks([0, 0.5, 1]); d['b'].set_xticklabels(['0', '105', '210']); c = app.axC('text')
        app.put(c['a'], [('T2 versus T2*', FG), 'T2: loss from spin-spin interaction inside the tissue', '     (intrinsic: tissue type sets it)', 'T2*: T2 plus loss from an uneven magnet and from', '     paramagnetic material nearby (extrinsic)', '', '1/T2* = 1/T2 + 1/T2(inhomogeneity)', 'Shimming reduces the extrinsic part.', 'Brain at 3 T: WM 80 ms, GM 110 ms, CSF about 2000 ms'], size=8.4, dy=0.105); S['x'] = []; S['y'] = []
    tau = t * 15.0; ph = RP * 0.0 + OM * app.clock + OFFR * 0.35 * t; mxy = np.exp(-tau / 80.0)
    v = v_iso(0.0, mxy, ph); app.get('spins').set(v, phase_rgb((OFFR * 0.35 * t) % (2 * np.pi))); M = v.mean(0); app.get('mvec').set(hp(app, 0, 0.12, 0), M * 1.4)
    xy = np.c_[mxy * np.cos(ph - OM * app.clock), mxy * np.sin(ph - OM * app.clock)]; S['sc'].set_offsets(xy); mm = xy.mean(0); S['ar'].set_data([0, mm[0]], [0, mm[1]])
    S['x'].append(u); S['y'].append(np.hypot(*mm)); S['l'].set_data(S['x'], S['y']); app.mark('B', 'C')

@step(2, 'T1 recovery: M regrows along B0', 'While the signal fades, M along z is rebuilding. Spins give energy to their surroundings and return toward alignment with B0. This takes seconds, far longer than T2 decay. Tissues differ: white matter regrows faster than gray matter, and CSF is slowest.',
      dur=18, cam='head', show={'patient': 0.2, 'table': 0.3}, uses=('spins', 'mvec'), table=0.0)
def s2_4(app, t, u, S):
    if not S:
        d = app.axB('one'); tt = np.linspace(0, 5600, 300)
        for k, c_ in (('WM', '#ffd27a'), ('GM', '#b0a0ff'), ('CSF', '#6cc4ff')): d['a'].plot(tt, 1 - np.exp(-tt / T1S[k]), color=c_, lw=1.8, label='%s  T1 = %d ms' % (k, T1S[k]))
        d['a'].legend(facecolor=PANEL, edgecolor='#334155', labelcolor=FG, fontsize=7.5, loc='lower right'); S['cur'] = d['a'].axvline(0, color='#fbbf24'); sty(d['a'], 'Longitudinal M after a 90 degree pulse (white matter drawn in the 3D view)', 'time (ms)', 'Mz / M0')
        c = app.axC('text'); app.put(c['a'], [('T1 recovery', FG), 'Mz(t) = M0 (1 - exp(-t / T1))', 'After about 1 T1 : 63% regrown, after 3 T1 : 95%', '', 'T1 (3 T): WM 830 ms, GM 1330 ms, CSF 4000 ms, fat 380 ms', 'Higher field -> longer T1.'], size=8.8, dy=0.13)
    tau = u * 5600.0; mz = 1 - np.exp(-tau / T1S['WM']); S['cur'].set_xdata([tau, tau])
    app.get('spins').set(v_iso(mz, 0.0, OM * app.clock), C_OR); app.get('mvec').set(hp(app, 0, 0.12, 0), np.array([0.0, 0, 0]) + np.array([mz, 0, 0]) * 1.4); app.mark('B', 'C')

@step(2, 'Two separate processes', 'Classic animations show M spiralling back to the z axis while the signal fades, as if one process. They are two. T2 decay is over within a few hundred ms. T1 recovery needs seconds. The time axis here is logarithmic so both fit on one plot.',
      dur=20, cam='head', show={'patient': 0.2, 'table': 0.3}, uses=('spins', 'mvec'), table=0.0)
def s2_5(app, t, u, S):
    if not S:
        d = app.axB('one'); tt = np.logspace(0, 4, 300); d['a'].semilogx(tt, np.exp(-tt / 80.0), color='#fb923c', lw=2, label='T2 decay (transverse), WM'); d['a'].semilogx(tt, 1 - np.exp(-tt / 830.0), color='#60a5fa', lw=2, label='T1 recovery (longitudinal), WM')
        d['a'].legend(facecolor=PANEL, edgecolor='#334155', labelcolor=FG, fontsize=7.5, loc='center left'); S['cur'] = d['a'].axvline(1, color='#fbbf24'); sty(d['a'], 'Both processes on one log-time axis', 'time after the pulse (ms, log scale)', 'fraction')
        c = app.axC('text'); app.put(c['a'], [('Take-home', FG), 'T2 decay is fast and visible: it is the signal the coil records.', 'T1 recovery is slow and hidden: it only sets how much M is', 'available for the next pulse.', '', 'So a "T1-weighted" image does not come from watching T1.', 'It comes from how much M has regrown when the next pulse', 'arrives (phase 8).'], size=8.6, dy=0.115)
    tau = 10 ** (4.0 * u); mz = 1 - np.exp(-tau / 830.0); mxy = np.exp(-tau / 80.0); S['cur'].set_xdata([tau, tau])
    app.get('spins').set(v_iso(mz, mxy, OM * app.clock + OFFR * 0.0), C_OR); app.get('mvec').set(hp(app, 0, 0.12, 0), np.array([mz, mxy * np.cos(OM * app.clock), mxy * np.sin(OM * app.clock)]) * 1.4); app.mark('B', 'C')

# ----------------------------------------------------------------------------------------------
# PHASE 3  detection
# ----------------------------------------------------------------------------------------------
phase_info(3, 'Detecting the signal', 'The precessing transverse M changes the flux through the receive coil. By Faraday induction this makes a voltage, which is amplified, demodulated and digitized. Each sample is one point in k-space.')

@step(3, 'Faraday induction in the receive coil', 'The coil is a loop parallel to z. As M turns in the xy plane, the magnetic flux through the loop changes sinusoidally. Faraday\'s law gives a voltage proportional to how fast the flux changes: EMF = -dPhi/dt. A larger or closer M gives a larger voltage.',
      dur=16, cam='head', show={'headcoil': 1.0, 'patient': 0.25, 'table': 0.3, 'preamp': None}, uses=('mvec',), table=0.0)
def s3_1(app, t, u, S):
    if not S:
        d = app.axB('two'); tt = np.linspace(0, 1, 400); S['tt'] = tt
        S['f'], = d['a'].plot(tt, np.cos(2 * np.pi * 4 * tt), color='#60a5fa', lw=1.6); sty(d['a'], 'Flux through the coil: Phi(t) = Phi0 cos(2 pi f t)', None, 'flux'); d['a'].set_xticks([])
        S['e'], = d['b'].plot(tt, np.sin(2 * np.pi * 4 * tt), color=COL['rx'], lw=1.6); sty(d['b'], 'Voltage: EMF = -dPhi/dt (largest where the flux changes fastest)', 'time', 'EMF')
        S['c1'] = d['a'].axvline(0, color='#fbbf24'); S['c2'] = d['b'].axvline(0, color='#fbbf24'); c = app.axC('text')
        app.put(c['a'], [('Faraday\'s law', FG), 'EMF = -N dPhi/dt', 'Phi = B . A for the loop', '', 'M along z: flux constant -> EMF = 0', 'M in the xy plane: flux oscillates -> EMF at the Larmor frequency', '', 'This is why only transverse M produces signal.'], size=8.8, dy=0.12)
    ph = 2 * np.pi * 4 * (u % 1.0); app.get('mvec').set(hp(app, 0, 0.12, 0), np.array([0.0, np.cos(ph), np.sin(ph)]) * 1.3)
    S['c1'].set_xdata([u, u]); S['c2'].set_xdata([u, u]); app.mark('B', 'C')

@step(3, 'Receiver chain: from voltage to numbers', 'The coil voltage rides on a carrier at 64 or 128 MHz. The receiver demodulates it down to a slow complex signal (the real and imaginary parts), and the ADC measures that signal at regular intervals. These samples are what the computer receives.',
      dur=24, cam='whole', show={'headcoil': None, 'preamp': None, 'cable': None, 'cable2': None, 'adc': None, 'pc': None, 'monitor': None, 'patient': 0.3, 'table': 0.4, 'bore': 0.06, 'cover': 0.08, 'pedestal': 0.4, 'room': 0.12}, uses=('chain',), table=0.0)
def s3_2(app, t, u, S):
    if not S:
        d = app.axB('two'); tt = np.linspace(0, 1, 1200); S['tt'] = tt; env = np.exp(-tt * 4)
        d['a'].plot(tt, env * np.cos(2 * np.pi * 60 * tt), color=COL['rx'], lw=0.5); d['a'].plot(tt, env, color='#fde68a', lw=1); S['c1'] = d['a'].axvline(0, color='#fbbf24'); sty(d['a'], 'At the coil: carrier (shown slowed) with a decaying envelope', None, 'volts'); d['a'].set_xticks([])
        bb = env * np.exp(1j * 2 * np.pi * 3 * tt); d['b'].plot(tt, bb.real, color='#60a5fa', lw=1, label='real'); d['b'].plot(tt, bb.imag, color='#f472b6', lw=1, label='imaginary')
        ts = np.linspace(0, 1, 48); bs = np.exp(-ts * 4) * np.exp(1j * 2 * np.pi * 3 * ts); d['b'].plot(ts, bs.real, 'o', color='#fbbf24', ms=3); d['b'].legend(facecolor=PANEL, edgecolor='#334155', labelcolor=FG, fontsize=7, loc='upper right')
        S['c2'] = d['b'].axvline(0, color='#fbbf24'); sty(d['b'], 'After demodulation: complex baseband; dots are ADC samples', 'time', 'signal')
        c = app.axC('text'); app.put(c['a'], [('Typical numbers (3 T)', FG), 'carrier 127.7 MHz  ->  baseband bandwidth about +/- 16 kHz', 'dwell time = 1 / bandwidth = 31 microseconds per sample', '256 samples per line -> readout about 8 ms', '', 'Each complex sample is one point of k-space.', 'The receiver delivers about 1 V to the ADC.'], size=8.6, dy=0.12)
    S['c1'].set_xdata([u, u]); S['c2'].set_xdata([u, u]); app.get('chain').set(app.clock * 0.25)
    app.decode(0 if u < 0.5 else 1); app.mark('B', 'C', 'D')

# ----------------------------------------------------------------------------------------------
# PHASE 4  slice selection (Z gradient)
# ----------------------------------------------------------------------------------------------
phase_info(4, 'Slice selection', 'A linear Z gradient makes the Larmor frequency change from head to toe. An RF pulse containing only a band of frequencies (the transmit bandwidth) then excites only one slab. Slice thickness = bandwidth / (gamma x G).')

@step(4, 'A gradient makes frequency depend on position', 'With only B0, every proton from head to toe has the same Larmor frequency. Two electromagnets at the ends add fields with the same sign: the pattern repeats on both sides and is not unique. Reversing one current gives a field that changes linearly and uniquely along z.',
      dur=24, cam='scanner', show={'gz': None, 'bore': 0.1, 'patient': 0.2, 'table': 0.3, 'gshell': 0.1, 'cover': 0.05}, uses=('curz', 'probez'), table=0.0)
def s4_1(app, t, u, S):
    mode = 0 if u < 0.25 else 1 if u < 0.6 else 2
    if S.get('mode') != mode:
        S['mode'] = mode; zs = np.linspace(-0.35, 0.35, 41); S['zs'] = zs; d = app.axB('one'); a = d['a']
        if mode == 0: y = np.zeros_like(zs); txt = 'Uniform B0: every slice has the same frequency'
        elif mode == 1: y, _ = maxwell_profile(zs, opposite=False); txt = 'Same-direction currents: symmetric, so two slices share a frequency'
        else: y, _ = maxwell_profile(zs, opposite=True); txt = 'Opposite currents: linear, so each position has its own frequency'
        S['y'] = y; a.plot(zs, y, color=COL['gz'], lw=2.2); a.axhline(0, color='#475569', lw=0.8); a.set_ylim(-6, 6); sty(a, txt, 'position along z (m)', 'field added to B0 (mT)')
        if mode == 1: a.plot([-0.2, 0.2], [y[np.argmin(abs(zs + 0.2))]] * 2, '--', color='#fbbf24')
        c = app.axC('text')
        app.put(c['a'], [('f(z) = 42.58 MHz/T x (B0 + G z)', FG), 'G = 10 mT/m: over 0.7 m the frequency spans about +/- 150 kHz', '', ('Step 1: uniform B0 - same frequency everywhere', '#94a3b8' if mode else '#fbbf24'), ('Step 2: coils with equal current - not unique', '#94a3b8' if mode != 1 else '#fbbf24'),
                         ('Step 3: opposite currents - unique, linear', '#fbbf24' if mode == 2 else '#94a3b8'), '', 'At the isocenter the added field is zero, so f = f0.'], size=8.6, dy=0.115)
    sg = [0, 0] if mode == 0 else ([1, 1] if mode == 1 else [1, -1])
    cz = app.get('curz'); cz.show(mode > 0); cz.set(app.clock * 0.2, signs=sg)
    app.get('probez').set(S['y'], -6, 6, 'coolwarm'); app.mark('B', 'C')

@step(4, 'RF bandwidth picks the slab', 'To excite one slice, turn on the gradient and send an RF pulse containing only the Larmor frequencies of that slice. The slice position follows the centre frequency, and the thickness follows the transmit bandwidth. Wider bandwidth means a thicker slice, steeper gradient a thinner one.',
      dur=22, cam='head', show={'patient': 0.35, 'table': 0.4, 'bodycoil': 0.4, 'bore': 0.05}, uses=('slab', 'rf', 'probez'), table=0.0)
def s4_2(app, t, u, S):
    G = 10.0e-3; bw = 1000 + 5000 * float(smooth(1 - abs(2 * u - 1))); z0 = 0.06 * np.sin(2 * np.pi * u)
    th = bw / (GAMMA_MHZ_T * 1e6 * G)
    if not S:
        d = app.axB('one'); S['a'] = d['a']; zs = np.linspace(-0.35, 0.35, 100); d['a'].plot(zs, GAMMA_MHZ_T * G * zs * 1e6 / 1e3, color=COL['gz'], lw=2); S['band'] = d['a'].axhspan(0, 1, color='#fde047', alpha=0.35)
        sty(d['a'], 'Frequency offset versus position, with the RF band shaded', 'position along z (m)', 'offset from f0 (kHz)'); d['a'].set_ylim(-160, 160); S['txt'] = app.axC('text')['a']
    S['band'].remove(); S['band'] = S['a'].axhspan((GAMMA_MHZ_T * G * z0 * 1e3) - bw / 2e3, (GAMMA_MHZ_T * G * z0 * 1e3) + bw / 2e3, color='#fde047', alpha=0.35)
    app.get('slab').set(HEAD_Y * 0 + app.w.table_z + z0, th); app.get('rf').set(app.clock * 0.7, rmax=0.25)
    pz = GAMMA_MHZ_T * G * np.linspace(-0.35, 0.35, 41) * 1e3; app.get('probez').set(pz, -150, 150, 'coolwarm')
    S['txt'].clear(); app.put(S['txt'], [('Slice thickness', FG), 'thickness = bandwidth / (gamma x G)', 'bandwidth = %.0f Hz,  G = %.0f mT/m' % (bw, G * 1e3), 'thickness = %.1f mm' % (th * 1e3), '', 'The slab centre follows the RF centre frequency.', 'Offset = %.0f kHz -> %.0f mm from isocenter' % (GAMMA_MHZ_T * G * z0 * 1e3, z0 * 1e3), '', 'Only protons inside the slab are tipped;', 'the rest stay along B0 and send no signal.'], size=8.6, dy=0.105)
    app.mark('B', 'C')

# ----------------------------------------------------------------------------------------------
# PHASE 5  frequency encoding (X gradient)
# ----------------------------------------------------------------------------------------------
phase_info(5, 'Frequency encoding', 'With the slice excited, a gradient along x makes each column of voxels precess at its own frequency. The coil records the sum of all columns. A Fourier transform separates the frequencies and so tells us which column each part of the signal came from.')

def decay_wave(tt, f, A, T2=0.5): return A * np.exp(-tt / T2) * np.cos(2 * np.pi * f * tt)

@step(5, 'Two voxels, one frequency: no way to separate', 'Take a slice with two voxels, CSF and fat. With no extra gradient both see B0 and precess at the same frequency. Their curves have the same wavelength, differ only in amplitude, and simply add. The coil sees the sum and cannot tell the parts apart.',
      dur=18, cam='head', show={'patient': 0.2, 'table': 0.3, 'headcoil': 0.3}, uses=('arr2', 'mvec3', 'slab'), table=0.0)
def s5_1(app, t, u, S):
    if not S:
        d = app.axB('two'); tt = np.linspace(0, 1, 500); S['tt'] = tt
        d['a'].plot(tt, decay_wave(tt, 6, 1.0), color='#6cc4ff', lw=1.2, label='CSF voxel'); d['a'].plot(tt, decay_wave(tt, 6, 0.6), color='#ff8a5c', lw=1.2, label='fat voxel'); d['a'].legend(facecolor=PANEL, edgecolor='#334155', labelcolor=FG, fontsize=7, loc='upper right')
        sty(d['a'], 'Individual voxel signals (the machine cannot see these)', None, None); d['a'].set_xticks([])
        d['b'].plot(tt, decay_wave(tt, 6, 1.6), color=COL['rx'], lw=1.4); sty(d['b'], 'What the coil records: their sum', 'time', 'signal'); S['c1'] = d['a'].axvline(0, color='#fbbf24'); S['c2'] = d['b'].axvline(0, color='#fbbf24')
        c = app.axC('one'); S['bars'] = c['a'].bar(['CSF', 'fat'], [1.0, 0.6], color=['#6cc4ff', '#ff8a5c'], alpha=0.35, hatch='//'); c['a'].set_ylim(0, 2); c['a'].text(0.5, 1.75, 'sum = 1.6   (could be 1.0 + 0.6, or 0.8 + 0.8, or ...)', color=FG, ha='center', fontsize=8)
        sty(c['a'], 'Unknown amplitudes: the sum alone does not give them', None, 'amplitude')
    S['c1'].set_xdata([u, u]); S['c2'].set_xdata([u, u]); ph = OM * app.clock; ex = np.exp(-t / 12.0)
    o = np.array([hp(app, -0.035, 0, 0), hp(app, 0.035, 0, 0)]); v = np.array([[0, 1.0 * ex * np.cos(ph), 1.0 * ex * np.sin(ph)], [0, 0.6 * ex * np.cos(ph), 0.6 * ex * np.sin(ph)]])
    app.get('arr2').set(o, v, np.array([hex2rgb('#6cc4ff'), hex2rgb('#ff8a5c')]), L=0.07); app.get('mvec3').set(hp(app, 0, 0.1, 0), (v[0] + v[1]) * 0.9); app.get('slab').set(app.w.table_z, 0.004); app.mark('B', 'C')

@step(5, 'The X gradient gives each column its own frequency', 'Turn on a gradient along x after excitation. The left voxel now feels a slightly lower field than the right, so it precesses slower. The summed signal becomes more complex. A Fourier transform finds the individual frequencies, and low frequency means left.',
      dur=24, cam='head', show={'gx': 0.9, 'patient': 0.2, 'table': 0.3, 'bore': 0.05}, uses=('arr2', 'slab', 'probex'), table=0.0)
def s5_2(app, t, u, S):
    gon = u > 0.25; tt = np.linspace(0, 1, 600)
    if not S:
        d = app.axB('two'); S['d'] = d; S['sig'], = d['a'].plot(tt, decay_wave(tt, 6, 1.6), color=COL['rx'], lw=1.2); sty(d['a'], 'Recorded signal (sum of the two voxels)', 'time', 'signal')
        S['sp'], = d['b'].plot([], [], color='#fbbf24', lw=1.6); d['b'].set_xlim(0, 20); d['b'].set_ylim(0, 1.1); sty(d['b'], 'Fourier transform: amplitude versus frequency', 'frequency', 'amplitude')
        c = app.axC('one'); S['cb'] = c['a'].bar(['left (CSF)', 'right (fat)'], [0, 0], color=['#6cc4ff', '#ff8a5c']); c['a'].set_ylim(0, 1.2); sty(c['a'], 'Reconstructed row: amplitude placed by frequency', None, 'amplitude')
    f1, f2 = (6, 6) if not gon else (6 - 2.0 * float(smooth((u - 0.25) / 0.2)), 6 + 2.0 * float(smooth((u - 0.25) / 0.2)))
    ph = OM * app.clock * np.array([1.0, 1.0]) + 2 * np.pi * np.array([f1, f2]) * 0.0
    s = decay_wave(tt, f1, 1.0) + decay_wave(tt, f2, 0.6); S['sig'].set_ydata(s)
    N = 4096; F = np.abs(np.fft.rfft(s, N)); fr = np.fft.rfftfreq(N, d=tt[1] - tt[0]); m = (fr > 0) & (fr < 20); S['sp'].set_data(fr[m], F[m] / F.max())
    S['cb'][0].set_height(1.0 if gon else 0); S['cb'][1].set_height(0.6 if gon else 0)
    rot = OM * app.clock * (1 + (np.array([-1, 1]) * 0.35 * float(smooth((u - 0.25) / 0.2)) if gon else 0)); S['ph'] = S.get('ph', np.zeros(2)) + dtc(app, S) * OM * np.array([f1, f2]) / 6.0; ph = S['ph']
    o = np.array([hp(app, -0.035, 0, 0), hp(app, 0.035, 0, 0)]); v = np.array([[0, np.cos(ph[0]), np.sin(ph[0])], [0, 0.6 * np.cos(ph[1]), 0.6 * np.sin(ph[1])]])
    app.get('arr2').set(o, v, np.array([hex2rgb('#6cc4ff'), hex2rgb('#ff8a5c')]), L=0.07); app.get('slab').set(app.w.table_z, 0.004)
    xs = np.linspace(-0.3, 0.3, 31); app.get('probex').set((xs / 0.3) * (1 if gon else 0), -1, 1, 'coolwarm')
    app.decode(3 if u > 0.7 else None, 'Fourier transform: split one summed signal into its frequencies' if u > 0.7 else None); app.mark('B', 'C', 'D')
# ----------------------------------------------------------------------------------------------
# sequence-diagram and k-space helpers
# ----------------------------------------------------------------------------------------------
SEQ = {   # normalized time 0..1;  rf=(t,w,amp), grads = (t0,t1,amp)
    'basic': dict(rf=[(0.06, 0.06, 0.8)], gz=[(0.03, 0.09, 0.8), (0.09, 0.12, -0.8)], gy=[(0.14, 0.24, 0.7)], gx=[(0.28, 0.40, -0.7), (0.40, 0.70, 0.7)], adc=[(0.40, 0.70)]),
    'se':    dict(rf=[(0.06, 0.05, 0.8), (0.28, 0.07, 1.0)], gz=[(0.03, 0.09, 0.8), (0.09, 0.12, -0.8), (0.25, 0.31, 0.8)], gy=[(0.16, 0.23, 0.7)], gx=[(0.13, 0.20, 0.7), (0.39, 0.59, 0.7)], adc=[(0.39, 0.59)]),
    'tse':   dict(rf=[(0.04, 0.03, 0.8)] + [(0.11 + 0.12 * k, 0.04, 1.0) for k in range(4)], gz=[(0.025, 0.055, 0.8)] + [(0.095 + 0.12 * k, 0.125 + 0.12 * k, 0.8) for k in range(4)],
                  gy=[(0.13 + 0.12 * k - 0.014, 0.13 + 0.12 * k - 0.004, 0.6 - 0.3 * k) for k in range(4)] + [(0.20 + 0.12 * k, 0.21 + 0.12 * k, -(0.6 - 0.3 * k)) for k in range(4)], gx=[(0.145 + 0.12 * k, 0.215 + 0.12 * k, 0.6) for k in range(4)], adc=[(0.145 + 0.12 * k, 0.215 + 0.12 * k) for k in range(4)]),
    'gre':   dict(rf=[(0.06, 0.05, 0.5)], gz=[(0.03, 0.09, 0.8), (0.09, 0.12, -0.8)], gy=[(0.14, 0.22, 0.7)], gx=[(0.14, 0.26, -0.7), (0.26, 0.60, 0.7)], adc=[(0.26, 0.60)]),
    'epi':   dict(rf=[(0.05, 0.05, 0.8), (0.17, 0.06, 1.0)], gz=[(0.02, 0.08, 0.8), (0.14, 0.20, 0.8)], gy=[(0.22, 0.26, 0.6)] + [(0.30 + 0.085 * k, 0.315 + 0.085 * k, 0.4) for k in range(7)],
                 gx=[(0.28 + 0.085 * k, 0.36 + 0.085 * k, 0.7 if k % 2 == 0 else -0.7) for k in range(8)], adc=[(0.285 + 0.085 * k, 0.355 + 0.085 * k) for k in range(8)]),
}
ROWY = dict(rf=4.0, gz=3.0, gy=2.0, gx=1.0, adc=0.0)
ROWC = dict(rf=COL['rf'], gz=COL['gz'], gy=COL['gy'], gx=COL['gx'], adc=COL['rx'])

def draw_seq(ax, kind):
    sq = SEQ[kind]; ax.set_xlim(0, 1); ax.set_ylim(-0.7, 5.0); ax.set_yticks([ROWY[k] for k in ROWY]); ax.set_yticklabels(['RF', 'Gz', 'Gy', 'Gx', 'ADC']); ax.set_xticks([])
    for k, y in ROWY.items(): ax.axhline(y, color='#334155', lw=0.6)
    for t, w, a in sq['rf']:
        x = np.linspace(t - w / 2, t + w / 2, 50); ax.plot(x, ROWY['rf'] + a * 0.8 * np.sinc((x - t) / (w / 3)), color=ROWC['rf'], lw=1.3)
    for key in ('gz', 'gy', 'gx'):
        for t0, t1, a in sq[key]:
            r = min(0.012, (t1 - t0) / 4); ax.fill([t0, t0 + r, t1 - r, t1], [ROWY[key], ROWY[key] + a * 0.8, ROWY[key] + a * 0.8, ROWY[key]], color=ROWC[key], alpha=0.75)
    for t0, t1 in sq['adc']: ax.add_patch(Rectangle((t0, -0.25), t1 - t0, 0.5, fc=ROWC['adc'], ec='none', alpha=0.8))
    ax.set_title('Pulse sequence diagram  (time runs left to right)', color=FG, fontsize=8.5, loc='left', pad=3)
    return ax.axvline(0, color='#fbbf24', lw=1.2)

def seq_active(kind, u):
    sq = SEQ[kind]; r = {'rf': False, 'gz': False, 'gy': False, 'gx': False, 'adc': False}
    for t, w, a in sq['rf']: r['rf'] |= abs(u - t) < w / 2
    for key in ('gz', 'gy', 'gx'):
        for t0, t1, a in sq[key]: r[key] |= t0 <= u <= t1
    for t0, t1 in sq['adc']: r['adc'] |= t0 <= u <= t1
    return r

def glow(app, kind, u, base=0.22):
    r = seq_active(kind, u)
    for n, key in (('gx', 'gx'), ('gy', 'gy'), ('gz', 'gz'), ('bodycoil', 'rf'), ('headcoil', 'adc')):
        app.w.target_op[n] = 1.0 if r[key] else base
    return r

SCAN_SHOW = {'gx': 0.3, 'gy': 0.3, 'gz': 0.3, 'bodycoil': 0.3, 'headcoil': 0.3, 'patient': 0.35, 'table': 0.5, 'bore': 0.07, 'cover': 0.05, 'b0': 0.12, 'pedestal': 0.4, 'gshell': 0.07}

N64 = 64
IMG64 = weighted(2500, 20)[::2, ::2]; K64 = to_kspace(IMG64)
def klog(K, mask=None):
    m = np.log10(1 + np.abs(K) * 50); m = m / m.max()
    if mask is not None: m = m * mask
    return m
def recon(K, mask): return from_kspace(K * mask)

# ----------------------------------------------------------------------------------------------
# PHASE 6  phase encoding (Y gradient)
# ----------------------------------------------------------------------------------------------
phase_info(6, 'Phase encoding', 'Frequency encoding separates columns but not voxels that share a column. A short Y gradient before readout leaves each row with a different phase. Repeating the experiment with different gradient strengths gives one equation per row, which can be solved.')
VT = [('CSF', '#6cc4ff', 1.0), ('soft tissue', '#c4b5fd', 0.6), ('fat', '#ff8a5c', 0.8), ('bone', '#a3a3a3', 0.2)]
GRID4 = lambda app: np.array([hp(app, -0.035, 0.035, 0), hp(app, -0.035, -0.035, 0), hp(app, 0.035, 0.035, 0), hp(app, 0.035, -0.035, 0)])   # CSF, ST (left col), fat, bone (right col)

def spec_plot(ax, sig, tt, xmax=20, color='#fbbf24'):
    N = 8192; F = np.abs(np.fft.rfft(sig, N)); fr = np.fft.rfftfreq(N, d=tt[1] - tt[0]); m = (fr > 0) & (fr < xmax)
    return fr[m], F[m] / (F.max() + 1e-9)

@step(6, 'Frequency encoding alone gives columns, not voxels', 'Now take a 2 by 2 slice: CSF and soft tissue in the left column, fat and bone in the right. The X gradient gives each column one frequency. The Fourier transform finds two frequencies, but each peak is the sum of two voxels. We know the column totals only.',
      dur=20, cam='head', show={'gx': 0.8, 'patient': 0.2, 'table': 0.3, 'bore': 0.05}, uses=('arr4', 'slab'), table=0.0)
def s6_1(app, t, u, S):
    tt = np.linspace(0, 1, 800)
    if not S:
        d = app.axB('two'); sig = sum(decay_wave(tt, f, a) for f, a in ((5, 1.0), (5, 0.6), (8, 0.8), (8, 0.2))); S['sig'] = d['a'].plot(tt, sig, color=COL['rx'], lw=1.1)[0]; sty(d['a'], 'Recorded signal: four voxels summed', 'time', 'signal')
        fr, F = spec_plot(None, sig, tt, 14); d['b'].plot(fr, F, color='#fbbf24'); d['b'].set_xlim(0, 14); sty(d['b'], 'Fourier transform: only two frequencies (peak heights = column sums 1.6 and 1.0)', 'frequency', None)
        c = app.axC('one'); a = c['a']; a.set_xlim(0, 2); a.set_ylim(0, 2); a.set_aspect('equal'); a.set_xticks([]); a.set_yticks([])
        for k, (n_, c_, v_) in enumerate(VT):
            xx, yy = (0, 1) if k == 0 else (0, 0) if k == 1 else (1, 1) if k == 2 else (1, 0); a.add_patch(Rectangle((xx, yy), 1, 1, fc=c_, alpha=0.25, ec='#475569')); a.text(xx + 0.5, yy + 0.5, n_ + '\n?', color=FG, ha='center', va='center', fontsize=9)
        a.text(0.5, 2.08, 'left column total 1.6', color='#fbbf24', ha='center', fontsize=8.5); a.text(1.5, 2.08, 'right total 1.0', color='#fbbf24', ha='center', fontsize=8.5); sty(a, 'What we can recover: column totals only', None, None)
    ph = OM * app.clock * np.array([1.0, 1.0, 1.6, 1.6]); ex = np.exp(-t / 12)
    v = np.array([[0, a * ex * np.cos(p), a * ex * np.sin(p)] for (n_, c_, a), p in zip(VT, ph)]); app.get('arr4').set(GRID4(app), v, np.array([hex2rgb(c_) for n_, c_, a in VT]), L=0.06); app.get('slab').set(app.w.table_z, 0.004)
    app.decode(3 if u > 0.6 else None); app.mark('B', 'C', 'D')

@step(6, 'A second gradient at the same time does not help', 'Turn on a Y gradient together with the X gradient. Fields now add: CSF gets B0 minus 2 gradient steps, bone gets B0 plus 2, and soft tissue and fat both stay at B0. Four voxels give only three frequencies, so the Fourier transform still cannot split them.',
      dur=18, cam='head', show={'gx': 0.8, 'gy': 0.8, 'patient': 0.2, 'table': 0.3, 'bore': 0.05}, uses=('arr4', 'slab'), table=0.0)
def s6_2(app, t, u, S):
    tt = np.linspace(0, 1, 800); fr_ = np.array([4, 6, 6, 8.0])
    if not S:
        d = app.axB('two'); sig = sum(decay_wave(tt, f, a) for f, (n_, c_, a) in zip(fr_, VT)); d['a'].plot(tt, sig, color=COL['rx'], lw=1.0); sty(d['a'], 'Signal with X and Y gradients on', 'time', 'signal')
        fr, F = spec_plot(None, sig, tt, 14); d['b'].plot(fr, F, color='#fbbf24'); d['b'].set_xlim(0, 14); sty(d['b'], 'Three peaks for four voxels: the middle one is fat + soft tissue', 'frequency', None)
        c = app.axC('text'); app.put(c['a'], [('Fields at the four voxels (Bx = By)', FG), 'CSF:           B0 - 2 Bx', 'soft tissue:  B0 - Bx + Bx = B0', 'fat:           B0 + Bx - Bx = B0', 'bone:          B0 + 2 Bx', '', 'Unique frequencies: 3.   Voxels: 4.', 'Making the gradients unequal could work for 4 voxels,', 'but a real image has 256 x 256 = 65,536 voxels that', 'would all need distinct fields. We need another property: phase.'], size=8.6, dy=0.1)
    S['ph'] = S.get('ph', np.zeros(4)) + dtc(app, S) * OM * fr_ / 5.0; ph = S['ph']; ex = np.exp(-t / 12)
    v = np.array([[0, a * ex * np.cos(p), a * ex * np.sin(p)] for (n_, c_, a), p in zip(VT, ph)]); app.get('arr4').set(GRID4(app), v, np.array([hex2rgb(c_) for n_, c_, a in VT]), L=0.06); app.get('slab').set(app.w.table_z, 0.004); app.mark('B', 'C')

@step(6, 'Phase: the missing property', 'Phase is the angle between rotating magnetizations. Two voxels in one column spin at the same rate, but one can lead the other. In phase their signals add (A + B). Out of phase they subtract (A - B). We cannot see A and B alone, but we can measure how the sum changes with the phase shift.',
      dur=26, cam='head', show={'patient': 0.2, 'table': 0.3, 'headcoil': 0.3}, uses=('arr2', 'slab'), table=0.0)
def s6_3(app, t, u, S):
    A_, B_ = 6.0, 2.0; tt = np.linspace(0, 1, 600)
    dphi = 0.0 if u < 0.25 else np.pi * float(smooth((u - 0.25) / 0.15)) if u < 0.55 else np.pi
    if u < 0.55 and u >= 0.4: dphi = np.pi
    if not S:
        d = app.axB('two'); S['l1'], = d['a'].plot(tt, A_ * decay_wave(tt, 5, 1.0), color='#6cc4ff', lw=1.2, label='CSF voxel (A)'); S['l2'], = d['a'].plot(tt, B_ * decay_wave(tt, 5, 1.0), color='#c4b5fd', lw=1.2, label='soft tissue voxel (B)')
        d['a'].legend(facecolor=PANEL, edgecolor='#334155', labelcolor=FG, fontsize=7, loc='upper right'); sty(d['a'], 'Same frequency, shifted phase', None, None); d['a'].set_xticks([]); d['a'].set_ylim(-8.5, 8.5)
        S['l3'], = d['b'].plot(tt, (A_ + B_) * decay_wave(tt, 5, 1.0), color=COL['rx'], lw=1.4); d['b'].set_ylim(-9, 9); sty(d['b'], 'What the coil records: A + B cos(shift)', 'time', 'signal'); S['cur'] = d['b'].axvline(0, color='#fbbf24')
        S['c'] = app.axC('text')['a']
    ex = lambda x: np.exp(-tt / 0.5) * np.cos(2 * np.pi * 5 * tt + x)
    S['l2'].set_ydata(B_ * ex(dphi)); S['l3'].set_ydata(A_ * ex(0) + B_ * ex(dphi)); S['cur'].set_xdata([u, u])
    rows = [('Measure the sum at a chosen TE (200 ms):', FG), ('', FG),
            ('In phase:         A + B = 8 amps', '#fbbf24' if u >= 0.1 else MUTED), ('Out of phase:    A - B = 4 amps', '#fbbf24' if u >= 0.55 else MUTED), ('', FG)]
    if u > 0.7: rows += [('Add the equations: 2A = 12, so A = 6 amps', '#86efac'), ('Subtract them:     2B = 4,    so B = 2 amps', '#86efac'), ('', FG), ('Two unknowns, two equations: both voxels found.', FG)]
    S['c'].clear(); app.put(S['c'], rows, size=9, dy=0.1)
    ph = OM * app.clock; o = np.array([hp(app, 0, 0.035, 0), hp(app, 0, -0.035, 0)]); v = np.array([[0, A_ / 6.0 * np.cos(ph), A_ / 6.0 * np.sin(ph)], [0, B_ / 6.0 * 1.4 * np.cos(ph + dphi), B_ / 6.0 * 1.4 * np.sin(ph + dphi)]])
    app.get('arr2').set(o, v, np.array([hex2rgb('#6cc4ff'), hex2rgb('#c4b5fd')]), L=0.07); app.get('slab').set(app.w.table_z, 0.004); app.mark('B', 'C')

@step(6, 'The Y gradient pulse writes a phase onto each row', 'Switch on a Y gradient briefly after excitation. Rows precess at different rates only while it is on. When it ends, all rows return to the same frequency, but each row keeps its phase lead. Repeat with a new gradient strength each time: one phase pattern per repetition.',
      dur=28, cam='head', show={'gy': 0.9, 'patient': 0.2, 'table': 0.3, 'bore': 0.05}, uses=('arr3', 'slab'), table=0.0)
def s6_4(app, t, u, S):
    nrep = 5; rep = min(int(u * nrep), nrep - 1); local = u * nrep - rep; amp = [-1.0, -0.5, 0.0, 0.5, 1.0][rep]
    if not S:
        d = app.axB('seq2'); S['cur'] = draw_seq(d['s'], 'basic'); S['seqax'] = d['s']
        S['kax'] = d['a']; S['kim'] = d['a'].imshow(np.zeros((9, 9)), cmap='viridis', vmin=0, vmax=1, extent=(-4.5, 4.5, -4.5, 4.5), origin='lower', interpolation='nearest'); sty(d['a'], 'k-space rows filled so far', 'kx', 'ky')
        S['wh'] = d['b']; d['b'].set_xlim(-1.3, 1.3); d['b'].set_ylim(-1.3, 1.3); d['b'].set_aspect('equal'); d['b'].add_patch(Circle((0, 0), 1, fill=False, ec='#475569')); S['pts'] = d['b'].scatter([0, 0, 0], [1, 1, 1], s=40, c=['#6cc4ff', '#c4b5fd', '#ff8a5c']); sty(d['b'], 'Row phases', None, None); d['b'].set_xticks([]); d['b'].set_yticks([])
        c = app.axC('text'); S['ct'] = c['a']; S['kk'] = np.zeros((9, 9))
    on = 0.14 < local < 0.24
    ph_acc = amp * 2 * np.pi * 0.9 * float(smooth((local - 0.14) / 0.10)); ys = np.array([0.035, 0.0, -0.035])
    phs = ph_acc * ys / 0.035; rate = 1.0 + (amp * ys / 0.035 * 0.5 if on else 0.0)
    ph = OM * app.clock + phs
    v = np.array([[0, np.cos(p), np.sin(p)] for p in ph]); app.get('arr3').set(np.array([hp(app, 0, y, 0) for y in ys]), v, np.array([hex2rgb(c_) for c_ in ('#6cc4ff', '#c4b5fd', '#ff8a5c')]), L=0.07); app.get('slab').set(app.w.table_z, 0.004)
    S['pts'].set_offsets(np.c_[np.cos(phs), np.sin(phs)])
    S['cur'].set_xdata([local, local]); S['kk'][:] = 0; S['kk'][:rep + 1, :] = 0
    for r_ in range(rep + (1 if local > 0.45 else 0)): S['kk'][r_ * 2, :] = 1
    S['kim'].set_data(S['kk'])
    g = seq_active('basic', local); app.w.target_op['gy'] = 1.0 if g['gy'] else 0.3; app.w.target_op['gx'] = 1.0 if g['gx'] else 0.25; app.w.target_op['gz'] = 1.0 if g['gz'] else 0.25
    S['ct'].clear(); app.put(S['ct'], [('Repetition %d of %d:  Gy amplitude = %+.1f' % (rep + 1, nrep, amp), '#fbbf24'), 'Phase gained = gamma x Gy x y x duration', 'Stronger or longer gradient -> bigger phase difference', '', 'One repetition per row of the image:', 'a 128 x 256 matrix needs 128 repetitions (or 256).', 'The smaller dimension is usually the phase-encode axis.'], size=8.8, dy=0.12)
    app.decode(2 if local > 0.45 else None, 'each repetition adds one line to k-space; the Gy strength picks the line'); app.mark('B', 'C', 'D')

# ----------------------------------------------------------------------------------------------
# PHASE 7  pixel to voxel
# ----------------------------------------------------------------------------------------------
phase_info(7, 'From pixel to voxel', 'A pixel is a flat square on the screen. In the scanner it is a voxel: the pixel area times the slice thickness. The coil records the sum of all voxel signals, and the encoding steps let the computer split it back into one number per voxel.')

@step(7, 'A pixel becomes a voxel', 'The slab picked by slice selection is cut into a grid. Each square of the grid is a pixel on the image and a small box of tissue in the head: a voxel, with width, height and thickness. Each box has its own T2 decay curve, and the coil records all of them added together.',
      dur=26, cam='head', show={'patient': 0.25, 'table': 0.3, 'bore': 0.04}, uses=('vox', 'slab'), table=0.0)
def s7_1(app, t, u, S):
    th = 0.0015 + 0.0085 * float(smooth((u - 0.15) / 0.3)); n = 16
    q = round(th * 2e3)
    if getattr(app, '_vq', None) != (app.i, q): app._vq = (app.i, q); app.get('vox').build(n, th, 0.17)
    app.get('vox').show(True); app.get('slab').show(False)
    if not S:
        d = app.axB('one'); tt = np.linspace(0, 1, 800); rng = np.random.default_rng(4); idx = [(4, 5), (7, 8), (9, 6), (11, 9), (5, 11), (8, 3)]; tot = 0
        for k, (i, j) in enumerate(idx):
            a = PH['pd'][i * 8, j * 8]; f = 4 + 0.5 * j + 0.15 * i; y = a * decay_wave(tt, f, 1.0, 0.6) + 0 * k; tot = tot + y
            d['a'].plot(tt, y - 2.2 * (k + 1), color='#94a3b8', lw=0.8)
        d['a'].plot(tt, tot + 1.5, color=COL['rx'], lw=1.3); d['a'].text(0.98, 3.2, 'sum = what the coil records', color=COL['rx'], ha='right', fontsize=8); d['a'].text(0.98, -2.0, 'individual voxels (not directly visible)', color=MUTED, ha='right', fontsize=7.5)
        S['cur'] = d['a'].axvline(0, color='#fbbf24'); sty(d['a'], 'Raw signal = sum of voxel signals', 'time', None); d['a'].set_yticks([])
        c = app.axC('img_txt'); S['im'] = c['a'].imshow(IMG64[::4, ::4], cmap='gray', interpolation='nearest'); c['a'].set_xticks([]); c['a'].set_yticks([]); sty(c['a'], '16 x 16 pixel grid (image)', None, None); S['t'] = c['t']
    S['cur'].set_xdata([u, u]); thick = th * 1e3
    S['t'].clear(); app.put(S['t'], [('Voxel size', '#fbbf24'), 'FOV 256 mm, matrix 256 x 256', '-> pixel = 1.0 x 1.0 mm', 'slice thickness 5 mm', '-> voxel = 1 x 1 x 5 mm', '        = 5 mm^3 = 5 microlitres', '', 'A 1 mm^3 box holds about', '3 x 10^19 water molecules.', '', 'Here, 16 x 16 for display.', 'Drawn thickness: %.1f mm' % thick], size=8.4, dy=0.075)
    app.decode(5 if u > 0.8 else None, 'the image is a matrix: one number per voxel, shown as gray level' if u > 0.8 else None); app.mark('B', 'C', 'D')

@step(7, 'Scan time = TR x number of phase encodes', 'Each repetition fills one row, so the total time is TR times the number of rows times averages. With TR of 500 ms and 256 rows, a single slice takes about two minutes. This cost is why faster sequences were invented (phase 9).',
      dur=22, cam='scanner', show=dict(SCAN_SHOW), uses=(), table=0.0)
def s7_2(app, t, u, S):
    if not S:
        d = app.axB('one'); S['bars'] = d['a'].barh([0], [0], color='#38bdf8'); d['a'].set_xlim(0, 128); d['a'].set_ylim(-1, 1); d['a'].set_yticks([]); sty(d['a'], 'Elapsed scan time (s) for 256 rows at TR = 500 ms', 'seconds', None)
        c = app.axC('text'); S['c'] = c['a']
    rows = int(256 * u); elapsed = rows * 0.5; S['bars'][0].set_width(elapsed)
    ph = (app.clock * 2) % 1.0; g = seq_active('basic', ph)
    for n_, k in (('gx', 'gx'), ('gy', 'gy'), ('gz', 'gz'), ('bodycoil', 'rf'), ('headcoil', 'adc')): app.w.target_op[n_] = 1.0 if g[k] else 0.25
    S['c'].clear(); app.put(S['c'], [('Time = TR x N(phase steps) x NEX', FG), 'TR = 500 ms,  N = 256,  NEX = 1', 'T = 0.5 s x 256 = 128 s  (2 min 8 s)', '', 'Rows acquired so far: %d / 256' % rows, 'Elapsed: %.0f s' % elapsed, '', 'Halve N or TR -> halve the time.', 'Shorter TR changes contrast too (see phase 8).'], size=9, dy=0.115)
    app.mark('B', 'C')

# ----------------------------------------------------------------------------------------------
# PHASE 8  contrast
# ----------------------------------------------------------------------------------------------
phase_info(8, 'Image contrast', 'Contrast is a difference in measured values. The image value at each voxel is the amplitude of its T2 decay curve at time TE, after the previous pulse left it with the magnetization that regrew during TR. Short TR with short TE gives T1 weighting, long TR with long TE gives T2 weighting.')
TT = [('WM', '#ffd27a', 0.70, 830., 80.), ('GM', '#b0a0ff', 0.80, 1330., 110.), ('CSF', '#6cc4ff', 1.00, 4000., 2000.)]
TPOS = lambda app: np.array([hp(app, -0.04, 0, 0), hp(app, 0.0, 0, 0), hp(app, 0.04, 0, 0)])
TCOL = np.array([hex2rgb(c) for n, c, *_ in TT])

@step(8, 'The image is the signal at TE', 'After a 90 degree pulse the three tissues start with different amounts of M and decay at different T2 rates. Choose a time TE and read each curve there: those three numbers become gray levels. A different TE gives a different picture of the same tissue.',
      dur=24, cam='head', show={'patient': 0.2, 'table': 0.3, 'bore': 0.04}, uses=('arr3', 'slab'), table=0.0)
def s8_1(app, t, u, S):
    te = 5 + 170 * u
    if not S:
        d = app.axB('one'); tt = np.linspace(0, 200, 300)
        for n_, c_, pd_, t1_, t2_ in TT: d['a'].plot(tt, pd_ * np.exp(-tt / t2_), color=c_, lw=1.8, label='%s  (T2 = %d ms)' % (n_, t2_))
        d['a'].legend(facecolor=PANEL, edgecolor='#334155', labelcolor=FG, fontsize=7.5); S['cur'] = d['a'].axvline(0, color='#fbbf24'); sty(d['a'], 'T2 decay curves (transverse signal after a 90 degree pulse)', 'time after the pulse = TE (ms)', 'signal')
        c = app.axC('lr'); S['sq'] = []
        c['a'].set_xlim(0, 3); c['a'].set_ylim(0, 1.4); c['a'].axis('off')
        for k in range(3): S['sq'].append(c['a'].add_patch(Rectangle((k + 0.05, 0.2), 0.9, 0.9, fc='#000', ec='#475569')))
        for k, (n_, *_r) in enumerate(TT): c['a'].text(k + 0.5, 0.05, n_, color=FG, ha='center', fontsize=8.5)
        sty(c['a'], 'Gray level at this TE', None, None); c['a'].axis('off'); S['bar'] = c['b'].bar([n for n, *_ in TT], [0, 0, 0], color=[c for n, c, *_ in TT]); c['b'].set_ylim(0, 1.05); sty(c['b'], 'Measured value', None, None)
    S['cur'].set_xdata([te, te]); vals = [pd_ * np.exp(-te / t2_) for n_, c_, pd_, t1_, t2_ in TT]
    for k, v_ in enumerate(vals): S['sq'][k].set_facecolor((v_,) * 3); S['bar'][k].set_height(v_)
    tau = te / 12.0; ph = OM * app.clock; vec = np.array([[0, v_ * np.cos(ph), v_ * np.sin(ph)] for v_ in vals]); app.get('arr3').set(TPOS(app), vec, TCOL, L=0.07); app.get('slab').set(app.w.table_z, 0.004); app.mark('B', 'C')

@step(8, 'TR sets how much M is ready: T1 weighting', 'The next pulse arrives after a repetition time TR. If TR is short, tissues with a short T1 have regrown more magnetization than tissues with a long T1, so they give more signal. A short TR therefore lets T1 differences show up in the image.',
      dur=24, cam='head', show={'patient': 0.2, 'table': 0.3, 'bore': 0.04}, uses=('arr3',), table=0.0)
def s8_2(app, t, u, S):
    TR = 200 + 3800 * float(smooth(1 - abs(2 * u - 1)))
    if not S:
        d = app.axB('one'); tt = np.linspace(0, 5000, 300)
        for n_, c_, pd_, t1_, t2_ in TT: d['a'].plot(tt, pd_ * (1 - np.exp(-tt / t1_)), color=c_, lw=1.8, label='%s  (T1 = %d ms)' % (n_, t1_))
        d['a'].legend(facecolor=PANEL, edgecolor='#334155', labelcolor=FG, fontsize=7.5, loc='lower right'); S['cur'] = d['a'].axvline(0, color='#fbbf24'); sty(d['a'], 'M along z available when the next pulse arrives', 'TR (ms)', 'Mz')
        c = app.axC('lr'); c['a'].axis('off'); S['sq'] = [c['a'].add_patch(Rectangle((k + 0.05, 0.2), 0.9, 0.9, fc='#000', ec='#475569')) for k in range(3)]; c['a'].set_xlim(0, 3); c['a'].set_ylim(0, 1.4)
        for k, (n_, *_r) in enumerate(TT): c['a'].text(k + 0.5, 0.05, n_, color=FG, ha='center', fontsize=8.5)
        S['t'] = c['b']
    S['cur'].set_xdata([TR, TR]); vals = [pd_ * (1 - np.exp(-TR / t1_)) for n_, c_, pd_, t1_, t2_ in TT]; mx = max(vals + [0.7])
    for k, v_ in enumerate(vals): S['sq'][k].set_facecolor((min(1, v_ / mx),) * 3)
    S['t'].clear(); S['t'].axis('off'); app.put(S['t'], [('TR = %d ms' % TR, '#fbbf24'), 'short TR: WM > GM > CSF', '   (large differences)', 'long TR: all fully regrown', '   (small differences)', '', 'T1 itself is never seen directly;', 'it only sets the starting size of', 'the next T2 curve.'], size=8.6, dy=0.1)
    app.get('arr3').set(TPOS(app), np.array([[v_ / mx * 1.0, 0, 0] for v_ in vals]), TCOL, L=0.1); app.mark('B', 'C')

@step(8, 'TR and TE together: T1, PD and T2 weighting', 'T1-weighted: short TR (about 500 ms) and short TE (about 15 ms). T2-weighted: long TR and long TE. Proton-density weighted: long TR and short TE. A short TR with a long TE is not used, because TR must always be longer than TE.',
      dur=30, cam='head', show={'patient': 0.15, 'table': 0.2, 'bore': 0.04}, uses=('arr3',), table=0.0)
def s8_3(app, t, u, S):
    presets = [('T1-weighted', 500, 15), ('PD-weighted', 2500, 20), ('T2-weighted', 4000, 100)]; k = min(int(u * 3), 2); name, TR, TE = presets[k]
    if not S:
        c = app.axC('three'); S['ims'] = []
        for key, (nm, tr, te) in zip('abc', presets):
            im = c[key].imshow(weighted(tr, te), cmap='gray', vmin=0, vmax=1.0); c[key].set_xticks([]); c[key].set_yticks([]); sty(c[key], '%s  TR %d / TE %d' % (nm, tr, te), None, None); c[key].set_xticks([]); c[key].set_yticks([]); S['ims'].append(c[key])
        d = app.axB('one'); S['d'] = d['a']; d['a'].set_xlim(0, 4500); d['a'].set_ylim(0, 130); sty(d['a'], 'Choice of (TR, TE) sets the weighting', 'TR (ms)', 'TE (ms)')
        for nm, tr, te in presets: d['a'].plot([tr], [te], 'o', color='#fbbf24', ms=7); d['a'].text(tr + 80, te + 6, nm, color=FG, fontsize=7.5)
        S['pt'], = d['a'].plot([500], [15], 'o', color='#22d3ee', ms=11, mfc='none', mew=2)
    for j, a in enumerate(S['ims']):
        for sp_ in a.spines.values(): sp_.set_color('#fbbf24' if j == k else '#334155'); sp_.set_linewidth(3 if j == k else 1)
    S['pt'].set_data([TR], [TE]); vals = [pd_ * (1 - np.exp(-TR / t1_)) * np.exp(-TE / t2_) for n_, c_, pd_, t1_, t2_ in TT]; mx = 0.55
    ph = OM * app.clock; app.get('arr3').set(TPOS(app), np.array([[0, v_ / mx * np.cos(ph), v_ / mx * np.sin(ph)] for v_ in vals]), TCOL, L=0.08); app.mark('B', 'C')

# ----------------------------------------------------------------------------------------------
# PHASE 9  echoes and pulse sequences
# ----------------------------------------------------------------------------------------------
phase_info(9, 'Echoes and pulse sequences', 'A spin echo uses a 180 degree pulse to reverse the dephasing so the signal re-emerges at TE, when the gradients can encode it. Turbo spin echo repeats the 180 degree pulse to collect many lines per TR, HASTE collects half the data in one shot, gradient echo uses a reversed gradient, and EPI oscillates the readout gradient.')

@step(9, 'The echo: reversing the dephasing', 'Signal fades because spins run at different rates. A small second pulse does not bring it back. A 180 degree pulse does: it mirrors every phase, so the fast spins that were ahead are now behind. They catch up at TE and the signal re-emerges.',
      dur=28, cam='head', show={'patient': 0.2, 'table': 0.3, 'bodycoil': 0.4, 'bore': 0.04}, uses=('spins', 'rf'), table=0.0)
def s9_1(app, t, u, S):
    t1 = 0.33; te = 0.66; tau = u
    ph = OFFR * 9.0 * (tau if tau < t1 else (t1 - (tau - t1))) if tau < te else OFFR * 9.0 * ((tau - te))
    if tau < te: ph = OFFR * 9.0 * (tau if tau < t1 else 2 * t1 - tau)
    mxy = np.exp(-u * 1.2)
    if not S:
        d = app.axB('one'); S['x'] = []; S['y'] = []; S['ln'], = d['a'].plot([], [], color=COL['rx'], lw=1.8); d['a'].set_xlim(0, 1); d['a'].set_ylim(0, 1.05)
        for x_, lab, c_ in ((0.0, '90', '#fde047'), (t1, '180', '#fb923c'), (te, 'echo (TE)', '#22d3ee')): d['a'].axvline(x_, color=c_, lw=1, ls='--'); d['a'].text(x_ + 0.01, 0.97, lab, color=c_, fontsize=8)
        sty(d['a'], 'Net transverse signal', 'time (RF 90 at 0, RF 180 at TE/2, echo at TE)', '|Mxy|'); c = app.axC('text')
        app.put(c['a'], [('Why the 180 pulse works', FG), 'Before it: fast spins lead, slow spins lag.', 'The pulse reverses the direction of rotation.', 'Now the leaders are behind and catch up at TE.', '', 'A small second pulse (20 degrees) only tilts the spins;', 'they keep spreading and no echo appears.', 'The echo is not perfect: T2 losses are not undone.'], size=8.6, dy=0.11)
    v = v_iso(0.0, mxy, ph); app.get('spins').set(v, phase_rgb(ph % (2 * np.pi))); M = v.mean(0); S['x'].append(u); S['y'].append(np.hypot(M[1], M[2])); S['ln'].set_data(S['x'], S['y'])
    app.get('rf').show(abs(u - t1) < 0.03 or u < 0.03); app.get('rf').set(app.clock * 0.9); app.mark('B', 'C')

def seq_step_common(app, S, kind, tag):
    if not S:
        d = app.axB('seq1'); S['cur'] = draw_seq(d['s'], kind); S['ax'] = d['a']; return d
    return None

@step(9, 'Spin echo sequence', 'The classic sequence: 90 degree pulse with slice gradient, a Gy phase-encode lobe, a 180 degree pulse at TE/2, then the readout gradient with the ADC open around TE. This is repeated every TR with a new Gy strength. Watch the coils light up in the scanner as each part of the diagram plays.',
      dur=26, cam='scanner', show=dict(SCAN_SHOW), table=0.0)
def s9_2(app, t, u, S):
    ph = (u * 2.2) % 1.0
    d = seq_step_common(app, S, 'se', 'SE')
    if d:
        tt = np.linspace(0, 1, 500); sig = np.exp(-(tt - 0.06) / 0.08) * (tt > 0.06) * np.cos(2 * np.pi * 30 * tt) * 0.6 + np.exp(-((tt - 0.49) / 0.06) ** 2) * np.cos(2 * np.pi * 30 * tt) * 0.9 * (tt > 0.33)
        d['a'].plot(tt, sig, color=COL['rx'], lw=0.9); d['a'].set_xlim(0, 1); d['a'].set_xticks([0.06, 0.28, 0.49]); d['a'].set_xticklabels(['90', 'TE/2: 180', 'TE: echo']); sty(d['a'], 'Signal: FID after the 90, echo at TE', None, None); S['c1'] = d['a'].axvline(0, color='#fbbf24')
        c = app.axC('text'); S['ct'] = c['a']
    S['cur'].set_xdata([ph, ph]); S['c1'].set_xdata([ph, ph]); glow(app, 'se', ph)
    S['ct'].clear(); app.put(S['ct'], [('Definitions', FG), 'TE: time from the 90 pulse to the echo peak', 'TR: time between successive 90 pulses', 'Gz lobes: select the slice (both pulses)', 'Gy lobe: phase encoding (between the pulses here)', 'Gx: frequency encoding while the ADC is open', '', 'One cycle gives ONE row of k-space.', '128 or 256 cycles are needed for a full image.'], size=8.6, dy=0.105)
    app.decode(0 if 0.39 < ph < 0.59 else None, 'readout: the ADC samples one line while Gx is on'); app.mark('B', 'C', 'D')

@step(9, 'Turbo spin echo: an echo train', 'Instead of waiting for TR, apply several 180 degree pulses after one 90. Each makes an echo, and each echo is given a different Gy lobe, so one TR fills several rows. With four echoes the scan is four times faster. The echoes shrink because T2 losses accumulate.',
      dur=26, cam='scanner', show=dict(SCAN_SHOW), table=0.0)
def s9_3(app, t, u, S):
    ph = (u * 1.6) % 1.0; d = seq_step_common(app, S, 'tse', 'TSE')
    if d:
        tt = np.linspace(0, 1, 800); sig = np.zeros_like(tt)
        for k in range(4): sig += np.exp(-((tt - (0.18 + 0.12 * k)) / 0.025) ** 2) * (0.9 * np.exp(-k * 0.35)) * np.cos(2 * np.pi * 30 * tt)
        d['a'].plot(tt, sig, color=COL['rx'], lw=0.9); d['a'].plot(np.linspace(0.18, 0.54, 50), 0.9 * np.exp(-np.linspace(0, 3, 50) * 0.35 / 1.0 * 1.0 * 0.12 / 0.12 * 0.33), '--', color='#fbbf24', lw=1); sty(d['a'], 'Echo train: each echo is weaker (T2 envelope)', None, None); d['a'].set_xlim(0, 1); d['a'].set_xticks([]); S['c1'] = d['a'].axvline(0, color='#fbbf24')
        c = app.axC('text'); app.put(c['a'], [('Turbo (fast) spin echo', FG), 'Echo train length (turbo factor) = 4 here', 'Rows per TR = 4  ->  scan time / 4', 'Used in most modern clinical protocols.', '', 'Each echo has a different TE, so the effective TE', 'is set by which echo fills the center of k-space.', 'Gy lobes rewind after each readout to keep echoes alive.'], size=8.6, dy=0.11)
    S['cur'].set_xdata([ph, ph]); S['c1'].set_xdata([ph, ph]); glow(app, 'tse', ph); app.decode(2 if ph > 0.4 else None, 'four lines per TR enter k-space, each at a different Gy'); app.mark('B', 'C', 'D')

@step(9, 'HASTE: one shot, half the data', 'The ultrafast spin echo keeps the echoes alive by choosing the Gy lobes so that dephasing is cancelled, and it collects all rows in one TR. HASTE also acquires just over half of k-space. Because k-space is symmetric, the computer fills in the other half numerically.',
      dur=26, cam='scanner', show=dict(SCAN_SHOW), table=0.0)
def s9_4(app, t, u, S):
    if not S:
        d = app.axB('lr'); S['k'] = d['a'].imshow(np.zeros((N64, N64)), cmap='magma', vmin=0, vmax=1, origin='lower', extent=(-32, 32, -32, 32)); sty(d['a'], 'Acquired k-space (rows)', 'kx', 'ky')
        S['k2'] = d['b'].imshow(np.zeros((N64, N64)), cmap='magma', vmin=0, vmax=1, origin='lower', extent=(-32, 32, -32, 32)); sty(d['b'], 'After conjugate fill', None, None); d['b'].set_xticks([]); d['b'].set_yticks([])
        c = app.axC('img_txt'); S['im'] = c['a'].imshow(np.zeros((N64, N64)), cmap='gray', vmin=0, vmax=1); c['a'].set_xticks([]); c['a'].set_yticks([]); sty(c['a'], 'Image', None, None); S['t'] = c['t']
        app.put(c['t'], [('Half-Fourier', FG), 'k(-kx,-ky) = conj k(kx,ky) for a real image.', 'So the missing half can be calculated.', 'Real scans acquire a bit over half (128+) to', 'allow phase correction. Cost: less signal averaging.'], size=8.4, dy=0.12)
    n_rows = int(N64 * 0.56 * u); mask = np.zeros((N64, N64)); mask[N64 // 2 - n_rows // 2 + 4: N64 // 2 + n_rows // 2 + 4, :] = 0
    rows = np.arange(N64 // 2 - 4, N64 // 2 - 4 + int(N64 * 0.56 * u)); mask[np.clip(rows, 0, N64 - 1), :] = 1
    Kp = K64 * mask; S['k'].set_data(klog(Kp, mask)); full = mask.copy()
    if u > 0.9:
        Kf = Kp.copy()
        for r in range(N64):
            for c_ in range(N64):
                if mask[r, c_] == 0 and mask[(N64 - r) % N64, (N64 - c_) % N64] == 1: Kf[r, c_] = np.conj(Kp[(N64 - r) % N64, (N64 - c_) % N64]); full[r, c_] = 1
        S['k2'].set_data(klog(Kf, full)); S['im'].set_data(from_kspace(Kf) / max(from_kspace(K64).max(), 1e-9))
    else:
        S['k2'].set_data(klog(Kp, mask)); S['im'].set_data(from_kspace(Kp) / max(from_kspace(K64).max(), 1e-9))
    ph = (u * 3) % 1.0; glow(app, 'tse', ph); app.decode(2 if u < 0.9 else 3, None); app.mark('B', 'C', 'D')

@step(9, 'Gradient echo: reverse the gradient instead', 'A gradient makes spins dephase, but reversing it makes them rephase. The negative lobe dephases the signal, then the positive readout lobe rewinds it into an echo, with no 180 degree pulse. It is fast, but field imperfections are not corrected, so the signal decays with T2*.',
      dur=26, cam='head', show={'patient': 0.2, 'table': 0.3, 'gx': 0.8, 'bore': 0.04}, uses=('spins', 'probex'), table=0.0)
def s9_5(app, t, u, S):
    ph = u; d = None
    if not S:
        d = app.axB('seq1'); S['cur'] = draw_seq(d['s'], 'gre'); tt = np.linspace(0, 1, 600); sig = np.exp(-((tt - 0.43) / 0.07) ** 2) * np.cos(2 * np.pi * 30 * tt) * 0.9; d['a'].plot(tt, sig, color=COL['rx'], lw=0.9); d['a'].set_xlim(0, 1); d['a'].set_xticks([]); sty(d['a'], 'Gradient echo: peak where the lobes cancel', None, None); S['c1'] = d['a'].axvline(0, color='#fbbf24')
        c = app.axC('text'); app.put(c['a'], [('Gradient recalled echo (GRE)', FG), 'Negative lobe: columns dephase. Positive lobe: they rewind.', 'Echo peaks when the areas of the two lobes are equal.', 'No 180 pulse, so shorter TR and less heating.', '', 'But static field errors are NOT undone, so', 'the signal follows T2*, not T2.'], size=8.6, dy=0.12)
    S['cur'].set_xdata([u, u]); S['c1'].set_xdata([u, u]); gx_net = 0.0
    if u < 0.14: g = 0.0
    elif u < 0.26: g = -1.0
    else: g = 1.0
    S['phi'] = S.get('phi', np.zeros(NSP)); S['phi'] += dtc(app, S) * 0 + 0.0
    xs = hp(app)[0] * 0 + app.get('spins').rel[:, 0]; cum = np.interp(u, [0, 0.14, 0.26, 0.60, 1.0], [0, 0, -0.12 * 1, -0.12 + 0.34, -0.12 + 0.34]) * 1.0
    ph_ = xs / 0.058 * 14.0 * cum * (-1 if False else 1); mxy = np.exp(-u * 1.3)
    app.get('spins').set(v_iso(0.0, mxy, OM * app.clock + ph_), phase_rgb(ph_ % (2 * np.pi))); app.get('probex').set(np.linspace(-1, 1, 31) * g, -1, 1, 'coolwarm')
    app.w.target_op['gx'] = 1.0 if g != 0 else 0.3; app.decode(0 if 0.26 < u < 0.6 else None, 'the readout samples the echo as the lobes cancel'); app.mark('B', 'C', 'D')

@step(9, 'Echo-planar imaging: one shot, whole image', 'Rapidly reverse the readout gradient to make a train of gradient echoes inside one spin echo, and add a small Gy blip between echoes. Each reversal reads one row of k-space, in a zig-zag path. The whole slice is collected in tens of milliseconds, and the coils switch so fast they are very loud.',
      dur=28, cam='scanner', show=dict(SCAN_SHOW), table=0.0)
def s9_6(app, t, u, S):
    if not S:
        d = app.axB('seq2'); S['cur'] = draw_seq(d['s'], 'epi'); S['ka'] = d['a']; sty(d['a'], 'k-space path (zig-zag)', 'kx', 'ky'); d['a'].set_xlim(-1.1, 1.1); d['a'].set_ylim(-1.1, 1.1)
        ys = np.linspace(-1, 1, 8); path = []
        for k, y in enumerate(ys): path += [(-1, y), (1, y)] if k % 2 == 0 else [(1, y), (-1, y)]
        path = np.array(path); d['a'].plot(path[:, 0], path[:, 1], color='#334155', lw=1); S['path'] = path; S['pl'], = d['a'].plot([], [], color='#fbbf24', lw=2)
        d['b'].axis('off'); c = app.axC('img_txt'); S['im'] = c['a'].imshow(np.zeros((N64, N64)), cmap='gray', vmin=0, vmax=1); c['a'].set_xticks([]); c['a'].set_yticks([]); sty(c['a'], 'Image forms as rows arrive', None, None)
        app.put(c['t'], [('Echo-planar imaging', FG), 'Readout reverses 64+ times per shot.', 'Gy blips step through the rows.', 'Whole slice in about 50 ms.', '', 'Used for diffusion and fMRI.', 'Drawback: distortion near air.'], size=8.6, dy=0.12)
    ph = u; S['cur'].set_xdata([ph, ph]); n = int(len(S['path']) * ph); S['pl'].set_data(S['path'][:n, 0], S['path'][:n, 1])
    mask = np.zeros((N64, N64)); mask[: int(N64 * np.clip((ph - 0.25) / 0.7, 0, 1)), :] = 1; mask = np.roll(mask, N64 // 2 - int(N64 * np.clip((ph - 0.25) / 0.7, 0, 1)) // 2, axis=0)
    S['im'].set_data(from_kspace(K64 * mask) / from_kspace(K64).max()); glow(app, 'epi', ph); app.decode(2 if ph < 0.9 else 3); app.mark('B', 'C', 'D')
# ----------------------------------------------------------------------------------------------
# PHASE 10  diffusion
# ----------------------------------------------------------------------------------------------
phase_info(10, 'Diffusion imaging', 'Water molecules move randomly. Two matched gradient pulses around a 180 degree pulse cancel the phase of molecules that stay put but leave moving molecules with a net phase error, so the signal drops where diffusion is fast. The size of the drop gives the apparent diffusion coefficient (ADC).')
DT = [('CSF', '#6cc4ff', 3.0e-3), ('GM', '#b0a0ff', 0.9e-3), ('WM', '#ffd27a', 0.7e-3), ('Lesion (acute stroke)', '#f87171', 0.4e-3)]

@step(10, 'Random motion and the diffusion gradient pair', 'A first gradient pulse writes a phase that depends on position. The 180 degree pulse reverses it. A second identical pulse then removes the phase for every molecule that has not moved. Molecules that diffused to a new place keep a leftover phase, so they dephase and the signal falls.',
      dur=28, cam='head', show={'patient': 0.2, 'table': 0.3, 'gx': 0.5, 'gy': 0.5, 'gz': 0.5, 'bore': 0.04}, uses=('spins',), table=0.0)
def s10_1(app, t, u, S):
    D = 3.0 if u < 0.5 else 0.4; pk = np.exp(-0.0 + 0)
    if not S:
        d = app.axB('two'); S['kd'] = d['a']; tt = np.linspace(0, 1, 200)
        g = np.where((tt > .15) & (tt < .3), 1, 0) + np.where((tt > .55) & (tt < .7), 1, 0)
        d['a'].fill_between(tt, 0, g, color=COL['gx'], alpha=0.7, step='mid'); d['a'].plot([0.4, 0.4], [0, 1.3], color='#fb923c', lw=3); d['a'].text(0.405, 1.15, '180', color='#fb923c', fontsize=8)
        S['c1'] = d['a'].axvline(0, color='#fbbf24'); d['a'].set_ylim(0, 1.5); sty(d['a'], 'Two matched diffusion gradient pulses around a 180', 'time', 'G'); d['a'].set_xticks([])
        S['ln'], = d['b'].plot([], [], color=COL['rx'], lw=1.8); d['b'].set_xlim(0, 1); d['b'].set_ylim(0, 1.05); sty(d['b'], 'Net signal', 'time', '|Mxy|'); S['x'] = []; S['y'] = []
        c = app.axC('text'); S['ct'] = c['a']
    S['c1'].set_xdata([u, u]); loc = u % 0.5 / 0.5
    kick = np.where(loc < 0.3, 0.0, 1.0) * (OFFR * 0 + 1)
    spread = float(np.clip((loc - 0.3) / 0.6, 0, 1)) * (D / 3.0) * 3.0 * OFFR
    ph = OM * app.clock + spread; mxy = np.exp(-0.3 * loc)
    v = v_iso(0, mxy, ph); app.get('spins').set(v, phase_rgb(spread % (2 * np.pi))); M = v.mean(0); S['x'].append(u); S['y'].append(np.hypot(M[1], M[2])); S['ln'].set_data(S['x'], S['y'])
    S['ct'].clear(); app.put(S['ct'], [('Fast diffusion' if D > 1 else 'Restricted diffusion', '#fbbf24'), 'Phase error ~ displacement x gradient area.', 'Fast mover (CSF): large error, dark in the image.', 'Restricted mover (acute stroke): small error, bright.', '', 'Spin-echo EPI is used, so the scan is one shot.'], size=8.8, dy=0.12)
    app.mark('B', 'C')

@step(10, 'b-value and the ADC', 'The attenuation follows S = S0 exp(-b ADC). The b-value combines gradient strength, duration and spacing. Two images, at b = 0 and a high b (for example 1000 s/mm2), give the ADC at each voxel: ADC = ln(S0 / S) / b.',
      dur=24, cam='head', show={'patient': 0.2, 'table': 0.3}, uses=('arr4',), table=0.0)
def s10_2(app, t, u, S):
    b = 1500 * u
    if not S:
        d = app.axB('one'); bb = np.linspace(0, 1500, 200)
        for n_, c_, D_ in DT: d['a'].plot(bb, np.exp(-bb * D_), color=c_, lw=1.8, label='%s  ADC = %.1f e-3 mm2/s' % (n_, D_ * 1e3))
        d['a'].legend(facecolor=PANEL, edgecolor='#334155', labelcolor=FG, fontsize=7.2); S['cur'] = d['a'].axvline(0, color='#fbbf24'); sty(d['a'], 'Signal falls with b-value: S = S0 exp(-b ADC)', 'b (s/mm2)', 'S / S0')
        c = app.axC('lr'); c['a'].axis('off'); S['sq'] = [c['a'].add_patch(Rectangle((k + 0.05, 0.2), 0.9, 0.9, fc='#000', ec='#475569')) for k in range(4)]; c['a'].set_xlim(0, 4); c['a'].set_ylim(0, 1.4)
        for k, (n_, *_r) in enumerate(DT): c['a'].text(k + 0.5, 0.05, n_.split(' ')[0], color=FG, ha='center', fontsize=8)
        S['t'] = c['b']; c['b'].axis('off')
    S['cur'].set_xdata([b, b]); vals = [np.exp(-b * D_) for n_, c_, D_ in DT]
    for k, v_ in enumerate(vals): S['sq'][k].set_facecolor((v_,) * 3)
    S['t'].clear(); S['t'].axis('off'); app.put(S['t'], [('b = %d s/mm2' % b, '#fbbf24'), 'Free water: signal almost gone.', 'Acute stroke: restricted, stays bright.', '', 'ADC map = ln(S0 / S) / b', 'removes the T2 contribution.'], size=8.6, dy=0.12)
    ph = OM * app.clock; GR = np.array([hp(app, -0.035, 0.035, 0), hp(app, -0.035, -0.035, 0), hp(app, 0.035, 0.035, 0), hp(app, 0.035, -0.035, 0)])
    app.get('arr4').set(GR, np.array([[0, v_ * np.cos(ph), v_ * np.sin(ph)] for v_ in vals]), np.array([hex2rgb(c_) for n_, c_, D_ in DT]), L=0.06); app.mark('B', 'C')

@step(10, 'Diffusion tensor imaging: fibre direction', 'In white matter water moves easily along axons and poorly across them. Applying the diffusion gradient in at least six directions gives a tensor at each voxel, drawn as an ellipsoid. Its long axis is the fibre direction; colour codes it (red left-right, green front-back, blue up-down).',
      dur=24, cam='head', show={'patient': 0.12, 'table': 0.2}, uses=('fib',), table=0.0)
def s10_3(app, t, u, S):
    app.get('fib').show(True)
    if not S:
        d = app.axB('one'); a = d['a']; a.set_aspect('equal'); a.set_xlim(-1.3, 1.3); a.set_ylim(-1.3, 1.3); a.add_patch(Circle((0, 0), 0.4, fc='#334155', ec='none')); a.set_xticks([]); a.set_yticks([])
        th = np.linspace(0, 2 * np.pi, 100); a.plot(1.0 * np.cos(th), 0.3 * np.sin(th), color='#fcd34d', lw=2); S['dir'], = a.plot([], [], color='#fbbf24', lw=3); sty(a, 'Diffusion ellipsoid: long along the fibre', None, None)
        c = app.axC('text'); app.put(c['a'], [('Tensor and anisotropy', FG), 'Three eigenvalues l1 >= l2 >= l3.', 'Isotropic (CSF): l1 = l2 = l3.', 'Fibre (WM): l1 much larger than l2, l3.', '', 'FA runs from 0 (round) to 1 (a line).', 'Tractography follows the long axis voxel to voxel.'], size=8.8, dy=0.12)
    a_ = 2 * np.pi * u * 2; S['dir'].set_data([-np.cos(a_), np.cos(a_)], [-np.sin(a_), np.sin(a_)]); app.mark('B', 'C')

# ----------------------------------------------------------------------------------------------
# PHASE 11  Fourier series and the MRI equation
# ----------------------------------------------------------------------------------------------
phase_info(11, 'Fourier analysis and the MRI equation', 'Any signal can be built from sinusoids. The signal in the coil, recorded over time, is a sum of sinusoids, one per spatial position; the MRI equation says the coil signal at gradient history k is the image integrated against exp(-2 pi i k . r). The Fourier transform inverts it.')

@step(11, 'A signal is a sum of sinusoids', 'Fourier showed that any waveform is a sum of sine waves with the right amplitudes and phases. Adding more terms makes the sum match a sharp-edged waveform better. The Fourier transform lists the amplitudes: the same information, shown by frequency instead of time.',
      dur=26, cam='scanner', show=dict(SCAN_SHOW), table=0.0)
def s11_1(app, t, u, S):
    nterm = 1 + int(11 * u); tt = np.linspace(0, 2, 600)
    if not S:
        d = app.axB('two'); S['t'] = tt; sq = np.sign(np.sin(np.pi * tt)); d['a'].plot(tt, sq, color='#475569', lw=1); S['l'], = d['a'].plot(tt, tt * 0, color='#fbbf24', lw=1.6); sty(d['a'], 'Square wave built from sine terms', 'time', None)
        S['b'] = d['b'].bar([1], [1], color='#38bdf8'); d['b'].set_xlim(0, 25); d['b'].set_ylim(0, 1.3); sty(d['b'], 'Amplitude of each sinusoid', 'harmonic', 'amplitude'); S['ct'] = app.axC('text')['a']; S['bars'] = d['b']
    s = sum(4 / np.pi / (2 * k + 1) * np.sin((2 * k + 1) * np.pi * tt) for k in range(nterm)); S['l'].set_ydata(s)
    S['bars'].cla(); S['bars'].bar([2 * k + 1 for k in range(nterm)], [4 / np.pi / (2 * k + 1) for k in range(nterm)], color='#38bdf8'); S['bars'].set_xlim(0, 25); S['bars'].set_ylim(0, 1.4); sty(S['bars'], 'Amplitude of each sinusoid', 'harmonic', 'amplitude')
    S['ct'].clear(); app.put(S['ct'], [('Terms used: %d' % nterm, '#fbbf24'), 'square wave = (4/pi) [sin x + sin 3x / 3 + sin 5x / 5 + ...]', '', 'MRI uses the same idea with complex exponentials', 'exp(i 2 pi f t): amplitude and phase in one number.'], size=8.8, dy=0.12)
    app.mark('B', 'C')

@step(11, 'The MRI equation', 'The coil signal is S(k) = integral of rho(r) exp(-i 2 pi k . r) dr, where rho is the image and k is the accumulated area of the gradients (in cycles per metre). Each sample is the image multiplied by a wave of spatial frequency k and summed. The image is the inverse Fourier transform of S.',
      dur=28, cam='head', show={'patient': 0.12, 'table': 0.2, 'headcoil': 0.3}, uses=('wave',), table=0.0)
def s11_2(app, t, u, S):
    kx = 3.0 * np.sin(2 * np.pi * u); ky = 2.0 * np.sin(2 * np.pi * u * 1.3 + 1); app.get('wave').show(True); app.get('wave').set(kx, ky, 0.0)
    if not S:
        d = app.axB('lr'); S['wv'] = d['a'].imshow(np.zeros((64, 64)), cmap='gray', vmin=-1, vmax=1, origin='lower'); sty(d['a'], 'Wave pattern for this k', None, None); d['a'].set_xticks([]); d['a'].set_yticks([])
        S['pr'] = d['b'].imshow(IMG64, cmap='gray', origin='lower'); sty(d['b'], 'Image (rho)', None, None); d['b'].set_xticks([]); d['b'].set_yticks([]); S['ct'] = app.axC('text')['a']
    X, Y = np.meshgrid(np.linspace(0, 1, 64), np.linspace(0, 1, 64)); w = np.cos(2 * np.pi * (kx * X + ky * Y)); S['wv'].set_data(w)
    val = (IMG64 * np.exp(-2j * np.pi * (kx * X + ky * Y))).sum()
    S['ct'].clear(); app.put(S['ct'], [('S(k) = sum of rho(r) exp(-i 2 pi k.r)', FG), 'kx = %.2f, ky = %.2f cycles per image' % (kx, ky), 'S = %.0f %+.0fi' % (val.real, val.imag), '', 'Low k: broad shapes and overall brightness.', 'High k: fine detail and edges.'], size=8.8, dy=0.12)
    app.decode(2, None); app.mark('B', 'C', 'D')

# ----------------------------------------------------------------------------------------------
# PHASE 12  k-space
# ----------------------------------------------------------------------------------------------
phase_info(12, 'k-space', 'k-space holds the raw data: one complex number per spatial frequency. Its center holds contrast and overall signal; its edges hold detail. Undersampling k-space shortens the scan but causes blur or aliasing.')

@step(12, 'k-space is filled one line at a time', 'Each readout fills a horizontal line; the phase-encode strength picks which line. Bright values sit near the centre where all spins are in phase. The computer keeps the complex values, so every point has an amplitude and a phase.',
      dur=26, cam='scanner', show=dict(SCAN_SHOW), table=0.0)
def s12_1(app, t, u, S):
    if not S:
        d = app.axB('lr'); S['k'] = d['a'].imshow(np.zeros((N64, N64)), cmap='magma', vmin=0, vmax=1, origin='lower', extent=(-32, 32, -32, 32)); sty(d['a'], 'k-space magnitude (log)', 'kx', 'ky')
        S['s'] = d['b']; d['b'].set_xlim(0, 1); d['b'].set_ylim(-1.2, 1.2); sty(d['b'], 'Line being read (real part)', 'sample', None)
        S['ln'], = d['b'].plot([], [], color=COL['rx'], lw=1); c = app.axC('img_txt'); S['im'] = c['a'].imshow(np.zeros((N64, N64)), cmap='gray', vmin=0, vmax=1); c['a'].set_xticks([]); c['a'].set_yticks([]); sty(c['a'], 'Image from lines so far', None, None)
        S['t'] = c['t']
    n = max(1, int(N64 * u)); order = np.argsort(np.abs(np.arange(N64) - N64 // 2), kind='stable') if False else np.arange(N64); mask = np.zeros((N64, N64)); mask[:n, :] = 1
    S['k'].set_data(klog(K64, mask)); S['im'].set_data(from_kspace(K64 * mask) / from_kspace(K64).max()); line = K64[n - 1].real; S['ln'].set_data(np.linspace(0, 1, N64), line / (np.abs(K64.real).max() + 1e-9))
    ph = (u * N64) % 1.0; glow(app, 'basic', 0.4 + 0.3 * ph)
    S['t'].clear(); app.put(S['t'], [('Line %d of %d' % (n, N64), '#fbbf24'), 'Real scanner: 128 to 256 lines.'], size=8.6, dy=0.12); app.decode(2, 'each readout = one complete line of k-space'); app.mark('B', 'C', 'D')

@step(12, 'Centre for contrast, edges for detail', 'Keep only a disk of low-frequency points and the image keeps its contrast but loses sharp edges. Keep only a ring of high frequencies and the image shows only outlines. Skipping every second line halves the time but folds the image over itself (aliasing).',
      dur=30, cam='scanner', show=dict(SCAN_SHOW), table=0.0)
def s12_2(app, t, u, S):
    Y, X = np.mgrid[-32:32, -32:32]; R = np.hypot(X, Y); mode = min(int(u * 4), 3)
    if mode == 0: mask = (R < 6); nm = 'center only (R < 6): contrast, no detail'
    elif mode == 1: mask = (R > 6); nm = 'edges only (R > 6): outlines'
    elif mode == 2: mask = np.ones((N64, N64)); nm = 'full k-space'
    else: mask = np.zeros((N64, N64)); mask[::2, :] = 1; nm = 'every second line: aliasing'
    if not S:
        d = app.axB('lr'); S['k'] = d['a'].imshow(np.zeros((N64, N64)), cmap='magma', vmin=0, vmax=1, origin='lower'); sty(d['a'], 'k-space kept', None, None); d['a'].set_xticks([]); d['a'].set_yticks([])
        S['m'] = d['b']; d['b'].axis('off'); c = app.axC('img_txt'); S['im'] = c['a'].imshow(np.zeros((N64, N64)), cmap='gray', vmin=0, vmax=1); c['a'].set_xticks([]); c['a'].set_yticks([]); sty(c['a'], 'Reconstructed image', None, None); S['t'] = c['t']
    kk = klog(K64, mask); S['k'].set_data(kk)
    im = from_kspace(K64 * mask); S['im'].set_data(np.clip(im / from_kspace(K64).max(), 0, 1))
    S['t'].clear(); app.put(S['t'], [(nm, '#fbbf24'), 'Undersampling trades time for artifacts.', 'Parallel imaging and compressed', 'sensing recover the missing lines.'], size=8.6, dy=0.12); app.decode(3, None); app.mark('B', 'C', 'D')

# ----------------------------------------------------------------------------------------------
# PHASE 13  k-space to image: the full decode
# ----------------------------------------------------------------------------------------------
phase_info(13, 'From k-space to the final image', 'The computer takes the complex k-space matrix, applies an inverse 2D Fourier transform, and takes the magnitude of each complex pixel. Each pixel value is then the signal of one voxel, shown as a gray level.')

@step(13, 'Inverse 2D Fourier transform, pixel by pixel', 'Every k-space point contributes one wave across the whole image. Adding them one by one, from the centre outward, builds the picture: first a blur, then edges, then fine detail. The final value at each pixel is the sum of all waves at that position.',
      dur=32, cam='scanner', show=dict(SCAN_SHOW), table=0.0)
def s13_1(app, t, u, S):
    if not S:
        d = app.axB('lr'); S['k'] = d['a'].imshow(np.zeros((N64, N64)), cmap='magma', vmin=0, vmax=1, origin='lower'); sty(d['a'], 'k-space points used', None, None); d['a'].set_xticks([]); d['a'].set_yticks([])
        S['w'] = d['b'].imshow(np.zeros((N64, N64)), cmap='gray', vmin=-1, vmax=1, origin='lower'); sty(d['b'], 'Wave added now', None, None); d['b'].set_xticks([]); d['b'].set_yticks([])
        c = app.axC('img_txt'); S['im'] = c['a'].imshow(np.zeros((N64, N64)), cmap='gray', vmin=0, vmax=1); c['a'].set_xticks([]); c['a'].set_yticks([]); sty(c['a'], 'Image so far', None, None); S['t'] = c['t']
        Y, X = np.mgrid[-32:32, -32:32]; S['R'] = np.hypot(X, Y)
    Rmax = 1 + 45 * u; mask = (S['R'] <= Rmax).astype(float); Ks = K64 * mask
    S['k'].set_data(klog(K64, mask)); S['im'].set_data(np.clip(from_kspace(K64 * mask) / from_kspace(K64).max(), 0, 1))
    k = int(Rmax); ang = 2 * np.pi * u * 6; xx, yy = np.meshgrid(np.linspace(0, 1, 64), np.linspace(0, 1, 64)); S['w'].set_data(np.cos(2 * np.pi * k * (np.cos(ang) * xx + np.sin(ang) * yy) / 4.0))
    g = app.get('wave'); g.show(True); g.set(k * np.cos(ang) / 8.0, k * np.sin(ang) / 8.0, 0.0)
    S['t'].clear(); app.put(S['t'], [('Radius in k-space: %d of 32' % min(32, k), '#fbbf24'), 'Each point = one wave.', 'Sum of all waves = image.'], size=8.6, dy=0.12); app.decode(3, 'pixel(x, y) = sum over k of S(k) exp(+i 2 pi k.r)'); app.mark('B', 'C', 'D')

@step(13, 'Magnitude and gray level: the pixel meets the voxel', 'The transform gives a complex number at each pixel. The magnitude becomes the gray level. In the patient this number is the signal of one voxel: the T2 curve amplitude at TE for that box of tissue. Rotate the 3D view and you are looking at the box that made that pixel.',
      dur=30, cam='head', show={'patient': 0.25, 'table': 0.3, 'headcoil': 0.3}, uses=('vox',), table=0.0)
def s13_2(app, t, u, S):
    n = 16
    if getattr(app, '_vq', None) != (app.i, 1): app._vq = (app.i, 1); app.get('vox').build(n, 0.006, 0.17)
    app.get('vox').show(True)
    if not S:
        c = app.axC('img_txt'); S['im'] = c['a'].imshow(IMG64[::4, ::4], cmap='gray', vmin=0, vmax=IMG64.max(), interpolation='nearest'); c['a'].set_xticks([]); c['a'].set_yticks([]); sty(c['a'], '16 x 16 image', None, None); S['sel'] = c['a'].add_patch(Rectangle((0, 0), 1, 1, fc='none', ec='#fbbf24', lw=2)); S['t'] = c['t']
        d = app.axB('one'); S['d'] = d['a']; S['cur'] = None
    i = int(u * 255) // 16; j = int(u * 255) % 16; S['sel'].set_xy((j - 0.5, i - 0.5)); val = IMG64[i * 4, j * 4]
    S['d'].clear(); tt = np.linspace(0, 200, 100); S['d'].plot(tt, val * np.exp(-tt / 80.0), color=COL['rx'], lw=1.8); S['d'].axvline(20, color='#fbbf24'); sty(S['d'], 'T2 curve of the selected voxel; TE = 20 ms read here', 'time (ms)', 'signal')
    S['t'].clear(); app.put(S['t'], [('Pixel (%d, %d)' % (i, j), '#fbbf24'), 'gray level = |S| = %.2f' % val, 'voxel 1 x 1 x 5 mm', 'in the head'], size=8.6, dy=0.12)
    app.decode(5, 'gray level = signal of that voxel at TE'); app.mark('B', 'C', 'D')
# ----------------------------------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------------------------------
def run_export(outdir, only=None, auto=False):
    import PIL.Image as I
    os.makedirs(outdir, exist_ok=True); pv.OFF_SCREEN = True
    p = pv.Plotter(off_screen=True, window_size=(1000, 780)); p.set_background(BG)
    app = App(p, headless=True, auto=auto); p.show(auto_close=False)
    cnt = {}
    for i, s in enumerate(STEPS):
        cnt[s['phase']] = cnt.get(s['phase'], 0) + 1; idn = '%d.%d' % (s['phase'], cnt[s['phase']])
        if only and idn != only: continue
        app.go(i, snap=True); app.state = 'play'; app.t = 0
        for frac in (0.5, 0.97):
            app.t = 0
            for _ in range(int(frac * s['dur'] * 10)): app.update(0.1)
            I.fromarray(app.snapshot()).save(os.path.join(outdir, 'step_%s_%d.png' % (idn, int(frac > 0.9))))
        print('wrote', idn, s['title'])

def run_qt(auto=False):
    from PyQt5 import QtWidgets, QtCore
    from pyvistaqt import QtInteractor
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    qa = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = QtWidgets.QMainWindow(); win.setWindowTitle('MRI physics, step by step'); win.resize(1700, 980)
    win.setStyleSheet('background:#0b0f14; color:#e2e8f0;')
    root = QtWidgets.QWidget(); win.setCentralWidget(root); outer = QtWidgets.QVBoxLayout(root); outer.setContentsMargins(4, 4, 4, 4)
    top = QtWidgets.QHBoxLayout(); outer.addLayout(top, 1)
    inter = QtInteractor(root); inter.set_background(BG); top.addWidget(inter.interactor, 5)
    right = QtWidgets.QVBoxLayout(); top.addLayout(right, 4)
    figs = (Figure(figsize=(6.4, 3.6), dpi=100), Figure(figsize=(6.4, 0.95), dpi=100), Figure(figsize=(6.4, 3.4), dpi=100)); cans = []
    for f, st in zip(figs, (36, 9, 34)):
        cv = FigureCanvasQTAgg(f); cv.setMinimumHeight(60); right.addWidget(cv, st); cans.append(cv)
    head = QtWidgets.QLabel(); head.setStyleSheet('color:#fbbf24; font-size:18px; font-weight:bold; padding:4px 8px;')
    body = QtWidgets.QLabel(); body.setWordWrap(True); body.setStyleSheet('font-size:15px; padding:0 8px 4px 8px;'); body.setMinimumHeight(70)
    stat = QtWidgets.QLabel(); stat.setStyleSheet('color:#94a3b8; font-size:12px; padding:2px 8px;')
    outer.addWidget(head); outer.addWidget(body); outer.addWidget(stat)
    app = App(inter, figs=figs, auto=auto); app.go(0)
    last = [time.time()]; frame = [0]
    def tick():
        now = time.time(); dt = min(now - last[0], 0.1); last[0] = now
        app.update(dt); frame[0] += 1
        h, b = app.caption(); head.setText(h); body.setText(b); stat.setText(app.status() + '     keys: Space pause/play, Right next, Left back, Shift+Right/Left next/previous phase, R restart step, 1-4 camera, Home whole machine')
        if frame[0] % 2 == 0:
            for c in cans: c.draw_idle()
        inter.render()
    timer = QtCore.QTimer(); timer.timeout.connect(tick); timer.start(33)
    def key(e):
        k = e.key(); sh = bool(e.modifiers() & QtCore.Qt.ShiftModifier); Q = QtCore.Qt
        if k == Q.Key_Space: app.toggle()
        elif k == Q.Key_Right: app.next_phase() if sh else app.next_step()
        elif k == Q.Key_Left: app.prev_phase() if sh else app.prev_step()
        elif k == Q.Key_R: app.go(app.i)
        elif k in (Q.Key_Home, Q.Key_1): app.cam_to('whole')
        elif k == Q.Key_2: app.cam_to('patient')
        elif k == Q.Key_3: app.cam_to('head')
        elif k == Q.Key_4: app.cam_to('voxels')
        elif k == Q.Key_Escape: win.close()
    win.keyPressEvent = key
    inter.interactor.keyPressEvent = key
    win.show(); qa.exec_()

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='MRI physics 3D tutorial')
    ap.add_argument('--auto', action='store_true', help='do not pause between phases')
    ap.add_argument('--export', metavar='DIR', help='headless: write PNG frames')
    ap.add_argument('--only', help='with --export, one step id such as 6.4')
    a = ap.parse_args()
    if a.export: run_export(a.export, a.only, a.auto)
    else: run_qt(a.auto)
