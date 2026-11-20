"""
electron_em_fields_3d.py

3D simulation of ONE electron in a vacuum (MCAT electromagnetism).
You can switch on an external electric field E (points along +x)
and an external magnetic field B (points along +z).

Blue = electric field E, purple = magnetic field B,
green arrow = velocity v, red arrow = force F on the electron.
Thin blue lines with a dot on the end = the electron's own E field
(the dot is the end the field points TOWARD, into the negative charge).

The electron is NEGATIVE, so the force is  F = -e * (E + v x B).
That means F points OPPOSITE to E.

Units are made-up simple units (charge/mass = 1) so the numbers stay small.

How to use:
  1. Change the numbers in the SETTINGS block below.
  2. Run the file. A 3D window opens and the electron starts moving.
  3. Drag with the mouse to rotate the 3D view.
  4. Close the window to stop.

Why this version is faster:
  The old version erased and redrew EVERYTHING every frame (slow).
  This version draws things once and only moves the lines that change.
  It also hides the 3D grid and axis numbers, which were very slow to draw.

New things (not in the course modules):
  - plt.pause()  : shows the plot and waits a moment, used for animation
  - ax.quiver()  : draws arrows in the 3D plot
  - np.cross()   : cross product of two vectors (v x B)
  - set_data_3d(): moves a line that is already drawn
  - .remove()    : deletes something that was drawn
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------- SETTINGS (change these) ----------------

# Strength of the external electric field (0 = off)
E_STRENGTH = 1.0

# Strength of the external magnetic field (0 = off)
B_STRENGTH = 1.5

# Starting speed of the electron along +y (sideways to B)
LAUNCH_SPEED_PERP = 0.0

# Starting speed of the electron along +z (along B, makes a spiral)
LAUNCH_SPEED_PARA = 0.0

# Slow motion: 1.0 = normal speed, 0.1 = ten times slower
SLOW_MO = 0.3

# Show or hide things (True = show, False = hide)
SHOW_E_LINES = True
SHOW_B_LINES = True
SHOW_VECTORS = True
SHOW_TRAIL = True

# How many frames to run before stopping
N_FRAMES = 5000

# ---------------- SETUP ----------------

# charge / mass of the electron (negative because electron is negative)
QM = -1.0

# external fields as vectors [x, y, z]
E_FIELD = np.array([E_STRENGTH, 0.0, 0.0])
B_FIELD = np.array([0.0, 0.0, B_STRENGTH])

# starting position and velocity
pos = np.array([0.0, 0.0, 0.0])
vel = np.array([0.0, LAUNCH_SPEED_PERP, LAUNCH_SPEED_PARA])

# the time that passes in one animation frame
DT_PER_FRAME = 0.05 * SLOW_MO

# each frame is split into this many tiny steps (keeps the physics accurate)
SUB_STEPS = 10

# lists that remember where the electron has been (for the trail)
trail_x = []
trail_y = []
trail_z = []

time_passed = 0.0

# ---------------- FUNCTIONS ----------------


# length of a vector
def size(v):
    return np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


# force on the electron:  F = q * (E + v x B)
def get_force(v):
    force = QM * (E_FIELD + np.cross(v, B_FIELD))
    return force


# move the electron one tiny step (Boris method, a stable way to do it)
def move_one_step(p, v, dt):
    # first half push from the electric field
    half_push = QM * E_FIELD * dt / 2
    v_minus = v + half_push

    # rotate the velocity around B (this is the magnetic part)
    t = QM * B_FIELD * dt / 2
    t_squared = t[0] ** 2 + t[1] ** 2 + t[2] ** 2
    s = 2 * t / (1 + t_squared)
    v_prime = v_minus + np.cross(v_minus, t)
    v_plus = v_minus + np.cross(v_prime, s)

    # second half push from the electric field
    new_v = v_plus + half_push

    # move the electron
    new_p = p + new_v * dt
    return new_p, new_v


# draw one arrow with a label, and return the pieces so we can delete them later
def draw_arrow(start, direction, length, color, label):
    pieces = []
    d = size(direction)
    if d < 0.001:
        return pieces
    arrow = ax.quiver(start[0], start[1], start[2],
                      direction[0], direction[1], direction[2],
                      length=length, normalize=True, color=color, linewidth=2)
    tip = start + direction / d * length
    text = ax.text(tip[0], tip[1], tip[2], label, color=color, fontsize=12)
    pieces.append(arrow)
    pieces.append(text)
    return pieces


# draw the straight field lines of E and B around the electron
# (this is only called now and then, NOT every frame)
def rebuild_field_lines(base_x, base_y, base_z):
    # delete the old lines
    for piece in field_pieces:
        piece.remove()
    field_pieces.clear()

    # external electric field lines (straight lines along +x)
    if SHOW_E_LINES and E_STRENGTH > 0.01:
        for y_shift in [-2.0, 2.0]:
            for z_shift in [-2.0, 2.0]:
                y = base_y + y_shift
                z = base_z + z_shift
                line, = ax.plot([base_x - 5, base_x + 5], [y, y], [z, z],
                                color="#378ADD", linewidth=0.8, alpha=0.4)
                arrow = ax.quiver(base_x, y, z, 1, 0, 0,
                                  length=0.6, normalize=True,
                                  color="#378ADD", linewidth=1, alpha=0.6)
                field_pieces.append(line)
                field_pieces.append(arrow)

    # external magnetic field lines (straight lines along +z)
    if SHOW_B_LINES and B_STRENGTH > 0.01:
        for x_shift in [-2.0, 2.0]:
            for y_shift in [-2.0, 2.0]:
                x = base_x + x_shift
                y = base_y + y_shift
                line, = ax.plot([x, x], [y, y], [base_z - 5, base_z + 5],
                                color="#7F77DD", linewidth=0.8, alpha=0.4)
                arrow = ax.quiver(x, y, base_z, 0, 0, 1,
                                  length=0.6, normalize=True,
                                  color="#7F77DD", linewidth=1, alpha=0.6)
                field_pieces.append(line)
                field_pieces.append(arrow)


# ---------------- DRAW EVERYTHING ONCE ----------------

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection="3d")
ax.view_init(25, 40)

# hide the grid, panes and numbers (drawing them every frame is the slowest part)
ax.set_axis_off()

# a short note instead of axis labels
fig.text(0.5, 0.04, "E field points along +x    B field points along +z    (drag to rotate)",
         ha="center", fontsize=10)

# the trail line (starts empty, we add points to it every frame)
trail_line, = ax.plot([], [], [], color="#1D9E75", linewidth=1.8)

# the electron (one big dot)
electron_dot, = ax.plot([0], [0], [0], "o", color="#1D9E75", markersize=10)

# the electron's own E field lines (8 short lines, one toward each corner)
radial_lines = []
radial_dirs = []
if SHOW_E_LINES:
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                direction = np.array([sx, sy, sz]) / np.sqrt(3)
                # markevery=[1] puts a dot only on the END of the line
                line, = ax.plot([0, 0], [0, 0], [0, 0], color="#378ADD",
                                linewidth=1, alpha=0.7,
                                marker="o", markersize=3, markevery=[1])
                radial_lines.append(line)
                radial_dirs.append(direction)

# the straight external field lines (rebuilt only when the electron has moved far)
field_pieces = []
last_base = None

# the four arrows v, F, E, B (these are deleted and redrawn every frame)
vector_pieces = []

# ---------------- MAIN LOOP ----------------

for frame in range(N_FRAMES):
    # stop if the window was closed
    if not plt.fignum_exists(fig.number):
        break

    # move the electron forward in time
    small_dt = DT_PER_FRAME / SUB_STEPS
    for step in range(SUB_STEPS):
        pos, vel = move_one_step(pos, vel, small_dt)
    time_passed = time_passed + DT_PER_FRAME

    # remember the position for the trail (only the last 400 points)
    trail_x.append(pos[0])
    trail_y.append(pos[1])
    trail_z.append(pos[2])
    if len(trail_x) > 400:
        trail_x.pop(0)
        trail_y.pop(0)
        trail_z.pop(0)

    # the camera follows the electron
    ax.set_xlim(pos[0] - 4, pos[0] + 4)
    ax.set_ylim(pos[1] - 4, pos[1] + 4)
    ax.set_zlim(pos[2] - 4, pos[2] + 4)

    # rebuild the straight field lines only if the electron moved 2 units
    base_x = 2 * round(pos[0] / 2)
    base_y = 2 * round(pos[1] / 2)
    base_z = 2 * round(pos[2] / 2)
    if last_base != (base_x, base_y, base_z):
        rebuild_field_lines(base_x, base_y, base_z)
        last_base = (base_x, base_y, base_z)

    # move the electron dot and the trail
    electron_dot.set_data_3d([pos[0]], [pos[1]], [pos[2]])
    if SHOW_TRAIL:
        trail_line.set_data_3d(trail_x, trail_y, trail_z)

    # move the electron's own field lines
    for i in range(len(radial_lines)):
        start = pos + radial_dirs[i] * 1.4
        end = pos + radial_dirs[i] * 0.3
        radial_lines[i].set_data_3d([start[0], end[0]],
                                    [start[1], end[1]],
                                    [start[2], end[2]])

    # delete the old v, F, E, B arrows and draw new ones
    for piece in vector_pieces:
        piece.remove()
    vector_pieces.clear()
    if SHOW_VECTORS:
        force = get_force(vel)
        speed_length = min(0.9 * size(vel), 2.2)
        force_length = min(0.9 * size(force), 2.2)
        vector_pieces.extend(draw_arrow(pos, vel, speed_length, "#639922", "v"))
        vector_pieces.extend(draw_arrow(pos, force, force_length, "#E24B4A", "F"))
        if E_STRENGTH > 0.01:
            vector_pieces.extend(draw_arrow(pos, E_FIELD, 0.5 + 0.4 * E_STRENGTH, "#378ADD", "E"))
        if B_STRENGTH > 0.01:
            vector_pieces.extend(draw_arrow(pos, B_FIELD, 0.5 + 0.4 * B_STRENGTH, "#7F77DD", "B"))

    # title with numbers
    title = "t = " + str(round(time_passed, 1))
    title = title + "   |v| = " + str(round(size(vel), 2))
    title = title + "   |F| = " + str(round(size(get_force(vel)), 2))
    ax.set_title(title)

    # wait a moment so the window can update
    plt.pause(0.001)

plt.show()
