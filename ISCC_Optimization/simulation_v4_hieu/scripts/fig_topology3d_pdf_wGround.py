#!/usr/bin/env python3
"""Fig. 1 -- a 3D snapshot of the simulated SI-TNTN, drawn from a real HPPP draw.

Replaces the schematic TikZ figure: UAVs sit at their sampled altitudes, ground
nodes on z = z_min, and the links drawn are the ones the SIJNR of (7) and (11)
actually contains -- G2A offloading, A2G broadcasting, and jamming on both.

Palette is the Okabe-Ito / Tol subset validated for CVD separation (worst-case
OKLab dE = 14 across deuter-/prot-/tritanopia); every class additionally carries
its own marker, so identity never rests on colour alone.
"""

from __future__ import annotations

import numpy as np

from _common import FIGURES, base_parser
from stochastic_mec import SystemParams, sample_topology

# class -> (colour, marker, size, label)
STYLE = {
    "ul_uav":   ("#0072B2", "s", 70, "UL UAV + inference server"),
    "dl_uav":   ("#D55E00", "^", 80, "DL UAV + inference server"),
    "ul_ue":    ("#117733", "o", 30, "UL UE (senses, offloads)"),
    "dl_ue":    ("#AA4499", "D", 26, "DL UE (receives)"),
    "inactive": ("#DDDDDD", "o", 18, "inactive UE"),
    "jammer":   ("#000000", "x", 42, "jammer"),
}


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--uavs", type=int, default=5)
    ap.add_argument("--ues", type=int, default=12)
    ap.add_argument("--jammers", type=int, default=3)
    ap.add_argument("--elev", type=float, default=20.0)
    ap.add_argument("--azim", type=float, default=-58.0)
    ap.add_argument("--pad", type=float, default=0.04, help="padding around figure in inches")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, to_rgba
    from matplotlib.lines import Line2D
    from matplotlib.transforms import Bbox

    rng = np.random.default_rng(args.seed)
    params = SystemParams()
    for _ in range(400):
        topo = sample_topology(rng, params, n_sbs=args.uavs, n_active_ue=args.ues,
                               n_inactive_ue=4, n_jammers=args.jammers)
        if topo.ul_cells.size >= 2 and topo.dl_cells.size >= 1:
            break

    side = params.area_side / 2.0
    fig = plt.figure(figsize=(7.0, 4.2))   # 5:3
    ax = fig.add_subplot(111, projection="3d")
    ax.set_position([-0.05, -0.10, 1.12, 1.18])

    ul_cells, dl_cells = topo.ul_cells, topo.dl_cells
    ul_ues, dl_ues = topo.ul_ues, topo.dl_ues

    # ---- ground plane & Voronoi cells ---------------------------------- #
    grid = np.linspace(-side, side, 400)
    gx_c, gy_c = np.meshgrid(grid, grid)
    uav_xy = topo.sbs_pos[:, :2]
    pts = np.stack([gx_c.ravel(), gy_c.ravel()], axis=1)
    nearest = np.argmin(
        np.linalg.norm(pts[:, None, :] - uav_xy[None, :, :], axis=-1), axis=1
    ).reshape(gx_c.shape)

    cell_colors = []
    for m in range(topo.n_sbs):
        if m in ul_cells:
            cell_colors.append(to_rgba(STYLE["ul_uav"][0], alpha=0.07))
        elif m in dl_cells:
            cell_colors.append(to_rgba(STYLE["dl_uav"][0], alpha=0.07))
        else:
            cell_colors.append(to_rgba("0.7", alpha=0.05))

    cmap = ListedColormap(cell_colors)
    levels = np.arange(topo.n_sbs + 1) - 0.5

    # sits marginally below z=0 so mpl's painter's algorithm never paints it
    # over a UAV marker that is genuinely above it
    ax.contourf(gx_c, gy_c, nearest, zdir="z", offset=-2.0, levels=levels,
                cmap=cmap, zorder=0)
    ax.contour(gx_c, gy_c, nearest, zdir="z", offset=-2.0,
               levels=np.arange(topo.n_sbs) + 0.5, colors="0.6",
               linewidths=0.8, zorder=1)
    ax.plot([-side, side, side, -side, -side], [-side, -side, side, side, -side],
            [0, 0, 0, 0, 0], color="0.6", lw=0.7, zorder=1)

    # ---- links: exactly the terms that enter the SIJNR ------------------ #
    for n in ul_ues:                                   # G2A offloading
        m = topo.ue_cell[n]
        a, b = topo.ue_pos[n], topo.sbs_pos[m]
        ax.plot(*zip(a, b), color=STYLE["ul_ue"][0], lw=0.8, alpha=0.75, zorder=2)
    for n in dl_ues:                                   # A2G broadcasting
        m = topo.ue_cell[n]
        a, b = topo.sbs_pos[m], topo.ue_pos[n]
        ax.plot(*zip(a, b), color=STYLE["dl_ue"][0], lw=0.8, alpha=0.75, zorder=2)
    for q in topo.jammer_pos:                          # jamming, both directions
        for m in np.concatenate([ul_cells, dl_cells])[:2]:
            ax.plot(*zip(q, topo.sbs_pos[m]), color="0.45", lw=0.6,
                    ls=(0, (3, 3)), alpha=0.8, zorder=1)

    # ---- altitude drop lines make the 3rd dimension readable ------------ #
    for m in range(topo.n_sbs):
        x, y, z = topo.sbs_pos[m]
        ax.plot([x, x], [y, y], [0, z], color="0.7", lw=0.6, ls=":", zorder=1)

    def scat(pos, key, **kw):
        if len(pos) == 0:
            return
        c, mk, s, _ = STYLE[key]
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=c, marker=mk, s=s,
                   edgecolors="white", linewidths=0.5, depthshade=False,
                   zorder=5, **kw)

    scat(topo.sbs_pos[ul_cells], "ul_uav")
    scat(topo.sbs_pos[dl_cells], "dl_uav")
    scat(topo.ue_pos[ul_ues], "ul_ue")
    scat(topo.ue_pos[dl_ues], "dl_ue")
    scat(topo.ue_inactive_pos, "inactive")
    if topo.jammer_pos.size:
        c, mk, s, _ = STYLE["jammer"]
        ax.scatter(*topo.jammer_pos.T, c=c, marker=mk, s=s, linewidths=1.2,
                   depthshade=False, zorder=6)

    ax.set_xlabel("$x$ [m]", labelpad=-4)
    ax.set_ylabel("$y$ [m]", labelpad=-4)
    ax.set_zlabel("Altitude [m]", labelpad=-6)
    ax.set_xlim(-side, side); ax.set_ylim(-side, side)
    ax.set_zlim(0, params.z_uav_max * 1.15)
    ax.set_box_aspect((1, 1, 0.34))
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.tick_params(labelsize=7, pad=-2)
    ax.grid(True, alpha=0.25)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0.0)

    handles = [Line2D([], [], color=c, marker=mk, ls="none",
                      markersize=np.sqrt(s) * 0.85, markeredgecolor="0.4",
                      markeredgewidth=0.4, label=lab)
               for c, mk, s, lab in STYLE.values()]
    handles += [
        Line2D([], [], color=STYLE["ul_ue"][0], lw=1.0, label="G2A offloading"),
        Line2D([], [], color=STYLE["dl_ue"][0], lw=1.0, label="A2G broadcasting"),
        Line2D([], [], color="0.45", lw=0.8, ls=(0, (3, 3)), label="jamming"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.74),
               ncol=3, fontsize=6.8, frameon=False, columnspacing=1.3,
               labelspacing=0.25, handletextpad=0.5, borderaxespad=0.0)

    # Compute tight bounding box enclosing all content (legend, 3D plot, labels)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    non_bg = (rgba[:, :, :3] < 250).any(axis=2)
    coords = np.argwhere(non_bg)
    if coords.size:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        dpi = fig.dpi
        h_px, w_px = rgba.shape[:2]
        pad = args.pad
        x_min = max(0.0, x0 / dpi - pad)
        x_max = min(w_px / dpi, (x1 + 1) / dpi + pad)
        y_min = max(0.0, (h_px - (y1 + 1)) / dpi - pad)
        y_max = min(h_px / dpi, (h_px - y0) / dpi + pad)
        tight_bbox = Bbox([[x_min, y_min], [x_max, y_max]])
    else:
        tight_bbox = "tight"

    out_target = args.out or FIGURES
    if out_target.suffix.lower() in {".pdf", ".png"}:
        out = out_target
    else:
        out = out_target / "topology3d_wGround.pdf"
    fig.savefig(out, dpi=300, bbox_inches=tight_bbox)
    print("UAV altitudes [m]:", np.round(topo.sbs_pos[:, 2], 1).tolist())
    print(f"UL UAVs {ul_cells.tolist()}  DL UAVs {dl_cells.tolist()}  "
          f"UL UEs {ul_ues.size}  DL UEs {dl_ues.size}  jammers {topo.n_jammer}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
