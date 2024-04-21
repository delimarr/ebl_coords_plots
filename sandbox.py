import pandas as pd
import numpy as np
import pyvista as pv

from os import listdir
from os.path import join

from src.tvData import get_df, get_cloud, get_nn_mask, mark_valid_signal, median_filter, get_dist
from src.constants import NN_THRESHOLD, FILES_S4, GOOD_FILES_S4

DATA_FOLDER = "./data/"

def get_plotter(title: str) -> pv.Plotter:
    """Create Plotter object with title.

    Args:
        title (str): title

    Returns:
        pv.Plotter: pv plotter object
    """
    pl = pv.Plotter(notebook=False)
    pl.add_title(title)
    pl.enable_eye_dome_lighting()
    pl.show_grid()
    pl.show_axes()
    return pl

df_recv, recv_flg = mark_valid_signal(
    get_df(FILES_S4, join(DATA_FOLDER, "gr_ringstrecke_s4/")),
    min_distance=3000,
    max_distance=12000
)
nn_mask = get_nn_mask(df_recv, 50)
df_recv = df_recv[1: -1]
got_signal_flg = df_recv.got_signal_flg.to_numpy()

color_mask = np.zeros(nn_mask.shape, dtype=np.uint8)
color_mask[nn_mask & got_signal_flg] = 1
color_mask[nn_mask & (~got_signal_flg)] = 2
color_mask[(~nn_mask) & got_signal_flg] = 3

df_recv.insert(df_recv.shape[1], "color_mask", color_mask)
pl_color = get_plotter("Messfehler vs GoT")
cloud_color = get_cloud(df_recv)
cloud_color["color_mask"] = color_mask

pl_color.add_points(
    get_cloud(df_recv[nn_mask & got_signal_flg]),
    color="black",
    label="Guter Punkt und starkes Signal",
    render_points_as_spheres=True
)
pl_color.add_points(
    get_cloud(df_recv[(~nn_mask) & (~got_signal_flg)]),
    color="red",
    label="Schlechter Punkt und schwaches Signal",
    render_points_as_spheres=True
)
pl_color.add_points(
    get_cloud(df_recv[nn_mask & (~got_signal_flg)]),
    color="blue",
    label="Guter Punkt und schwaches Signal",
    render_points_as_spheres=True
)

pl_color.add_points(
    get_cloud(df_recv[(~nn_mask) & got_signal_flg]),
    color="yellow",
    label="Schlechter Punkt und gutes Signal",
    render_points_as_spheres=True
)

pl_color.add_legend()
pl_color.export_html("./verteidigung/mess_vs_signal.html")
pl_color.show()
   