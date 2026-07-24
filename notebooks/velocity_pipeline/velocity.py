import numpy as np
import pandas as pd
def _calculate_smooth_velocity(trayectory):
    from scipy.signal import savgol_filter
    # derivada del ajuste polinómico = velocidad (px/frame). interp define los bordes.
    sg_window = 7   # frames por ventana
    sg_poly = 2     # grado del polinomio

    if len(trayectory) > sg_window:
        return savgol_filter(trayectory, sg_window, sg_poly, deriv=1, mode='interp')
    return np.nan




def extract_velocities(tray):
    tray.reset_index(drop=True, inplace=True)
    # Ordenamos por particula y por tiempo (frame) para garantizar la diferencia entre particulas.  
    tray.sort_values(['particle', 'frame'], inplace=True)
    # Luego utilizamos el Filtro de `Savtizky-Golay` en el que en lugar de dos puntos, tomamos una ventana de tiempo y ajustamos un polinomio `p(t) = at^2 + bt + c`. La velocidad es la derivada analitica de ese polinomio. 
    tray['vx'] = tray.groupby('particle', group_keys=True).x.transform(_calculate_smooth_velocity)
    tray['vy'] = tray.groupby('particle', group_keys=True).y.transform(_calculate_smooth_velocity)

    tray['v'] = np.sqrt(tray['vx']**2 + tray['vy']**2)

    # NO recortamos bordes: el arranque de la caída (pocket que empieza a caer) vive en los
    # primeros cuadros de cada traza y es señal real; el ruido de borde lo filtra el valle en nb02.
    vel = tray.dropna(subset=['vx', 'vy'])
    return vel
    
def _get_dense_grid(grid_data,frame_idx, value_col, nx_bins, ny_bins):
    try:
        data = grid_data.loc[frame_idx][value_col].unstack(fill_value=0)
        return data.reindex(index=range(ny_bins), columns=range(nx_bins), fill_value=0).values
    except KeyError:
        return np.zeros((ny_bins, nx_bins))

def extract_velocity_field(vel):
    grid_size = 25 # píxeles
    x_bins = np.arange(vel.x.min(), vel.x.max() + grid_size, grid_size)
    y_bins = np.arange(vel.y.min(), vel.y.max() + grid_size, grid_size)

    nx_bins = len(x_bins) - 1
    ny_bins = len(y_bins) - 1

    
    vel['x_bin'] = pd.cut(vel['x'], bins=x_bins, labels=False)
    vel['y_bin'] = pd.cut(vel['y'], bins=y_bins, labels=False)



    grid_data = vel.groupby(['frame', 'y_bin', 'x_bin'])[['vx', 'vy']].mean()

    frames_list = sorted(vel['frame'].unique())

    Px = np.array([_get_dense_grid(grid_data, f, 'vx', nx_bins, ny_bins) for f in frames_list])
    Py = np.array([_get_dense_grid(grid_data, f, 'vy', nx_bins, ny_bins) for f in frames_list])
    ctx = {
        "nx_bins": nx_bins,
        "ny_bins": ny_bins,
        "x_bins": x_bins,
        "y_bins": y_bins
    }
    return Px, Py, ctx, frames_list
