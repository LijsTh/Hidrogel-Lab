from matplotlib.animation import FuncAnimation
from IPython.display import Video



def render_particle_movie(df, frames=None, step=1, color=None, cmap='viridis', clim=None,
                          s=18, size_map=None, c='#222222', alpha=.8, bg=None, figsize=(10, 18),
                          out="../outputs/movie.mp4", fps=80, dpi=80, capture_fps=None, title=''):
    """Renderiza un mp4 de las bolitas trackeadas, frame a frame. REUTILIZABLE.

    df           : tabla con columnas 'x','y','frame' (y, opcional, la columna `color`).
    frames       : frames a animar (default: 0..max submuestreado de a `step`).
    color        : nombre de columna para colorear los puntos (default: color plano `c`, SIN etiquetar).
    clim         : (min,max) para la escala de color, si se usa `color`.
    size_map     : dict {valor_de_color: tamano}, para resaltar una categoria especifica (opcional).
    bg           : funcion frame->imagen 2D para dibujar el video real detras (gancho a futuro; default None).
    out          : ruta del mp4 (default: OUTDIR/movie.mp4).
    capture_fps  : fps del video original, solo para mostrar t=frame/capture_fps en el titulo (opcional).
    Devuelve un IPython.display.Video embebido.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    if frames is None:
        frames = range(0, int(df['frame'].max()) + 1, step)
    frames = list(frames)
    by   = dict(tuple(df.groupby('frame')))
    xlim = (df.x.min() - 10, df.x.max() + 10)
    ylim = (df.y.max() + 10, df.y.min() - 10)          # y invertido (coords de imagen)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect('equal')
    ax.set_xlabel('x [px]'); ax.set_ylabel('y [px]')
    imbg = ax.imshow(bg(frames[0]), cmap='gray', extent=[xlim[0], xlim[1], ylim[0], ylim[1]], zorder=0) if bg else None
    import pandas as pd
    color_is_categorical = color is not None and not pd.api.types.is_numeric_dtype(df[color])
    if color is not None and not color_is_categorical:
        sc = ax.scatter([], [], s=s, cmap=cmap, alpha=alpha, edgecolors=None, zorder=2)
        if clim: sc.set_clim(*clim)
    else:
        sc = ax.scatter([], [], s=s, c=c, alpha=alpha, edgecolors=None, zorder=2)
    ttl = ax.set_title(title)

    def upd(i):
        fr = frames[i]; d = by.get(fr)
        sc.set_offsets(d[['x', 'y']].values if d is not None else np.empty((0, 2)))
        if color is not None and d is not None:
            if color_is_categorical:
                sc.set_facecolor(d[color].values)
            else:
                sc.set_array(d[color].values)
            if size_map is not None:
                sc.set_sizes(d[color].map(size_map).fillna(s).values)
        if bg is not None: imbg.set_data(bg(fr))
        tstr = f' | t={fr/capture_fps:.1f}s' if capture_fps else f' | t={fr/fps:.1f}s'
        ttl.set_text(f'{title}\nframe {fr}{tstr} | N={0 if d is None else len(d)}')

    FuncAnimation(fig, upd, frames=len(frames), interval=1000/fps, blit=False).save(
        out, writer='ffmpeg', fps=fps, dpi=dpi)
    plt.close(fig)
    return Video(out, embed=True)