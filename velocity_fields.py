import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import convolve2d
from tqdm.std import tqdm


def to_vector_field(df):
    grid = 25
    x_bins = np.arange(0, 472, grid)
    y_bins = np.arange(0, 1024, grid)

    nx_bins = len(x_bins) - 1
    ny_bins = len(y_bins) - 1

    df["x_bin"] = np.digitize(df.x.to_numpy(), x_bins)
    df["y_bin"] = np.digitize(df.y.to_numpy(), y_bins)

    Px = df.pivot_table(
        index="y_bin", columns="x_bin", values="velocity_x", aggfunc="mean"
    )
    Px = Px.reindex(index=range(ny_bins), columns=range(nx_bins))
    Py = df.pivot_table(
        index="y_bin", columns="x_bin", values="velocity_y", aggfunc="mean"
    )
    Py = Py.reindex(index=range(ny_bins), columns=range(nx_bins))

    return np.stack((Px.to_numpy(), Py.to_numpy()), axis=-1)


# def smooth_vector_field(P, kernel_size=3):
#     kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2)
#     Px_smooth = convolve2d(P[:, :, 0], kernel, mode="same")
#     Py_smooth = convolve2d(P[:, :, 1], kernel, mode="same")

#     return np.stack((Px_smooth, Py_smooth), axis=-1)


def plot_vector_field(P):
    if P.shape[-1] != 2:
        raise ValueError("Ojo con las dimensiones!")
    x = np.arange(P.shape[1])
    y = np.arange(P.shape[0])
    X, Y = np.meshgrid(x, y)
    U = P[:, :, 0]  # x-component
    V = P[:, :, 1]  # y-component
    S = np.sqrt(U**2 + V**2)
    plt.figure(figsize=(12, 8))
    plt.quiver(X, Y, U, V, S, color="red")
    plt.gca().invert_yaxis()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Vector Field")
    plt.axis("equal")  # optional: equal aspect ratio
    plt.tight_layout()


def main():
    filenames = []
    for i in tqdm(range(1200)):
        frame = pd.read_csv(f"data/velocities_{i}.csv")
        P = to_vector_field(frame)
        plot_vector_field(P)
        filename = f"frames/vector_field_{i}.png"
        filenames.append(filename)
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close()  # Important: close the figure to free memory

    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    videowriter = cv2.VideoWriter("output_video.mp4", fourcc, 2, (472, 1200))
    for file in filenames:
        frame = cv2.imread(file)
        videowriter.write(frame)
        # imageio.get_writer("output_video.mp4", "mp4", images, fps=2)  # Adjust fps as needed
    videowriter.release()


if __name__ == "__main__":
    main()
