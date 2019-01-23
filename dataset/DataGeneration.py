
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D


def generate_dataset(nb_points=100, nb_labels=3, var=0.1):

    centers = []
    while len(centers) < nb_labels:
        new_center = np.random.rand(3)
        if all([np.linalg.norm(center - new_center) > 0.5 for center in centers]):
            centers.append(new_center)

    points = []
    labels = []
    for i in range(nb_points):
        center = centers[i % nb_labels]
        point = np.random.normal(center, var)
        points.append(point)
        labels.append(i % nb_labels)

    idx = np.random.permutation(nb_points)

    return np.array(points)[idx], np.array(labels)[idx], np.array(centers)


def plot_dataset(points, labels, centers=None):

    fig = plt.figure("Data")
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=labels)
    if centers is not None:
        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], s=150, c='red')

    plt.show()


def animate(points, labels, weights, indexes):
    
    centers = weights[:]
    maxi_len = len(max(centers, key=len))
    for i in range(len(centers)):
        while len(centers[i]) < maxi_len:
            centers[i].append(centers[i][-1])
        centers[i] = np.array(centers[i])
    centers = np.array(centers)
    size = centers.shape[1]

    def update(num, nodes):
        print(f"Frame {num}")
        nodes._offsets3d = (centers[num][:, 0], centers[num][:, 1], centers[num][:, 2])
        col = ['blue'] * size
        col[indexes[num]] = 'red'
        nodes._facecolor3d = col
        return nodes,

    fig = plt.figure("Animation")
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=labels)
    nodes = ax.scatter(centers[0][:, 0], centers[0][:, 1], centers[0][:, 2], s=200, c='blue')
    points_anim = animation.FuncAnimation(fig, update, len(centers), fargs=(nodes,),
                                          interval=1, repeat=False)
    return fig, points_anim


if __name__ == '__main__':
    plot_dataset(*generate_dataset())