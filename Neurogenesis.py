
import numpy as np
from sklearn.metrics import confusion_matrix

from dataset.DataGeneration import generate_dataset, animate
from dataset.load_iris import get_dataset


DATASET = 'not IRIS'

np.random.seed(12321)

INSERT_THRESHOLD = 0.80
FIRING_THRESHOLD = 0.10

BEST_LR = 2
NEIGH_LR = 0.05

FIRING_RATE = 1.05
BEST_FIRING_RATE = 0.3
NEIGH_FIRING_RATE = 0.1

MAX_AGE = 25

LABEL_PLUS = 1
LABEL_MOINS = 0.1


class Node:

    def __init__(self, input_size, weights=None):
        if weights is None:
            self.weights = np.random.rand(input_size)
        else:
            self.weights = weights

        self.labels = {}
        self.firing_counter = 1

    def compute_distance(self, input_vector):
        return np.linalg.norm(self.weights - input_vector)**2

    def update_weights(self, input_vector, best=False):
        eps = BEST_LR if best else NEIGH_LR
        self.weights += eps * self.firing_counter * (input_vector - self.weights)

    def update_firing_counter(self, best=False):
        fr = BEST_FIRING_RATE if best else NEIGH_FIRING_RATE
        self.firing_counter += fr * FIRING_RATE * (1 - self.firing_counter) - fr

    def add_label(self, label):
        self.labels[label] = 1

    def copy_labels(self, labels):
        for key in labels.keys():
            self.labels[key] = 1

    def update_labels(self, label):
        self.labels[label] += LABEL_PLUS
        for lab in self.labels.keys():
            if lab != label:
                self.labels[lab] -= LABEL_MOINS

    def get_label(self):
        return max(self.labels.keys(), key=lambda k: self.labels[k])

    def __repr__(self):
        return "\n{ Weights: " + str(self.weights) + "\nFiring counter: " + str(self.firing_counter) + "}"


class Edges:

    def __init__(self, nb_nodes):
        self.edges = np.zeros((nb_nodes, nb_nodes))
        self.ages = np.zeros((nb_nodes, nb_nodes))

    def set_edge(self, index1, index2, value):
        self.edges[index1, index2] = value
        self.edges[index2, index1] = value

    def set_age(self, index1, index2, value):
        self.ages[index1, index2] = value
        self.ages[index2, index1] = value

    def add_edge(self, best_index, second_index, new_node=False):
        n = self.edges.shape[0]
        if new_node:
            tmp = self.edges
            self.edges = np.zeros((n+1, n+1))
            self.edges[:-1, :-1] = tmp
            tmp = self.ages
            self.ages = np.zeros((n+1, n+1))
            self.ages[:-1, :-1] = tmp
            self.set_edge(best_index, second_index, 0)
            self.set_age(best_index, second_index, 0)
            self.set_edge(best_index, n, 1)
            self.set_edge(second_index, n, 1)
        else:
            self.set_edge(best_index, second_index, 1)
            self.set_age(best_index, second_index, 0)

    def update_age(self, best):
        neighbours = self.get_neighbours(best)
        for neighbour in neighbours:
            age = self.ages[best, neighbour]
            self.set_age(best, neighbour, age+1)

    def get_neighbours(self, node_index):
        # print("self.edges[node_index] :\n", self.edges[node_index], '\n\n', '-'*200, '\n')    # debug
        return np.nonzero(self.edges[node_index])[0]

    def remove_edges(self):
        for i in range(self.edges.shape[0]):
            for j in range(i+1, self.edges.shape[0]):
                if self.ages[i, j] > MAX_AGE:
                    # print(f"Remove edge ({i}, {j})")
                    self.set_edge(i, j, 0)
                    self.set_age(i, j, 0)

    def remove_node(self, node_index):
        self.edges = np.delete(self.edges, node_index, axis=0)
        self.edges = np.delete(self.edges, node_index, axis=1)
        self.ages = np.delete(self.ages, node_index, axis=0)
        self.ages = np.delete(self.ages, node_index, axis=1)

    def __repr__(self):
        return "{Edges:\n" + str(self.edges) + "\nAges:\n" + str(self.ages) + "\n}\n"


class Layer:

    def __init__(self, input_size):
        self.input_size = input_size

        self.nb_nodes = 2
        self.nodes = []
        for i in range(self.nb_nodes):
            self.nodes.append(Node(input_size))

        self.edges = Edges(self.nb_nodes)

        # print(f"Nb nodes : {self.nb_nodes}")
        # print(f"nodes : {self.nodes}")
        # print(f"Edges : {self.edges}\n\n\n")

    def get_neighbours(self, best_index):
        return self.edges.get_neighbours(best_index)

    def find_bmus(self, input_vector):
        distance = np.array([node.compute_distance(input_vector)
                             for node in self.nodes])
        # print(f"Distance of each node : {distance}")
        indexes = np.argsort(distance)
        return indexes[0], indexes[1], distance[indexes[0]]

    def add_node(self, best_index, second_index, input_vector, label):
        new_weights = (input_vector + self.nodes[best_index].weights) / 2
        new_node = Node(self.input_size, new_weights)
        new_node.copy_labels(self.nodes[0].labels)
        new_node.labels[label] += 1
        self.nodes.append(new_node)
        self.edges.add_edge(best_index, second_index, new_node=True)
        self.nb_nodes += 1

    def remove_nodes(self):
        # print("self.nb_nodes :\n", self.nb_nodes, '\n\n', '-'*200, '\n')    # debug
        for node in range(self.nb_nodes-1, -1, -1):
            # print("node :\n", node, '\n\n', '-'*200, '\n')    # debug
            if len(self.edges.get_neighbours(node)) == 0:
                # print("self.edges :\n", self.edges, '\n\n', '-'*200, '\n')    # debug
                # print(f"Removing node {node}, "
                #       f"with edges {self.edges.edges[node]}, "
                #       f"with ages {self.edges.ages[node]}, "
                #       f"current number of nodes : {self.nb_nodes}")
                self.edges.remove_node(node)
                self.nodes.pop(node)
                self.nb_nodes -= 1

    def train(self, samples, labels, epochs=10):

        weights, indexes = [], []
        for epoch in range(epochs):
            for sample, label in zip(samples, labels):

                if label not in self.nodes[0].labels:
                    for node in self.nodes:
                        node.add_label(label)

                # print('-'*80)
                # print(f"Epoch {epoch}, sample {sample}")

                # print(f"Find BMU")
                best, second, best_dist = self.find_bmus(sample)
                # print(f"Best : {best}, best dist : {best_dist}, second : {second}\n\n")

                # print("Add edge")
                self.edges.add_edge(best, second, new_node=False)
                # print("Edges : ", self.edges, "\n\n")

                # print(f"Best firing counter : {self.nodes[best].firing_counter}\n\n")
                # print("np.exp(-best_dist) :\n", np.exp(-best_dist), '\n\n', '-'*200, '\n')    # debug
                # print("self.nodes[best].firing_counter :\n", self.nodes[best].firing_counter, '\n\n', '-'*200, '\n')    # debug
                if np.exp(-best_dist) < INSERT_THRESHOLD and \
                        self.nodes[best].firing_counter < FIRING_THRESHOLD:
                    # print("Add a node")
                    self.add_node(best, second, sample, label)

                else:
                    # print("Update weights")
                    # print("Before : ", self.nodes)
                    self.nodes[best].update_weights(sample, best=True)
                    for neighbour in self.get_neighbours(best):
                        self.nodes[neighbour].update_weights(sample)
                    self.nodes[best].update_labels(label)
                    # print("After : ", self.nodes, "\n\n\n")

                # print("Nb nodes : ", self.nb_nodes)
                # print("Len nodes : ", len(self.nodes))
                # print("nodes : ", self.nodes, "\n\n")
                # print("Edges : ", self.edges, "\n\n")
                # print("Update firing counter")
                self.nodes[best].update_firing_counter(best=True)
                # print("self.get_neighbours(best)  :\n", self.get_neighbours(best) , '\n\n', '-'*200, '\n')    # debug
                for neighbour in self.get_neighbours(best):
                    self.nodes[neighbour].update_firing_counter()
                # print(f"nodes : {self.nodes}\n\n")

                # print("\n\nUpdate age")
                self.edges.update_age(best)
                # print(self.edges)
                # print("\n\nRemove edges ?")
                self.edges.remove_edges()
                # print("Edges : ", self.edges)
                # print("\n\nRemove nodes ?")
                self.remove_nodes()
                # print("nodes : ", self.nodes)

                weights.append([node.weights.copy() for node in self.nodes])
                indexes.append(best)

        return weights, indexes

    def predict(self, samples):

        labels = []
        for sample in samples:
            best, _, _ = self.find_bmus(sample)
            labels.append(self.nodes[best].get_label())
        return np.array(labels)


if __name__ == '__main__':

    if DATASET == "IRIS":

        data, labels = get_dataset()
        n = data.shape[0]
        idx = np.random.permutation(n)
        data, labels = data[idx], labels[idx]

    else:
        n = 500
        data, labels, centers = generate_dataset(nb_points=n, nb_labels=5)

    train_data, train_labels = data[:4*n//5], labels[:4*n//5]
    test_data, test_labels = data[4*n//5:], labels[4*n//5:]

    input_size = data.shape[1]
    network = Layer(input_size)

    weights, indexes = network.train(train_data, train_labels, 1)
    print(f"Number of nodes : {network.nb_nodes}")

    train_err = 100 - (network.predict(train_data) == train_labels).sum() / (4 * n // 5) * 100
    test_err = 100 - (network.predict(test_data) == test_labels).sum() / (n // 5) * 100
    print(f"Train error : {train_err:.4}%\n"
          f"Generalization error : {test_err:.4}%")

    print("Confusion matrix :\n", confusion_matrix(test_labels, network.predict(test_data)))

    if DATASET is not "IRIS":
        fig, animation = animate(train_data, train_labels, weights, indexes)
        fig.show()
