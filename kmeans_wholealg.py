import numpy as np
import random
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D

# data generation
sphere = Hypersphere(dim=2, equip=False)
cluster = sphere.random_von_mises_fisher(kappa=20, n_samples=30)
#cluster = cluster.tolist()
SO3 = SpecialOrthogonal(3, equip=False)
rotation1 = SO3.random_uniform()
rotation2 = SO3.random_uniform()
cluster_1 = cluster @ rotation1
cluster_2 = cluster @ rotation2

data = np.vstack((cluster_1,cluster_2))

def dot_product(vector1, vector2):
    total = []
    for a, b in zip(vector1, vector2):
        total.append(a * b)
    return sum(total)
def norm(vector):
    vector = np.array(vector)
    return np.sqrt(dot_product(vector, vector))
def log(mean, instance):
    numer = (instance - mean * dot_product(mean, instance))
    denomin = np.sqrt(1-(dot_product(mean,instance)**2))
    log_result = (numer/denomin) * (np.arccos(dot_product(mean,instance)))
    return log_result
def exp(initial, direction):
    learning_rate = 0.5
    step = learning_rate * direction
    update_step = (initial*np.cos(norm(step))) + (step/norm(step))*np.sin(norm(step))
    return update_step

def jaccard(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    similarity=float(intersection) / union
    distance=1-similarity
    return similarity,distance
def plot_it(dt, cls):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    #ax.plot_surface(x, y, z, color="white", linewidth=0.25)
    ax.scatter(dt[:, 0], dt[:, 1], dt[:, 2], color='blue')
    ax.scatter(cls[0]['center'][0], cls[0]['center'][1], cls[0]['center'][2], color='red')
    ax.scatter(cls[1]['center'][0], cls[1]['center'][1], cls[1]['center'][2], color='red')
    #ax.scatter(cls[2]['center'][0], cls[2]['center'][1], cls[2]['center'][2], color='red')
    ax.plot_wireframe(x, y, z, color="black", linewidth=0.25)
    ax.set_title("Intrinsic K-Means on the Sphere")
    plt.show()

'''data = pd.read_csv('../train.csv', nrows = 500)
data = data.to_numpy()[:,1:]
#labels = data[:,0].tolist()'''

'''
iris = datasets.load_iris()
data = iris.data'''

normalised = []

# normalisation
miin = sum(data)/len(data)
for idx, point in enumerate(data):
    normalised.append(point - miin)
normalised = np.array(normalised)
normalised2 = []
for point in normalised:
    normalised2.append(point / norm(point))
data = normalised2

k = 2
means = []
for ctr in range(k):
    choc = random.choice(data)
    means.append(choc)
    # kmeans ++ here
    whereisit = []
    for point in data:
        whereisit.append((point == choc).all())
    ix = whereisit.index(True)
    #print(len(data))
    data.pop(ix)


clusters = {}

for idx in range(k):
    center = means[idx]
    points = []
    cluster = {
        'center': center,
        'points': []
    }

    clusters[idx] = cluster

counter = 0

while counter < 5:
    for point in data:
        distances = []
        for mean_idx in range(len(means)):
            log_raw = log(means[mean_idx], point)
            dis = dot_product(log_raw, log_raw)
            distances.append(dis)
        assigned_cluster = np.argmin(distances)

        clusters[assigned_cluster]['points'].append(point)

    for i in range(len(means)):
        points = np.array(clusters[i]['points'])
        logs = []
        if len(points) > 0:
            for point in points:
                log_raw = log(clusters[i]['center'], point)
                logs.append(log_raw)

            avg = sum(logs)/len(logs)
            new_center = exp(clusters[i]['center'], avg)
            clusters[i]['center'] = new_center
            clusters[i]['points'] = []
    counter += 1

#plot_it(data, clusters)

# SILHOUETTE SCORE
for point in data:
    distances = []
    for mean_idx in range(len(means)):
        dis = np.arccos(dot_product(means[mean_idx], point))
        distances.append(dis)
    assigned_cluster = np.argmin(distances)

    clusters[assigned_cluster]['points'].append(point)
whole = []
for c in range(k):
    one_point = []
    a_silh = []
    b_silh = []
    for point in clusters[c]['points']:
        print('-----')
        for a in range(k):
            dis = np.arccos(dot_product(point, clusters[c]['center']))
            #print('own center,', dis)
            dis2 = np.arccos(dot_product(point, clusters[a]['center']))
            #print('other center', dis2)
            a_silh.append(dis)
            b_silh.append(dis2)
        silh = []
        for i in range(len(a_silh)):
            silh.append((b_silh[i] - a_silh[i])/max(a_silh[i],b_silh[i]))
        one_point.append(sum(silh)/len(silh))
        print(sum(silh)/len(silh))
    whole.append(sum(one_point)/len(one_point))
print(np.mean(whole))
data = np.array(data)
plot_it(data, clusters)

jaccard(clusters[0]['points'].tolist(), clusters[1]['points'].tolist())

