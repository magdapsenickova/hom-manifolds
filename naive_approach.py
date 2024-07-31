import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from geomstats.geometry.hypersphere import Hypersphere
import matplotlib.pyplot as plt
import seaborn as sns

def dot_product(vector1, vector2):
    total = []
    for a, b in zip(vector1, vector2):
        total.append(a * b)
    return sum(total)
def mu(instance, correct, incorrect):
    nomin = distance(instance, correct) - distance(instance, incorrect)
    denomin = distance(instance, correct) + distance(instance, incorrect)
    return nomin/denomin
def mu2(instance, correct, incorrect):
    nomin = norm(instance - correct) - norm(instance - incorrect)
    denomin = norm(instance - correct) + norm(instance - incorrect)
    return nomin/denomin
def norm(vector):
    return np.sqrt(dot_product(vector, vector))
def distance(vector1, vector2):
    return np.arccos(dot_product(vector1, vector2))
def monotone(values):
    return (np.e ** (-1 * values)) / ((1 + np.e ** (-1 * values)) ** 2)
def update_corr(instance, prototype_correct, prototype_incorrect):
    term1 = ((2 * distance(instance, prototype_incorrect)) /
             ((distance(instance, prototype_correct) + distance(instance, prototype_incorrect)) ** 2))
    term2 = (-1)*(instance - dot_product(instance,prototype_correct) * prototype_correct)/ (np.sqrt(1 - (dot_product(instance, prototype_correct) ** 2)))
    prelim_result = term1 * term2

    # multiplying the whole chain rule
    result = monotone(mu(instance,prototype_correct,prototype_incorrect)) * (prelim_result)
    return result
def update_incor(instance, prototype_correct, prototype_incorrect):
    term1 = ((2 * distance(instance, prototype_incorrect)) /
             ((distance(instance, prototype_correct) + distance(instance, prototype_incorrect)) ** 2))
    term2 = (-1) * (instance - dot_product(instance, prototype_incorrect) * prototype_incorrect) / (np.sqrt(1 - (dot_product(instance, prototype_incorrect) ** 2)))
    prelim_result = term1 * term2

    # multiplying the whole chain rule
    result = monotone(mu(instance,prototype_correct,prototype_incorrect)) * (prelim_result)
    return result
def exponential_map_corr(initial, direction, rate):
    learning_rate = - rate #0.25

    step = learning_rate * direction
    update_step = initial * np.cos(norm(step)) + (np.sin(norm(step))/norm(step)) * step
    return update_step
def exponential_map_incor(initial, direction, rate):
    learning_rate = + rate #0.25

    step = learning_rate * direction
    update_step = initial * np.cos(norm(step)) + (np.sin(norm(step))/norm(step)) * step
    return update_step

'''def plot_updates(point, updates):
    xs = []
    ys = []
    zs = []
    for row in updates:
        xs.append(row[0])
        ys.append(row[1])
        zs.append(row[2])'''

''' fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(point[0], point[1], point[2], color = 'red', label = 'instance')
    ax.scatter(xs, ys, zs,color='blue', label = 'prototype')
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color="black", linewidth = 0.25)
    ax.set_title("GLVQ on the Sphere")
    ax.legend(loc = 'upper left')
    plt.show()'''

def fit(data,labels, data_test, labels_test, learning_rate, k):
    # loading data
    #data = pd.read_csv('../train.csv',nrows=30)
    #data = data.to_numpy()[0:30] #shortened data
    #labels = data[:,0].tolist()
    normalised = []

    # data for scoring
    #data_test = pd.read_csv('../train.csv',nrows = 35)
    #data_test = data_test.to_numpy()[30:35]
    #labels_test = data_test[:,0].tolist()

    normalised_test = []

    # normalisation

    miin = sum(data)/len(data)

    for point in data:
        normalised.append(point - miin)
    normalised2 = []
    for idx, point in enumerate(normalised):
        normalised2.append(point/norm(point))
    normalised2 = np.array(normalised2)
    labelled_data = [list(a) for a in zip(normalised2, labels)]

    #k = 100
    prototypes = []
    for cls_num in range(0,10):
        new_prt = next(x for x in labelled_data if x[1] == cls_num)
        prototypes.append(new_prt)
        whereisit = []
        for datapoint in labelled_data:
            whereisit.append((datapoint[0] == new_prt[0]).all())
        ix = whereisit.index(True)
        labelled_data.pop(ix)

    for it in range(k-10):
        curr = random.choice(labelled_data)
        prototypes.append(curr)
        whereisit = []
        for datapoint in labelled_data:
            whereisit.append((datapoint[0] == curr[0]).all())
        ix = whereisit.index(True)
        labelled_data.pop(ix)

    naiv_prototypes = prototypes

    # the lvq loop
    for point in labelled_data:
        #print('------------------------------------------------------------')

        # computing and sorting distances
        dst = {}
        for i in range(len(prototypes)):
            dst[i] = distance(prototypes[i][0], point[0])
        all_sorted = sorted(dst.items(), key=lambda x: x[1])
        point_lbl = point[1]

        if prototypes[all_sorted[0][0]][1] == point_lbl:
            IDX_COR = all_sorted[0][0]
            IDX_INCOR = all_sorted[1][0]
            prototype_correct = prototypes[all_sorted[0][0]][0]
            prototype_incorrect = prototypes[all_sorted[1][0]][0]
        else:
            #print('False | point label: {}, prototype label: {}'.format(point_lbl, prototypes[all_sorted[0][0]][1]))
            for prt in all_sorted:
                if prototypes[prt[0]][1] == point_lbl:
                    IDX_COR = prt[0]
                    IDX_INCOR = all_sorted[0][0]
                    prototype_correct = prototypes[prt[0]][0]
                    prototype_incorrect = prototypes[all_sorted[0][0]][0]
                    break
        # initialisation
        instance = point[0]
        counter = 0

        # the lvq loop
        # correct prototype
        while counter < 1:
            # correct prototype
            v_corr = update_corr(instance, prototype_correct, prototype_incorrect)
            prototype_correct = exponential_map_corr(prototype_correct, v_corr, learning_rate)
            #print('CORRECT| norm: %.8f'% norm(prototype_correct), 'distance: %.3f,' % distance(prototype_correct,instance))
            counter += 1

        prototypes[IDX_COR][0] = prototype_correct
        counter = 0

        # incorrect prototype
        while counter < 1:
            v_incor = update_incor(instance, prototype_correct, prototype_incorrect)
            prototype_incorrect = exponential_map_incor(prototype_incorrect, v_incor, learning_rate)
            #print('INCORRECT| norm: %.8f'% norm(prototype_incorrect), 'distance: %.3f,' % distance(prototype_incorrect,instance))
            counter += 1

        prototypes[IDX_INCOR][0] = prototype_incorrect

    # TESTING ###########################
    # normalisation
    normalised2_test = []
    miin = sum(data_test)/len(data_test)
    for point in data_test:
        normalised_test.append(point - miin)

    normalised2_test = []
    for idx, point in enumerate(normalised_test):
        normalised2_test.append(point/norm(point))
    normalised2_test = np.array(normalised2_test)
    labelled_data_test = [list(a) for a in zip(normalised2_test, labels_test)]

    dis = {}
    res = []
    y_pred = []
    for point in labelled_data_test:
        exam = point[0]
        true_label = point[1]
        for j in range(len(prototypes)):
            dis[j] = distance(exam,prototypes[j][0])
        which = min(dis, key=dis.get)
        y_pred.append(prototypes[which][1])
        res.append(true_label == prototypes[which][1])

    return y_pred#('mine:', res.count(True)/len(res)*100)


X = pd.read_csv('../train.csv',nrows=10000)
X = X.to_numpy()#[0:500]
y = X[:,0].tolist()
X, X_test, y, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#X_test = pd.read_csv('../train.csv',nrows = 600)
#X_test = X_test.to_numpy()[500:600]
#y_test = X_test[:,0].tolist()

predictions = (fit(X, y, X_test, y_test, 0.25, 100))
cm = (confusion_matrix(y_test, predictions))

# plotting
sns.heatmap(cm, annot=True,fmt='d', cmap='YlGnBu')
plt.ylabel('Prediction',fontsize=12)
plt.xlabel('Actual',fontsize=12)
plt.title('Confusion Matrix',fontsize=16)
plt.show()

print('y_test',y_test)
print('y_pred',predictions)
print('accuracy', accuracy_score(y_test, predictions))
print('precision', precision_score(y_test, predictions, average=None))
print('recall', recall_score(y_test, predictions, average=None))
print('f1', f1_score(y_test, predictions, average=None))

'''precision, recall, thresholds = precision_recall_curve(y_test,predictions)
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])'''
'''# NAIVE APPROACH ######################################################
for point in labelled_data:
    print('------------')

    # computing and sorting distances
    naiv_dst = {}
    for i in range(len(naiv_prototypes)):
        naiv_dst[i] = np.linalg.norm(naiv_prototypes[i][0] - point[0])
        #naiv_dst[i] = distance(naiv_prototypes[i][0], point[0])
    naiv_all_sorted = sorted(naiv_dst.items(), key=lambda x: x[1])
    point_lbl = point[1]

    if naiv_prototypes[naiv_all_sorted[0][0]][1] == point_lbl:
        naiv_IDX_COR = naiv_all_sorted[0][0]
        naiv_IDX_INCOR = naiv_all_sorted[1][0]
        naiv_prototype_correct = naiv_prototypes[naiv_all_sorted[0][0]][0]
        naiv_prototype_incorrect = naiv_prototypes[naiv_all_sorted[1][0]][0]
    else:
        #print('False')
        for naiv_prt in naiv_all_sorted:
            if naiv_prototypes[naiv_prt[0]][1] == point_lbl:
                naiv_IDX_COR = naiv_prt[0]
                naiv_IDX_INCOR = naiv_all_sorted[0][0]
                naiv_prototype_correct = naiv_prototypes[naiv_prt[0]][0]
                naiv_prototype_incorrect = naiv_prototypes[naiv_all_sorted[0][0]][0]
                break

    # initialisation
    instance = point[0]
    counter = 0

    # the lvq loop
    # correct prototype
    while counter < 10:
        # correct prototype
        #term = ((2 * norm(instance - naiv_prototype_incorrect)) /((norm(instance - naiv_prototype_correct) + norm(instance - naiv_prototype_incorrect)) ** 2))
        term = (norm(instance - naiv_prototype_incorrect)) / (norm(instance - naiv_prototype_correct) + norm(instance - naiv_prototype_incorrect)) ** 2
        upd_cor = monotone(mu2(instance, prototype_correct, prototype_incorrect)) * term * (instance - naiv_prototype_correct) #mu is wrong
        naiv_prototype_correct = naiv_prototype_correct - 0.5 * upd_cor
        #print('CORRECT| norm: %.8f'% norm(prototype_correct), 'distance: %.3f,' % distance(prototype_correct,instance))
        counter += 1
    # something is missing here
    naiv_prototypes[naiv_IDX_COR][0] = naiv_prototype_correct

    counter = 0

    # incorrect prototype
    while counter < 10:
        term = (norm(instance - naiv_prototype_incorrect)) / (norm(instance - naiv_prototype_correct) + norm(instance - naiv_prototype_incorrect)) ** 2
        upd = monotone(mu2(instance,prototype_correct,prototype_incorrect)) * term * (instance - naiv_prototype_incorrect)
        naiv_prototype_incorrect = naiv_prototype_incorrect + 0.5 * upd
        #print('INCORRECT| norm: %.8f'% norm(prototype_incorrect), 'distance: %.3f,' % distance(prototype_incorrect,instance))
        counter += 1

    naiv_prototypes[naiv_IDX_INCOR][0] = naiv_prototype_incorrect

# TESTING ###########################
naiv_dis = {}
naiv_res = []

for point in labelled_data_test:
    exam = point[0]
    true_label = point[1]
    for j in range(len(naiv_prototypes)):
        naiv_dis[j] = np.linalg.norm(exam - naiv_prototypes[j][0])
    naiv_which = min(naiv_dis, key=naiv_dis.get)

    naiv_res.append(true_label == naiv_prototypes[naiv_which][1])

print('naiv:', naiv_res.count(True)/len(naiv_res)*100)'''