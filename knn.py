import numpy as np

def distance(x1,x2):
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    return np.sqrt(((x1-x2)**2).sum())

def knn(x_train,labels,x,k=3):
    retval = []
    x_train = np.asarray(x_train)
    for data in x:
        vals = []
        dist = []
        for ix in range(0, x_train.shape[0]):
            dist.append(distance(x_train[ix], x))
            vals.append([dist[ix], labels[ix]])

        vals_sorted = sorted(vals, key=lambda x: x[0])
        pred_array = np.asarray(vals_sorted[:k])
        pred_array = np.unique(pred_array[:, 1], return_counts=True)
        print pred_array
        index = pred_array[1].argmax()
        retval.append(pred_array[0][index])


    return retval

