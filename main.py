import numpy as np
import pandas as pd
import time
from collections import Counter

# LOAD MNIST DATA

def load_mnist_csv(path):
    data = pd.read_csv(path)
    y = data.iloc[:, 0].values
    X = data.iloc[:, 1:].values.astype(np.float32) / 255.0
    return X, y

X_train, y_train = load_mnist_csv("MNIST_train.csv")
X_val, y_val = load_mnist_csv("MNIST_validation.csv")

print("Train:", X_train.shape, "Val:", X_val.shape)

# METRICS

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision_recall_f1(y_true, y_pred):
    classes = np.unique(np.concatenate([y_true, y_pred]))
    precisions, recalls, f1s = [], [], []
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
    return np.mean(precisions), np.mean(recalls), np.mean(f1s)

# PCA Model

class PCAModel:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None

    def fit(self, X):
        X = np.array(X, dtype=float)
        self.mean = X.mean(axis=0)
        Xc = X - self.mean
        cov = np.cov(Xc, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1]
        self.components = eigvecs[:, idx[:self.n_components]]

    def predict(self, X):
        return (X - self.mean) @ self.components

# Softmax Regression

class SoftmaxRegression:
    def __init__(self, learning_rate=0.1, epochs=200,
                 batch_size=128, l2=0.0, lr_decay=0.0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.lr_decay = lr_decay
        self.W2 = None
        self.b2 = None

    def _softmax(self, z):
        z -= np.max(z, axis=1, keepdims=True)
        e = np.exp(z)
        return e / np.sum(e, axis=1, keepdims=True)

    def _one_hot(self, y, C):
        return np.eye(C)[y]

    def fit(self, X, y):
        n, d = X.shape
        C = np.max(y) + 1
        Y = self._one_hot(y, C)

        self.W2 = np.zeros((d, C))
        self.b2 = np.zeros((1, C))
        idx = np.arange(n)

        for ep in range(self.epochs):
            lr_t = self.learning_rate / (1 + self.lr_decay * ep)

            np.random.shuffle(idx)
            X = X[idx]
            Y = Y[idx]

            for i in range(0, n, self.batch_size):
                xb = X[i:i+self.batch_size]
                yb = Y[i:i+self.batch_size]

                logits = xb @ self.W2 + self.b2
                probs = self._softmax(logits)

                gradW = xb.T @ (probs - yb) / xb.shape[0]
                gradb = np.mean(probs - yb, axis=0, keepdims=True)

                gradW += self.l2 * self.W2

                self.W2 -= lr_t * gradW
                self.b2 -= lr_t * gradb

    def predict(self, X):
        return np.argmax(self._softmax(X @ self.W2 + self.b2), axis=1)

    def predict_proba(self, X):
        return self._softmax(X @ self.W2 + self.b2)

# KMeans Clustering

class KMeans:
    def __init__(self, n_clusters=10, max_iter=100, tol=1e-4, random_state=0):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None

    def slow_dist(self, X, C):
        D = np.zeros((len(X), len(C)))
        for i, x in enumerate(X):
            for j, c in enumerate(C):
                D[i, j] = np.linalg.norm(x - c)
        return D

    def fit(self, X):
        X = np.array(X, float)
        np.random.seed(self.random_state)
        n = len(X)

        self.centroids = X[np.random.choice(n, self.n_clusters, replace=False)]

        for _ in range(self.max_iter):
            D = self.slow_dist(X, self.centroids)
            labels = np.argmin(D, axis=1)

            newC = np.zeros_like(self.centroids)
            for k in range(self.n_clusters):
                pts = X[labels == k]
                newC[k] = pts.mean(axis=0) if len(pts) else self.centroids[k]

            shift = np.linalg.norm(self.centroids - newC)
            self.centroids = newC
            if shift < self.tol:
                break

        self.labels_ = labels

    def predict(self, X):
        return np.argmin(self.slow_dist(X, self.centroids), axis=1)
    

def knn_predict_fast(Xt, yt, Xq, k):
    preds = []
    for x in Xq:
        d = np.sum((Xt - x)**2, axis=1)
        idx = np.argpartition(d, k)[:k]
        preds.append(Counter(yt[idx]).most_common(1)[0][0])
    return np.array(preds)


def knn_cluster_predict(Xt, yt, Xq, cl_train, cl_val, k):
    preds = []
    for i, x in enumerate(Xq):
        c = cl_val[i]
        idxs = np.where(cl_train == c)[0]

        if len(idxs) < k:
            preds.append(knn_predict_fast(Xt, yt, [x], k)[0])
            continue

        Xc = Xt[idxs]
        yc = yt[idxs]
        d = np.sum((Xc - x)**2, axis=1)
        top = np.argpartition(d, k)[:k]
        preds.append(Counter(yc[top]).most_common(1)[0][0])

    return np.array(preds)


def knn_predict_local_weighted(Xt, yt, Xq, k, sigma):
    preds = []
    for x in Xq:
        d = np.sum((Xt - x)**2, axis=1)
        idx = np.argpartition(d, k)[:k]
        d_top = d[idx]
        y_top = yt[idx]

        w = np.exp(-d_top / (2*sigma*sigma))

        scores = np.zeros(10)
        for cls, wt in zip(y_top, w):
            scores[cls] += wt

        preds.append(scores.argmax())
    return np.array(preds)


def preds_to_probs(preds, C=10):
    P = np.zeros((len(preds), C))
    for i, c in enumerate(preds):
        P[i, c] = 1
    return P



# Best parameters:
PCA_COMP = 60
K = 5
LR = 0.2
L2 = 0.001
BS = 256
EPOCHS = 1500
SIGMA = 0.4
CLUSTERS = 10


# PCA
pca = PCAModel(PCA_COMP)
pca.fit(X_train)
X_train_pca = pca.predict(X_train)
X_val_pca   = pca.predict(X_val)


# Polynomial Features (degree 2) 
def poly2(X):
    return np.concatenate([X, X*X], axis=1)

X_train_poly = poly2(X_train_pca)
X_val_poly   = poly2(X_val_pca)


# Softmax
print("\nTraining Softmax...")
softmax = SoftmaxRegression(
    learning_rate=LR,
    epochs=EPOCHS,
    batch_size=BS,
    l2=L2
)
softmax.fit(X_train_poly[:6000], y_train[:6000])
pred_soft = softmax.predict(X_val_poly)
acc_soft = accuracy(y_val, pred_soft)
print("Softmax Accuracy:", acc_soft)


# Standard KNN 
print("\nStandard KNN...")
pred_knn_std = knn_predict_fast(
    X_train_pca[:6000], y_train[:6000],
    X_val_pca, K
)
acc_knn_std = accuracy(y_val, pred_knn_std)
print("KNN Std Accuracy:", acc_knn_std)


# KMeans + Cluster KNN
print("\nRunning KMeans...")
km = KMeans(CLUSTERS)
km.fit(X_train_pca)
cl_train = km.labels_
cl_val   = km.predict(X_val_pca)

pred_knn_cluster = knn_cluster_predict(
    X_train_pca[:6000], y_train[:6000],
    X_val_pca, cl_train[:6000], cl_val,
    K
)
acc_knn_cluster = accuracy(y_val, pred_knn_cluster)
print("Cluster KNN Accuracy:", acc_knn_cluster)


# Local Weighted KNN
print("\nLocal Weighted KNN...")
pred_knn_lw = knn_predict_local_weighted(
    X_train_pca[:6000], y_train[:6000],
    X_val_pca, K, SIGMA
)
acc_knn_lw = accuracy(y_val, pred_knn_lw)
print("LW-KNN Accuracy:", acc_knn_lw)


# Metric KNN
print("\nMetric KNN (Scaled)...")
std = np.std(X_train_pca, axis=0) + 1e-6
X_train_scaled = X_train_pca / std
X_val_scaled   = X_val_pca / std

pred_knn_metric = knn_predict_fast(
    X_train_scaled[:6000], y_train[:6000],
    X_val_scaled, K
)
acc_knn_metric = accuracy(y_val, pred_knn_metric)
print("Metric KNN Accuracy:", acc_knn_metric)



# ENSEMBLE (5 MODEL WEIGHT SEARCH)

probs_soft = softmax.predict_proba(X_val_poly)
probs_knn_std = preds_to_probs(pred_knn_std)
probs_knn_cluster = preds_to_probs(pred_knn_cluster)
probs_knn_lw = preds_to_probs(pred_knn_lw)
probs_knn_metric = preds_to_probs(pred_knn_metric)

print("\nSearching best ensemble weights...")

best_acc = 0
best_w = None
weights = np.linspace(0, 1, 6)

for w1 in weights:
    for w2 in weights:
        for w3 in weights:
            for w4 in weights:
                w5 = 1 - (w1+w2+w3+w4)
                if w5 < 0:
                    continue

                ensemble = (
                    w1*probs_soft +
                    w2*probs_knn_std +
                    w3*probs_knn_cluster +
                    w4*probs_knn_lw +
                    w5*probs_knn_metric
                )

                preds = np.argmax(ensemble, axis=1)
                acc = accuracy(y_val, preds)

                if acc > best_acc:
                    best_acc = acc
                    best_w = (w1,w2,w3,w4,w5)

print("\nBest Ensemble Weights:")
print(" Softmax     =", best_w[0])
print(" KNN Std     =", best_w[1])
print(" KNN Clust   =", best_w[2])
print(" LW-KNN      =", best_w[3])
print(" Metric-KNN  =", best_w[4])

print("\n FINAL ENSEMBLE ACCURACY =", best_acc)

final_weights = np.array(best_w)

final_ensemble_probs = (
    final_weights[0] * probs_soft +
    final_weights[1] * probs_knn_std +
    final_weights[2] * probs_knn_cluster +
    final_weights[3] * probs_knn_lw +
    final_weights[4] * probs_knn_metric
)

final_predictions = np.argmax(final_ensemble_probs, axis=1)