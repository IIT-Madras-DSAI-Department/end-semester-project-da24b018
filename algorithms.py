import numpy as np
import pandas as pd
import time
from collections import Counter

# --------------------------
# Load MNIST CSVs
# --------------------------
def load_mnist_csv(path):
    data = pd.read_csv(path)
    y = data.iloc[:, 0].values
    X = data.iloc[:, 1:].values.astype(np.float32) / 255.0
    return X, y

# Load training and validation data
X_train, y_train = load_mnist_csv("MNIST_train.csv")
X_val, y_val = load_mnist_csv("MNIST_validation.csv")

print("Train:", X_train.shape, "Val:", X_val.shape)

# --------------------------
# Metrics
# --------------------------
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision_recall_f1(y_true, y_pred):
    classes = np.unique(np.concatenate([y_true, y_pred]))
    precisions, recalls, f1s = [], [], []
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
    return np.mean(precisions), np.mean(recalls), np.mean(f1s)

# SOFTMAX (MULTINOMIAL LOGISTIC REGRESSION)

class SoftmaxRegression:
    def __init__(self, learning_rate=0.1, epochs=200):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.W = None
        self.b = None

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _one_hot(self, y, n_classes):
        return np.eye(n_classes)[y]

    def fit(self, X, y, verbose=False):
        n, d = X.shape
        n_classes = np.max(y) + 1
        self.W = np.zeros((d, n_classes))
        self.b = np.zeros((1, n_classes))
        Y = self._one_hot(y, n_classes)

        for epoch in range(self.epochs):
            logits = X @ self.W + self.b
            probs = self._softmax(logits)
            grad_W = X.T @ (probs - Y) / n
            grad_b = np.mean(probs - Y, axis=0, keepdims=True)
            self.W -= self.learning_rate * grad_W
            self.b -= self.learning_rate * grad_b
            if verbose and epoch % 50 == 0:
                loss = -np.mean(np.sum(Y * np.log(probs + 1e-9), axis=1))
                print(f"Epoch {epoch}: loss={loss:.4f}")

    def predict(self, X):
        logits = X @ self.W + self.b
        probs = self._softmax(logits)
        return np.argmax(probs, axis=1)

# --------------------------
# Hyperparameter tuning
# --------------------------
lrs = [0.05, 0.1, 0.2]
epochs_list = [100, 200, 400]
best_acc = 0
best_params = None

for lr in lrs:
    for ep in epochs_list:
        model = SoftmaxRegression(learning_rate=lr, epochs=ep)
        start = time.time()
        model.fit(X_train[:4000], y_train[:4000])
        preds = model.predict(X_val[:2000])
        acc = accuracy(y_val[:2000], preds)
        prec, rec, f1 = precision_recall_f1(y_val[:2000], preds)
        t = time.time() - start
        print(f"LR={lr}, Epochs={ep} : Acc={acc:.4f}, F1={f1:.4f}, Time={t:.2f}s")
        if acc > best_acc:
            best_acc = acc
            best_params = (lr, ep)

print("\nBest Softmax Params:", best_params, "Accuracy:", best_acc)

# --------------------------
# Hyperparameter tuning
# --------------------------
lrs = [0.05, 0.1, 0.2]
epochs_list = [500, 750, 1000]
best_acc = 0
best_params = None

for lr in lrs:
    for ep in epochs_list:
        model = SoftmaxRegression(learning_rate=lr, epochs=ep)
        start = time.time()
        model.fit(X_train[:4000], y_train[:4000])
        preds = model.predict(X_val[:2000])
        acc = accuracy(y_val[:2000], preds)
        prec, rec, f1 = precision_recall_f1(y_val[:2000], preds)
        t = time.time() - start
        print(f"LR={lr}, Epochs={ep} : Acc={acc:.4f}, F1={f1:.4f}, Time={t:.2f}s")
        if acc > best_acc:
            best_acc = acc
            best_params = (lr, ep)

print("\nBest Softmax Parameters:", best_params, "Accuracy:", best_acc)

# Softmax Regression (Improved Version)
class SoftmaxRegressionImproved:
    def __init__(self, learning_rate=0.1, epochs=200,
                 batch_size=128, l2=0.0, lr_decay=0.0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.lr_decay = lr_decay
        self.W2 = None
        self.b2 = None

    def _softmax2(self, z):
        z -= np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _one_hot2(self, y, n_classes):
        return np.eye(n_classes)[y]

    def fit(self, X, y, verbose=False):
        n, d = X.shape
        n_classes = np.max(y) + 1

        self.W2 = np.zeros((d, n_classes))
        self.b2 = np.zeros((1, n_classes))
        Y = self._one_hot2(y, n_classes)

        indices = np.arange(n)

        for epoch in range(self.epochs):

            # Learning rate decay
            lr_t = self.learning_rate / (1 + self.lr_decay * epoch)

            np.random.shuffle(indices)
            X = X[indices]
            Y = Y[indices]

            for i in range(0, n, self.batch_size):
                xb = X[i:i+self.batch_size]
                yb = Y[i:i+self.batch_size]

                logits = xb @ self.W2 + self.b2
                probs = self._softmax2(logits)

                grad_W = xb.T @ (probs - yb) / xb.shape[0]
                grad_b = np.mean(probs - yb, axis=0, keepdims=True)

                # L2 regularization
                grad_W += self.l2 * self.W2

                self.W2 -= lr_t * grad_W
                self.b2 -= lr_t * grad_b

            if verbose and epoch % 100 == 0:
                preds_all = self._softmax2(X @ self.W2 + self.b2)
                loss = -np.mean(np.sum(Y * np.log(preds_all + 1e-9), axis=1))
                print(f"Epoch {epoch}: loss={loss:.4f}")

    def predict(self, X):
        logits = X @ self.W2 + self.b2
        probs = self._softmax2(logits)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X):
        logits = X @ self.W2 + self.b2
        return self._softmax2(logits)


# ============================================================
# Hyperparameter Tuning
# ============================================================
lrs2 = [0.2, 0.3]
epochs_list2 = [1500]
batch_sizes2 = [128, 256]
l2_list2 = [1e-4, 1e-3]
decay_list2 = [0.0, 1e-4]

best_acc2 = 0
best_params2 = None

for lr in lrs2:
    for ep in epochs_list2:
        for bs in batch_sizes2:
            for l2 in l2_list2:
                for decay in decay_list2:

                    model2 = SoftmaxRegressionImproved(
                        learning_rate=lr,
                        epochs=ep,
                        batch_size=bs,
                        l2=l2,
                        lr_decay=decay
                    )

                    start = time.time()
                    model2.fit(X_train[:5000], y_train[:5000])
                    preds = model2.predict(X_val[:2000])
                    acc = accuracy(y_val[:2000], preds)
                    prec, rec, f1 = precision_recall_f1(y_val[:2000], preds)
                    t = time.time() - start

                    print(f"LR={lr}, EP={ep}, BS={bs}, L2={l2}, DECAY={decay} : ACC={acc:.4f}, F1={f1:.4f}, Time={t:.2f}s")

                    if acc > best_acc2:
                        best_acc2 = acc
                        best_params2 = (lr, ep, bs, l2, decay)

print("\nBEST PARAMETERS :", best_params2, "BEST ACCURACY:", best_acc2)

# PCA MODEL

class PCAModel:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None

    def fit(self, X):
        X = np.array(X, dtype=float)

        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        cov_matrix = np.cov(X_centered, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        sorted_idx = np.argsort(eigenvalues)[::-1]

        self.explained_variance = eigenvalues[sorted_idx][:self.n_components]
        self.components = eigenvectors[:, sorted_idx][:, :self.n_components]

    def predict(self, X):
        if self.mean is None or self.components is None:
            raise ValueError("PCA model has not been fitted yet.")

        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

    def reconstruct(self, X):
        Z = self.predict(X)
        return np.dot(Z, self.components.T) + self.mean

    def detect_anomalies(self, X, threshold=None, return_errors=False):
        X_reconstructed = self.reconstruct(X)
        errors = np.mean((X - X_reconstructed) ** 2, axis=1)

        if threshold is None:
            threshold = np.percentile(errors, 95)

        is_anomaly = (errors > threshold).astype(int)

        if return_errors:
            return is_anomaly, errors
        return is_anomaly



# APPLY PCA BEFORE SOFTMAX

pca_components = 100
pca = PCAModel(n_components=pca_components)

pca.fit(X_train)

X_train_pca = pca.predict(X_train)
X_val_pca   = pca.predict(X_val)

print("Original shape:", X_train.shape)
print("PCA shape:", X_train_pca.shape)



# PCA + Softmax Hyperparameter Tuning (with PCA components)

pca_comp_list = [80, 100, 120]

lrs = [0.2, 0.3]
epochs_list = [1500]
batch_sizes = [128, 256]
l2_list = [1e-4, 1e-3]
decay_list = [0.0, 1e-4]

best_acc_all = 0
best_full_params = None

for pca_comp in pca_comp_list:

    print(f"\nFITTING PCA WITH {pca_comp} COMPONENTS")
    pca = PCAModel(n_components=pca_comp)
    pca.fit(X_train)

    X_train_pca = pca.predict(X_train)
    X_val_pca   = pca.predict(X_val)

    print("PCA shape:", X_train_pca.shape)

    # Softmax tuning on PCA representation
    for lr in lrs:
        for ep in epochs_list:
            for bs in batch_sizes:
                for l2 in l2_list:
                    for decay in decay_list:

                        model2 = SoftmaxRegressionImproved(
                            learning_rate=lr,
                            epochs=ep,
                            batch_size=bs,
                            l2=l2,
                            lr_decay=decay
                        )

                        start = time.time()

                        # Train on first 5000 PCA samples
                        model2.fit(X_train_pca[:5000], y_train[:5000])

                        preds = model2.predict(X_val_pca[:2000])
                        acc = accuracy(y_val[:2000], preds)
                        prec, rec, f1 = precision_recall_f1(y_val[:2000], preds)
                        t = time.time() - start

                        print(f"PCA={pca_comp}, LR={lr}, EP={ep}, BS={bs}, L2={l2}, DECAY={decay} : ACC={acc:.4f}, Time={t:.2f}s")

                        if acc > best_acc_all:
                            best_acc_all = acc
                            best_full_params = (pca_comp, lr, ep, bs, l2, decay)

print("\nBEST PARAMETERS:", best_full_params)
print("BEST ACCURACY:", best_acc_all)

# DECISION TREE CLASSIFIER

class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def entropy(self, y):
        if len(y) == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return -np.sum(probs * np.log2(probs))

    def best_split(self, X, y):
        n, d = X.shape
        best_gain = 0
        best_feature, best_threshold = None, None
        base_entropy = self.entropy(y)
        for feature in range(d):
            thresholds = np.unique(X[:, feature])
            for t in thresholds:
                left_mask = X[:, feature] <= t
                right_mask = ~left_mask
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                y_left, y_right = y[left_mask], y[right_mask]
                gain = base_entropy - (
                    len(y_left)/n * self.entropy(y_left)
                    + len(y_right)/n * self.entropy(y_right)
                )
                if gain > best_gain:
                    best_gain, best_feature, best_threshold = gain, feature, t
        return best_feature, best_threshold

    def build_tree(self, X, y, depth=0):
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) < self.min_samples_split:
            return Counter(y).most_common(1)[0][0]
        feat, thr = self.best_split(X, y)
        if feat is None:
            return Counter(y).most_common(1)[0][0]
        left_mask = X[:, feat] <= thr
        right_mask = ~left_mask
        return {
            'feature': feat,
            'threshold': thr,
            'left': self.build_tree(X[left_mask], y[left_mask], depth+1),
            'right': self.build_tree(X[right_mask], y[right_mask], depth+1)
        }

    def fit(self, X, y):
        self.tree = self.build_tree(X, y)

    def predict_one(self, x, node=None):
        if not isinstance(node, dict):
            return node
        if x[node['feature']] <= node['threshold']:
            return self.predict_one(x, node['left'])
        else:
            return self.predict_one(x, node['right'])

    def predict(self, X):
        return np.array([self.predict_one(x, self.tree) for x in X])

# --------------------------
# Hyperparameter tuning
# --------------------------
depths = [5, 10, 15]
min_splits = [2, 5, 10]
best_acc = 0
best_params = None

for d in depths:
    for m in min_splits:
        dt = DecisionTree(max_depth=d, min_samples_split=m)
        start = time.time()
        dt.fit(X_train[:2000], y_train[:2000])
        preds = dt.predict(X_val[:1000])
        acc = accuracy(y_val[:1000], preds)
        prec, rec, f1 = precision_recall_f1(y_val[:1000], preds)
        t = time.time() - start
        print(f"Depth={d}, MinSplit={m} : Acc={acc:.4f}, F1={f1:.4f}, Time={t:.2f}s")
        if acc > best_acc:
            best_acc = acc
            best_params = (d, m)

print("\nBest DecisionTree Params:", best_params, "Accuracy:", best_acc)

# RANDOM FOREST (Bagging Ensemble)

class RandomForest:
    def __init__(self, n_estimators=5, max_depth=10, sample_ratio=0.8):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.sample_ratio = sample_ratio
        self.trees = []

    def fit(self, X, y):
        n = X.shape[0]
        self.trees = []
        for _ in range(self.n_estimators):
            idx = np.random.choice(n, int(n * self.sample_ratio), replace=True)
            Xb, yb = X[idx], y[idx]
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(Xb, yb)
            self.trees.append(tree)

    def predict(self, X):
        preds = np.array([tree.predict(X) for tree in self.trees])
        # majority voting
        final = [Counter(preds[:, i]).most_common(1)[0][0] for i in range(X.shape[0])]
        return np.array(final)

# --------------------------
# Hyperparameter tuning
# --------------------------
n_estimators_list = [3, 5, 8]
depth_list = [8, 10]
best_acc = 0
best_params = None

for n_est in n_estimators_list:
    for depth in depth_list:
        rf = RandomForest(n_estimators=n_est, max_depth=depth)
        start = time.time()
        rf.fit(X_train[:2000], y_train[:2000])
        preds = rf.predict(X_val[:1000])
        acc = accuracy(y_val[:1000], preds)
        prec, rec, f1 = precision_recall_f1(y_val[:1000], preds)
        t = time.time() - start
        print(f"Trees={n_est}, Depth={depth} : Acc={acc:.4f}, F1={f1:.4f}, Time={t:.2f}s")
        if acc > best_acc:
            best_acc = acc
            best_params = (n_est, depth)

print("\nBest RandomForest Parameters:", best_params, "Accuracy:", best_acc)

# PCA TO SPEED UP BOOSTING AND STACKING

def PCA(X, k):
    X_centered = X - np.mean(X, axis=0)
    C = (X_centered.T @ X_centered) / X_centered.shape[0]
    eigvals, eigvecs = np.linalg.eigh(C)
    idx = np.argsort(eigvals)[::-1]
    W = eigvecs[:, idx[:k]]
    return X_centered @ W, W

X_train_pca, W_pca = PCA(X_train, k=20)
X_val_pca = (X_val - np.mean(X_train, axis=0)) @ W_pca

print("PCA shapes -> Train:", X_train_pca.shape, "Val:", X_val_pca.shape)


# MULTI-CLASS GRADIENT BOOSTING

class GradientBoosting:
    def __init__(self, n_estimators=5, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.models = []
        self.n_classes = None

    def _one_hot(self, y):
        return np.eye(self.n_classes)[y]

    def _softmax(self, F):
        e = np.exp(F - np.max(F, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)

    def fit(self, X, y):
        n = X.shape[0]
        self.n_classes = np.max(y) + 1
        Y = self._one_hot(y)
        F = np.zeros((n, self.n_classes))
        self.models = []

        for _ in range(self.n_estimators):
            P = self._softmax(F)
            R = Y - P
            trees = []

            # Train 1 tree per class
            for c in range(self.n_classes):
                tree = DecisionTree(max_depth=self.max_depth)
                tree.fit(X, R[:, c])
                trees.append(tree)

            # Update logits
            for c in range(self.n_classes):
                F[:, c] += self.learning_rate * trees[c].predict(X)

            self.models.append(trees)

    def predict(self, X):
        n = X.shape[0]
        F = np.zeros((n, self.n_classes))

        for trees in self.models:
            for c in range(self.n_classes):
                F[:, c] += self.learning_rate * trees[c].predict(X)

        return np.argmax(F, axis=1)


# Hyperparameter Tuning for Gradient Boosting

estimators_list = [3, 4, 5]
lrs = [0.05, 0.1, 0.2]
depths = [2, 3]

best_acc = 0
best_params = None


for depth in depths:
    for n in estimators_list:
        for lr in lrs:

            gb = GradientBoosting(n_estimators=n, learning_rate=lr, max_depth=depth)

            # adaptive subsampling size based on depth
            if depth == 2:
                train_subset = 1000
                val_subset = 600
            else:
                train_subset = 800
                val_subset = 500

            start = time.time()
            gb.fit(X_train_pca[:train_subset], y_train[:train_subset])
            preds = gb.predict(X_val_pca[:val_subset])

            acc = accuracy(y_val[:val_subset], preds)
            prec, rec, f1 = precision_recall_f1(y_val[:val_subset], preds)
            t = time.time() - start

            print(f"Depth={depth}, Est={n}, LR={lr} → "
                  f"Acc={acc:.4f}, F1={f1:.4f}, Time={t:.2f}s")

            # Track best
            if acc > best_acc:
                best_acc = acc
                best_params = (depth, n, lr)

print("\nBest Gradient Boosting Parameters:", best_params, "Accuracy:", best_acc)

# Optimized Tuning for Gradient Boosting (PCA = 30 only)

pca_k = 30
depths = [3, 4]
estimators_list = [3, 4]
lrs = [0.1, 0.2]

best_acc = 0
best_params = None

print(f"\nRunning PCA with k={pca_k}")
X_train_pca, W_pca = PCA(X_train, k=pca_k)
X_val_pca = (X_val - np.mean(X_train, axis=0)) @ W_pca

for depth in depths:
    for n in estimators_list:
        for lr in lrs:

            gb = GradientBoosting(
                n_estimators=n,
                learning_rate=lr,
                max_depth=depth
            )

            train_subset = 600
            val_subset = 400

            start = time.time()
            gb.fit(X_train_pca[:train_subset], y_train[:train_subset])
            preds = gb.predict(X_val_pca[:val_subset])

            acc = accuracy(y_val[:val_subset], preds)
            prec, rec, f1 = precision_recall_f1(y_val[:val_subset], preds)
            t = time.time() - start

            print(f"PCA={pca_k}, Depth={depth}, Est={n}, LR={lr} → "
                  f"Acc={acc:.4f}, F1={f1:.4f}, Time={t:.2f}s")

            if acc > best_acc:
                best_acc = acc
                best_params = (pca_k, depth, n, lr)

print("\nBEST BOOSTING PARAMETERS:")
print(best_params, "Accuracy:", best_acc)

# Optimized Tuning for Gradient Boosting

pca_list = [35, 40]
depth_list = [4, 5]
estimators_list = [2, 3]
lr_list = [0.25, 0.30]

best_acc = 0
best_params = None

for pca_k in pca_list:

    print(f"Running PCA with k={pca_k}")
    X_train_pca, W_pca = PCA(X_train, k=pca_k)
    X_val_pca = (X_val - np.mean(X_train, axis=0)) @ W_pca

    train_subset = 600
    val_subset = 400

    for depth in depth_list:
        for est in estimators_list:
            for lr in lr_list:

                gb = GradientBoosting(
                    n_estimators=est,
                    learning_rate=lr,
                    max_depth=depth
                )

                start = time.time()
                gb.fit(X_train_pca[:train_subset], y_train[:train_subset])
                preds = gb.predict(X_val_pca[:val_subset])

                acc = accuracy(y_val[:val_subset], preds)
                prec, rec, f1 = precision_recall_f1(y_val[:val_subset], preds)
                t = time.time() - start

                print(f"PCA={pca_k}, Depth={depth}, Est={est}, LR={lr} : Acc={acc:.4f}, F1={f1:.4f}, Time={t:.2f}s")

                if acc > best_acc:
                    best_acc = acc
                    best_params = (pca_k, depth, est, lr)

print("\nNEW BEST BOOSTING PARAMETERS:")
print(best_params, "Accuracy:", best_acc)

# ENSEMBLE TRAINING

# FAST SOFTMAX

softmax_fast = SoftmaxRegression(learning_rate=0.2, epochs=200)
softmax_fast.fit(X_train, y_train)
soft_pred = softmax_fast.predict(X_val)
print("Softmax (fast) Acc:", accuracy(y_val, soft_pred))

# FAST DECISION TREE

dt_fast = DecisionTree(max_depth=10, min_samples_split=2)
dt_fast.fit(X_train[:2000], y_train[:2000])
dt_pred = dt_fast.predict(X_val)
print("Decision Tree (fast) Acc:", accuracy(y_val, dt_pred))

# FAST RANDOM FOREST (3 trees only)

rf_fast = RandomForest(n_estimators=3, max_depth=8)
rf_fast.fit(X_train[:4000], y_train[:4000])
rf_pred = rf_fast.predict(X_val)
print("Random Forest (fast) Acc:", accuracy(y_val, rf_pred))

# FAST GRADIENT BOOSTING with PCA

X_train_pca_fast, W_fast = PCA(X_train, k=35)
X_val_pca_fast = (X_val - np.mean(X_train, axis=0)) @ W_fast

gb_fast = GradientBoosting(
    n_estimators=2,
    learning_rate=0.25,
    max_depth=4
)
gb_fast.fit(X_train_pca_fast[:2000], y_train[:2000])
gb_pred = gb_fast.predict(X_val_pca_fast)
print("Gradient Boosting (fast) Acc:", accuracy(y_val, gb_pred))

# STACKING ENSEMBLE
def one_hot(p, C=10):
    return np.eye(C)[p]

soft_oh = one_hot(soft_pred)
dt_oh   = one_hot(dt_pred)
rf_oh   = one_hot(rf_pred)
gb_oh   = one_hot(gb_pred)

X_meta = np.concatenate([soft_oh, dt_oh, rf_oh, gb_oh], axis=1)
y_meta = y_val

print("Meta feature shape:", X_meta.shape)


meta = SoftmaxRegression(learning_rate=0.2, epochs=150)
meta.fit(X_meta, y_meta)

final_pred = meta.predict(X_meta)

acc = accuracy(y_val, final_pred)
prec, rec, f1 = precision_recall_f1(y_val, final_pred)

print("\nFINAL ENSEMBLE RESULTS\n")
print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1:", f1)

# PCA + Polynomial Features + Softmax with tuning


pca_components_list = [40, 60]

def poly_features(X):
    return np.concatenate([X, X**2], axis=1)

lr_list = [0.1, 0.2, 0.3]
l2_list = [1e-4, 1e-3]
batch_list = [128, 256]

best_acc_poly = 0
best_params_poly = None

for pca_comp in pca_components_list:
    pca_poly = PCAModel(n_components=pca_comp)
    pca_poly.fit(X_train)

    X_train_pca_poly = pca_poly.predict(X_train)
    X_val_pca_poly   = pca_poly.predict(X_val)

    X_train_poly = poly_features(X_train_pca_poly)
    X_val_poly   = poly_features(X_val_pca_poly)

    print(f"\nPCA={pca_comp}, POLY SHAPE={X_train_poly.shape}")

    for lr in lr_list:
        for l2 in l2_list:
            for bs in batch_list:

                model_poly = SoftmaxRegressionImproved(
                    learning_rate=lr,
                    epochs=800,
                    batch_size=bs,
                    l2=l2,
                    lr_decay=0.0
                )

                start = time.time()
                model_poly.fit(X_train_poly[:6000], y_train[:6000])
                preds_poly = model_poly.predict(X_val_poly)

                acc_poly = accuracy(y_val, preds_poly)
                t_poly = time.time() - start

                print(f"PCA={pca_comp}, LR={lr}, L2={l2}, BS={bs} : ACC={acc_poly:.4f} , Time={t_poly:.2f}s")

                if acc_poly > best_acc_poly:
                    best_acc_poly = acc_poly
                    best_params_poly = (pca_comp, lr, l2, bs)

print("\nBEST PARAMETERS:", best_params_poly)
print("BEST ACCURACY:", best_acc_poly)

# PCA + KNN (with tuning)

pca_knn_list = [40, 60]

k_list = [3, 4, 5]

def knn_predict(X_train, y_train, X_test, k):
    y_pred = []
    for i in range(X_test.shape[0]):
        dists = np.sum((X_train - X_test[i])**2, axis=1)
        idx = np.argsort(dists)[:k]
        votes = Counter(y_train[idx]).most_common(1)[0][0]
        y_pred.append(votes)
    return np.array(y_pred)

best_acc_knn = 0
best_params_knn = None

for pca_comp in pca_knn_list:

    pca_knn = PCAModel(n_components=pca_comp)
    pca_knn.fit(X_train)

    X_train_knn = pca_knn.predict(X_train)
    X_val_knn   = pca_knn.predict(X_val)

    print(f"\nPCA={pca_comp} for KNN, shape={X_train_knn.shape}")

    for k in k_list:

        start = time.time()
        preds_knn = knn_predict(X_train_knn[:6000], y_train[:6000],
                                X_val_knn, k=k)

        acc_knn = accuracy(y_val, preds_knn)
        t_knn = time.time() - start

        print(f"PCA={pca_comp}, k={k} : ACC={acc_knn:.4f}, Time={t_knn:.2f}s")

        if acc_knn > best_acc_knn:
            best_acc_knn = acc_knn
            best_params_knn = (pca_comp, k)

print("\nBEST PARAMETERS:", best_params_knn)
print("BEST ACCURACY:", best_acc_knn)

# Predictions from best models
pred_knn_best = knn_predict(X_train_knn[:6000], y_train[:6000], X_val_knn, k=5)
pred_softmax_best = model_poly.predict(X_val_poly)

final_preds = []
for a, b in zip(pred_knn_best, pred_softmax_best):
    vote = Counter([a, b]).most_common(1)[0][0]
    final_preds.append(vote)

final_preds = np.array(final_preds)

acc_ensemble = accuracy(y_val, final_preds)
prec_e, rec_e, f1_e = precision_recall_f1(y_val, final_preds)

print(f"\nENSEMBLE ACCURACY = {acc_ensemble}")
print(f"ENSEMBLE F1 = {f1_e}")

# BEST PARAMETERS FROM RESULTS

best_pca_poly = 60
best_lr_poly = 0.2
best_l2_poly = 0.001
best_bs_poly = 256
best_epochs_poly = 800
best_pca_knn = 60
best_k_knn = 5

pca_final = PCAModel(n_components=best_pca_poly)
pca_final.fit(X_train)

X_train_pca_final = pca_final.predict(X_train)
X_val_pca_final   = pca_final.predict(X_val)


def poly_features(X):
    return np.concatenate([X, X**2], axis=1)

X_train_poly_final = poly_features(X_train_pca_final)
X_val_poly_final   = poly_features(X_val_pca_final)

model_poly_final = SoftmaxRegressionImproved(
    learning_rate=best_lr_poly,
    epochs=best_epochs_poly,
    batch_size=best_bs_poly,
    l2=best_l2_poly,
    lr_decay=0.0
)

model_poly_final.fit(X_train_poly_final[:6000], y_train[:6000])

preds_softmax = model_poly_final.predict(X_val_poly_final)
acc_softmax = accuracy(y_val, preds_softmax)

X_train_knn_final = X_train_pca_final
X_val_knn_final   = X_val_pca_final

# Efficient KNN prediction function
def knn_predict_fast(X_train, y_train, X_test, k):
    y_pred = []
    for i in range(X_test.shape[0]):
        d = np.sum((X_train - X_test[i])**2, axis=1)
        idx = np.argpartition(d, k)[:k]  # faster than argsort
        votes = Counter(y_train[idx]).most_common(1)[0][0]
        y_pred.append(votes)
    return np.array(y_pred)

preds_knn = knn_predict_fast(
    X_train_knn_final[:6000],
    y_train[:6000],
    X_val_knn_final,
    best_k_knn
)

acc_knn = accuracy(y_val, preds_knn)
print(f"KNN Accuracy = {acc_knn:.4f}")

def softmax_probs(model, X):
    logits = X @ model.W2 + model.b2
    z = logits - np.max(logits, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

probs_softmax = softmax_probs(model_poly_final, X_val_poly_final)

C = 10
probs_knn = np.zeros((len(preds_knn), C))
for i, c in enumerate(preds_knn):
    probs_knn[i, c] = 1.0

best_ens_acc = 0
best_alpha = 0

for alpha in np.linspace(0, 1, 21):
    ensemble_probs = alpha * probs_knn + (1 - alpha) * probs_softmax
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    acc = accuracy(y_val, ensemble_preds)

    if acc > best_ens_acc:
        best_ens_acc = acc
        best_alpha = alpha

print(f"\nBest Ensemble alpha = {best_alpha:.2f}")
print(f"Best Ensemble Accuracy = {best_ens_acc:.4f}")

best_pca = 60

pca_final = PCAModel(n_components=best_pca)
pca_final.fit(X_train)

X_train_pca = pca_final.predict(X_train)
X_val_pca   = pca_final.predict(X_val)

def poly2_features(X):
    return np.concatenate([X, X**2], axis=1)

X_train_poly = poly2_features(X_train_pca)
X_val_poly   = poly2_features(X_val_pca)

model_poly = SoftmaxRegressionImproved(
    learning_rate=0.2,
    epochs=1500,
    batch_size=256,
    l2=0.001,
    lr_decay=0.0
)

model_poly.fit(X_train_poly[:6000], y_train[:6000])
preds_soft = model_poly.predict(X_val_poly)
acc_soft = accuracy(y_val, preds_soft)

print(f"SoftmaxPoly Accuracy = {acc_soft:.4f}")

def knn_predict_fast(X_train, y_train, X_test, k):
    y_pred = []
    for i in range(X_test.shape[0]):
        d = np.sum((X_train - X_test[i])**2, axis=1)
        idx = np.argpartition(d, k)[:k]
        votes = Counter(y_train[idx]).most_common(1)[0][0]
        y_pred.append(votes)
    return np.array(y_pred)

preds_knn = knn_predict_fast(X_train_pca[:6000], y_train[:6000], X_val_pca, 5)
acc_knn = accuracy(y_val, preds_knn)

print(f"KNN Accuracy = {acc_knn:.4f}")

probs_soft = model_poly.predict_proba(X_val_poly)

C = 10
probs_knn = np.zeros((len(preds_knn), C))
for i, c in enumerate(preds_knn):
    probs_knn[i, c] = 1.0

best_alpha = 0
best_ens = 0

for alpha in np.linspace(0,1,41):
    probs = alpha*probs_knn + (1-alpha)*probs_soft
    preds = np.argmax(probs, axis=1)
    acc = accuracy(y_val, preds)

    if acc > best_ens:
        best_ens = acc
        best_alpha = alpha

print(f"\nBest Ensemble alpha = {best_alpha:.3f}")
print(f"Best Ensemble Accuracy = {best_ens:.4f}")

class KMeans:
    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4, random_state=0):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None

    def slow_compute_distances(self, X, centroids):
        distances = np.zeros((len(X), len(centroids)))
        for i, x in enumerate(X):
            for j, c in enumerate(centroids):
                distances[i, j] = np.linalg.norm(x - c)
        return distances

    def fit(self, X):
        X = np.array(X, dtype=float)
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape

        # initialize centroids randomly
        idx = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[idx]

        for _ in range(self.max_iter):
            # assign clusters
            distances = self.slow_compute_distances(X, self.centroids)
            labels = np.argmin(distances, axis=1)

            # compute new centroids
            new_centroids = np.zeros((self.n_clusters, n_features))
            for k in range(self.n_clusters):
                pts = X[labels == k]
                new_centroids[k] = pts.mean(axis=0) if len(pts) > 0 else self.centroids[k]

            # check convergence
            shift = np.linalg.norm(self.centroids - new_centroids)
            self.centroids = new_centroids
            if shift < self.tol:
                break

        self.labels_ = labels

    def predict(self, X):
        X = np.array(X, dtype=float)
        distances = self.slow_compute_distances(X, self.centroids)
        return np.argmin(distances, axis=1)

best_pca = 60

pca_final2 = PCAModel(n_components=best_pca)
pca_final2.fit(X_train)

X_train_pca2 = pca_final2.predict(X_train)
X_val_pca2   = pca_final2.predict(X_val)

kmeans = KMeans(n_clusters=10, max_iter=50, tol=1e-4)
kmeans.fit(X_train_pca2)

cluster_train = kmeans.labels_
cluster_val   = kmeans.predict(X_val_pca2)

def knn_predict_fast(X_train, y_train, X_test, k):
    preds = []
    for i in range(len(X_test)):
        d = np.sum((X_train - X_test[i])**2, axis=1)
        idx = np.argpartition(d, k)[:k]  # faster than argsort
        vote = Counter(y_train[idx]).most_common(1)[0][0]
        preds.append(vote)
    return np.array(preds)

preds_knn_std = knn_predict_fast(
    X_train_pca2[:6000], y_train[:6000],
    X_val_pca2,
    k=5
)

acc_knn_std = accuracy(y_val, preds_knn_std)
print(f"KNN Accuracy = {acc_knn_std:.4f}")

def knn_cluster_predict(X_train, y_train, X_test,
                        cluster_train, cluster_test, k):

    preds = []
    for i in range(len(X_test)):
        c = cluster_test[i]

        idx = np.where(cluster_train == c)[0]
        Xc, yc = X_train[idx], y_train[idx]

        d = np.sum((Xc - X_test[i])**2, axis=1)
        top = np.argpartition(d, k)[:k]
        vote = Counter(yc[top]).most_common(1)[0][0]
        preds.append(vote)

    return np.array(preds)

preds_knn_cluster = knn_cluster_predict(
    X_train_pca2[:6000], y_train[:6000],
    X_val_pca2,
    cluster_train[:6000],
    cluster_val,
    k=5
)

acc_knn_cluster = accuracy(y_val, preds_knn_cluster)
print(f"KNN Cluster Accuracy = {acc_knn_cluster:.4f}")

def softmax_prob(model, X):
    logits = X @ model.W2 + model.b2
    z = logits - np.max(logits, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)

probs_softmax2 = softmax_prob(model_poly_final, X_val_poly_final)
preds_softmax2 = np.argmax(probs_softmax2, axis=1)
acc_softmax2 = accuracy(y_val, preds_softmax2)

print(f"SoftmaxPoly Accuracy = {acc_softmax2:.4f}")

C = 10

probs_knn_std = np.zeros((len(preds_knn_std), C))
probs_knn_cluster = np.zeros((len(preds_knn_cluster), C))

for i, c in enumerate(preds_knn_std):
    probs_knn_std[i, c] = 1.0

for i, c in enumerate(preds_knn_cluster):
    probs_knn_cluster[i, c] = 1.0

best_acc = 0
best_w1 = best_w2 = best_w3 = 0

weights = np.linspace(0, 1, 11)

for w1 in weights:
    for w2 in weights:
        w3 = 1 - w1 - w2
        if w3 < 0:
            continue

        probs = (
            w1 * probs_softmax2 +
            w2 * probs_knn_std +
            w3 * probs_knn_cluster
        )

        preds = np.argmax(probs, axis=1)
        acc = accuracy(y_val, preds)

        if acc > best_acc:
            best_acc = acc
            best_w1, best_w2, best_w3 = w1, w2, w3

print(f"\nBest Ensemble Weights:")
print(f" Softmax = {best_w1:.2f}")
print(f" KNN Std = {best_w2:.2f}")
print(f" KNN Cluster = {best_w3:.2f}")

print(f"\n Final Ensemble Accuracy = {best_acc:.4f}")

def knn_predict_local_weighted(X_train, y_train, X_test, k, sigma=0.5):
    preds = []
    for i in range(len(X_test)):
        d = np.sum((X_train - X_test[i])**2, axis=1)

        idx = np.argpartition(d, k)[:k]
        d_top = d[idx]
        y_top = y_train[idx]

        w = np.exp(-d_top / (2 * sigma * sigma))

        class_scores = np.zeros(10)
        for cls, weight in zip(y_top, w):
            class_scores[cls] += weight

        preds.append(np.argmax(class_scores))

    return np.array(preds)

feature_std = np.std(X_train_pca2, axis=0) + 1e-6

X_train_scaled = X_train_pca2 / feature_std
X_val_scaled   = X_val_pca2 / feature_std

preds_knn_lw = knn_predict_local_weighted(
    X_train_pca2[:6000],
    y_train[:6000],
    X_val_pca2,
    k=5,
    sigma=0.4,
)

acc_knn_lw = accuracy(y_val, preds_knn_lw)
print(f"LW-KNN Accuracy = {acc_knn_lw:.4f}")

preds_knn_metric = knn_predict_fast(
    X_train_scaled[:6000],
    y_train[:6000],
    X_val_scaled,
    k=5
)

acc_knn_metric = accuracy(y_val, preds_knn_metric)
print(f"Metric KNN Accuracy = {acc_knn_metric:.4f}")


def preds_to_probs(preds, C=10):
    P = np.zeros((len(preds), C))
    for i, c in enumerate(preds):
        P[i, c] = 1.0
    return P

probs_softmax2 = softmax_prob(model_poly_final, X_val_poly_final)
probs_knn_std      = preds_to_probs(preds_knn_std)
probs_knn_cluster  = preds_to_probs(preds_knn_cluster)
probs_knn_lw       = preds_to_probs(preds_knn_lw)
probs_knn_metric   = preds_to_probs(preds_knn_metric)

best_acc_final = 0
best_w = None

weights = np.linspace(0, 1, 6)

for w1 in weights:
    for w2 in weights:
        for w3 in weights:
            for w4 in weights:
                w5 = 1 - (w1 + w2 + w3 + w4)
                if w5 < 0:
                    continue

                ensemble = (
                    w1 * probs_softmax2 +
                    w2 * probs_knn_std +
                    w3 * probs_knn_cluster +
                    w4 * probs_knn_lw +
                    w5 * probs_knn_metric
                )

                preds = np.argmax(ensemble, axis=1)
                acc = accuracy(y_val, preds)

                if acc > best_acc_final:
                    best_acc_final = acc
                    best_w = (w1, w2, w3, w4, w5)

print("\nBest Ensemble Weights:")
print(f"Softmax      = {best_w[0]:.2f}")
print(f"KNN Std      = {best_w[1]:.2f}")
print(f"KNN Cluster  = {best_w[2]:.2f}")
print(f"LW-KNN       = {best_w[3]:.2f}")
print(f"Metric KNN   = {best_w[4]:.2f}")

print(f"\n FINAL 5-MODEL ENSEMBLE ACCURACY = {best_acc_final:.4f}")