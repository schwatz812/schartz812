import torch
import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from sklearn.decomposition import PCA  # Import PCA
class BPFCMClassifier:
    def __init__(self, n_clusters, n_labels, max_iter=100, burn_in=20, thin=2,
                 alpha0=1.0, nu0=None, S0=None, beta=None):
        self.n_clusters = n_clusters
        self.n_labels = n_labels
        self.max_iter = max_iter
        self.burn_in = burn_in
        self.thin = thin
        self.alpha0 = alpha0

        # Model parameters
        self.membership_matrix = None  # U
        self.cluster_centers = None  # V
        self.covariance_matrices = None  # Σ
        self.mixing_proportions = None  # π
        self.weights = None  # Linear connection weights for classification
        # MCMC samples
        self.retained_samples = []

    def _initialize_parameters(self, X):
        """Initialize model parameters"""
        n, p = X.shape

        # Initialize membership matrix with FCM (simplified)
        # In practice, you would use proper FCM initialization
        self.membership_matrix = torch.rand(n, self.n_clusters)
        self.membership_matrix = self.membership_matrix / torch.sum(self.membership_matrix, dim=1, keepdim=True)

        # Initialize cluster centers
        self.cluster_centers = torch.randn(self.n_clusters, p)

        # Initialize covariance matrices
        self.covariance_matrices = torch.stack([torch.eye(p) for _ in range(self.n_clusters)])

        # Initialize mixing proportions
        self.mixing_proportions = torch.ones(self.n_clusters) / self.n_clusters

        # Initialize weights for classification - create a tensor that requires gradients
        self.weights = torch.randn(self.n_clusters, self.n_labels).requires_grad_(True)

    def _update_membership(self, X):
        """Update membership matrix using Metropolis-Hastings"""
        n, _ = X.shape
        # Propose new membership matrix
        proposal_sigma = 0.05
        proposed_U = self.membership_matrix + proposal_sigma * torch.randn_like(self.membership_matrix)
        proposed_U = torch.clamp(proposed_U, 0, 1)
        proposed_U = proposed_U / torch.sum(proposed_U, dim=1, keepdim=True)

        # Calculate acceptance probability (simplified)
        # In practice, you would compute the full Metropolis-Hastings ratio
        current_log_likelihood = self._compute_log_likelihood(X, self.membership_matrix)
        proposed_log_likelihood = self._compute_log_likelihood(X, proposed_U)

        acceptance_prob = torch.min(torch.ones(n), torch.exp(proposed_log_likelihood - current_log_likelihood))

        # Accept or reject
        random_vals = torch.rand(n)
        accepted = random_vals < acceptance_prob

        # Update membership matrix
        new_U = self.membership_matrix.clone()
        new_U[accepted] = proposed_U[accepted]
        self.membership_matrix = new_U

    def _compute_log_likelihood(self, X, U):
        """Compute log-likelihood for the data given membership matrix"""
        n, p = X.shape
        log_likelihood = torch.zeros(n)

        for i in range(n):
            for k in range(self.n_clusters):
                diff = X[i] - self.cluster_centers[k]
                inv_cov = torch.inverse(self.covariance_matrices[k])
                mahalanobis = torch.dot(torch.matmul(diff, inv_cov), diff)
                log_det = torch.logdet(self.covariance_matrices[k])
                log_likelihood[i] += U[i, k] * (-0.5 * (p * np.log(2 * np.pi) + log_det + mahalanobis))

        return log_likelihood

    def _update_cluster_centers(self, X):
        """Update cluster centers using Gibbs sampling"""
        n, p = X.shape

        for k in range(self.n_clusters):
            inv_cov = torch.inverse(self.covariance_matrices[k])
            weighted_sum = torch.zeros(p)
            weight_total = 0

            for i in range(n):
                weighted_sum += self.membership_matrix[i, k] * torch.matmul(inv_cov, X[i])
                weight_total += self.membership_matrix[i, k]

            precision = weight_total * inv_cov
            mean = torch.matmul(torch.inverse(precision), weighted_sum)

            # Sample from multivariate normal
            L = torch.linalg.cholesky(torch.inverse(precision))
            z = torch.randn(p)
            self.cluster_centers[k] = mean + torch.matmul(L, z)

    def _update_covariance_matrices(self, X):
        """Update covariance matrices using Gibbs sampling (simplified)"""
        n, p = X.shape

        for k in range(self.n_clusters):
            scale_matrix = torch.eye(p)
            df = p + 2  # Minimal degrees of freedom

            for i in range(n):
                diff = X[i] - self.cluster_centers[k]
                outer_product = torch.outer(diff, diff)
                scale_matrix += self.membership_matrix[i, k] * outer_product

            # Sample from Inverse-Wishart (simplified)
            self.covariance_matrices[k] = scale_matrix / (df + n + p + 2)

    def _update_mixing_proportions(self):

        alpha = torch.ones(self.n_clusters)  # Prior

        for k in range(self.n_clusters):
            alpha[k] += torch.sum(self.membership_matrix[:, k])

        # Sample from Dirichlet
        self.mixing_proportions = torch.distributions.Dirichlet(alpha).sample()

    def _update_weights_newton(self, X, Y):
        learning_rate = 0.01
        reg_lambda = 0.01

        # Compute features from membership matrix
        features = self.membership_matrix

        # Detach weights from the computation graph for the update
        weights = self.weights.detach().clone()

        # Forward pass
        logits = torch.matmul(features, weights)
        probs = torch.sigmoid(logits)

        # Compute loss with Generalized Cross-Entropy
        gce_loss = -torch.mean((Y * torch.pow(probs, 0.7) + (1 - Y) * torch.pow(1 - probs, 0.7)) / 0.7)
        reg_loss = reg_lambda * torch.sum(weights ** 2)
        total_loss = gce_loss

        # Compute gradients
        grad = torch.zeros_like(weights)
        hessian = torch.zeros(self.n_clusters, self.n_labels, self.n_clusters, self.n_labels)

        batch_size = X.shape[0]

        # Compute gradient and Hessian for Newton's method
        for i in range(batch_size):
            for k in range(self.n_clusters):
                for l in range(self.n_labels):
                    # Gradient computation
                    error = probs[i, l] - Y[i, l]
                    grad[k, l] += features[i, k] * error / batch_size

                    # Hessian computation (approximation using IRLS)
                    for j in range(self.n_clusters):
                        hessian[k, l, j, l] += features[i, k] * features[i, j] * probs[i, l] * (
                                1 - probs[i, l]) / batch_size

        # Add regularization
        grad += 2 * reg_lambda * weights

        # Reshape Hessian for inversion
        H = hessian.reshape(self.n_clusters * self.n_labels, self.n_clusters * self.n_labels)
        G = grad.reshape(-1)

        # Add small value to diagonal for numerical stability
        H += torch.eye(H.shape[0]) * 1e-5

        # Newton update: w = w - H^(-1) * g
        try:
            H_inv = torch.inverse(H)
            update = torch.matmul(H_inv, G)
            update = update.reshape(self.n_clusters, self.n_labels)
            weights = weights - learning_rate * update
        except:
            # Fallback to gradient descent if Hessian is singular
            weights = weights - learning_rate * grad

        # Assign new weights (not in-place)
        self.weights = weights.clone().requires_grad_(True)

        return total_loss.item()

    def fit(self, X, Y):
        """Fit the model to the data"""
        X_tensor = torch.tensor(X, dtype=torch.float32)
        Y_tensor = torch.tensor(Y, dtype=torch.float32)

        n, p = X_tensor.shape

        # Initialize parameters
        self._initialize_parameters(X_tensor)

        # MCMC iterations
        for t in range(self.max_iter):
            # Update membership matrix
            self._update_membership(X_tensor)

            # Update cluster centers
            self._update_cluster_centers(X_tensor)

            # Update covariance matrices
            self._update_covariance_matrices(X_tensor)

            # Update mixing proportions
            self._update_mixing_proportions()

            # Update classification weights using Newton's method
            loss = self._update_weights_newton(X_tensor, Y_tensor)

            if t % 10 == 0:
                print(f"Iteration {t}, Loss: {loss:.4f}")

            # Store samples after burn-in, according to thinning interval
            if t >= self.burn_in and (t - self.burn_in) % self.thin == 0:
                sample = {
                    'U': self.membership_matrix.clone(),
                    'V': self.cluster_centers.clone(),
                    'Sigma': self.covariance_matrices.clone(),
                    'pi': self.mixing_proportions.clone(),
                    'weights': self.weights.clone()
                }
                self.retained_samples.append(sample)

        # Find MAP estimate
        self._find_map_estimate(X_tensor, Y_tensor)

        return self

    def _find_map_estimate(self, X, Y):
        """Find the MAP estimate from the retained samples"""
        if not self.retained_samples:
            return

        # Compute posterior for each sample
        posteriors = []
        for sample in self.retained_samples:
            # Compute log posterior (likelihood + prior)
            U = sample['U']
            V = sample['V']
            Sigma = sample['Sigma']
            pi = sample['pi']
            weights = sample['weights']

            # Data likelihood
            log_likelihood = torch.sum(self._compute_log_likelihood(X, U))

            # Prior terms (simplified)
            log_prior = 0.0

            # Classification loss
            logits = torch.matmul(U, weights)
            probs = torch.sigmoid(logits)
            classification_loss = F.binary_cross_entropy(probs, Y)

            # Combined posterior
            posterior = log_likelihood - log_prior - classification_loss
            posteriors.append(posterior.item())

        # Find sample with maximum posterior
        map_idx = np.argmax(posteriors)
        map_sample = self.retained_samples[map_idx]

        # Set parameters to MAP estimate
        self.membership_matrix = map_sample['U']
        self.cluster_centers = map_sample['V']
        self.covariance_matrices = map_sample['Sigma']
        self.mixing_proportions = map_sample['pi']
        self.weights = map_sample['weights']

    def predict_proba(self, X):
        """Predict class probabilities for X"""
        X_tensor = torch.tensor(X, dtype=torch.float32)

        # Compute membership matrix for new data (simplified)
        n, p = X_tensor.shape
        U = torch.zeros(n, self.n_clusters)

        for i in range(n):
            for k in range(self.n_clusters):
                diff = X_tensor[i] - self.cluster_centers[k]
                inv_cov = torch.inverse(self.covariance_matrices[k])
                mahalanobis = torch.dot(torch.matmul(diff, inv_cov), diff)
                U[i, k] = torch.exp(-0.5 * mahalanobis) * self.mixing_proportions[k]

        # Normalize
        U = U / torch.sum(U, dim=1, keepdim=True)

        # Compute probabilities using the linear model and softmax
        logits = torch.matmul(U, self.weights)
        probs = torch.sigmoid(logits)

        return probs.detach().numpy()

    def predict(self, X, threshold=0.5):
        """Predict class labels for X"""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    @staticmethod
    def load_and_preprocess_data(mat_file, n_components=10):
        """Load and preprocess .mat file dataset with PCA dimensionality reduction"""
        data = sio.loadmat(mat_file)
        try:
            X = data['X']  # Features
            Y = data['Y']  # Labels
        except KeyError:
            # Try common alternative keys
            keys = list(data.keys())
            print(f"Available keys in the .mat file: {keys}")

            # Try to identify feature and label matrices based on shape
            matrices = [k for k in keys if isinstance(data[k], np.ndarray) and len(data[k].shape) == 2]

            if len(matrices) >= 2:
                X = data[matrices[0]]
                Y = data[matrices[1]]
            else:
                raise ValueError("Could not identify feature and label matrices in the .mat file")

        # Convert to numpy arrays if not already
        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32)

        # Apply PCA for dimensionality reduction
        print(f"Original feature dimension: {X.shape[1]}")
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(X)
        print(f"Reduced feature dimension: {X_reduced.shape[1]}")
        print(f"Explained variance ratio: {sum(pca.explained_variance_ratio_):.4f}")

        return X_reduced, Y


def main():
    # Load data with PCA reduction to 10 features
    mat_file = "LIBs_1200.mat"
    X, Y = BPFCMClassifier.load_and_preprocess_data(mat_file, n_components=10)

    # Split data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Initialize and train model
    model = BPFCMClassifier(
        n_clusters=6,  # Adjust based on your dataset
        n_labels=Y.shape[1],  # Number of labels
        max_iter=2000,
        burn_in=50,
        thin=3
    )

    # Fit model
    model.fit(X_train, Y_train)

    # Evaluate
    Y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = np.mean(np.all(Y_pred == Y_test, axis=1))
    hamming_loss = np.mean(np.sum(Y_pred != Y_test, axis=1) / Y_test.shape[1])

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Hamming Loss: {hamming_loss:.4f}")

    # Per-class metrics
    for i in range(Y_test.shape[1]):
        class_accuracy = np.mean(Y_pred[:, i] == Y_test[:, i])
        print(f"Class {i + 1} Accuracy: {class_accuracy:.4f}")


if __name__ == "__main__":
    main()