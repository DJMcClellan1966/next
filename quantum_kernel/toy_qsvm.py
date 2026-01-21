"""
Toy Quantum-Inspired SVM Example
Compares classical SVM with quantum-inspired kernel methods on Iris dataset
"""
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import sys
import os

# Add parent directory to path to import quantum_kernel
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_kernel import get_kernel, KernelConfig


def prepare_data():
    """Load and prepare Iris dataset"""
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    
    # Use only 2 classes for simplicity
    mask = (y == 0) | (y == 1)
    X, y = X[mask], y[mask]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler


def classical_svm(X_train, X_test, y_train, y_test):
    """Classical SVM with RBF kernel"""
    print("\n" + "="*60)
    print("Classical SVM (RBF Kernel)")
    print("="*60)
    
    svm = SVC(kernel='rbf', gamma='scale', random_state=42)
    svm.fit(X_train, y_train)
    
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return accuracy


def quantum_kernel_svm(X_train, X_test, y_train, y_test):
    """Quantum-inspired SVM using quantum kernel methods"""
    print("\n" + "="*60)
    print("Quantum-Inspired SVM (Quantum Kernel)")
    print("="*60)
    
    # Initialize quantum kernel with quantum methods enabled
    config = KernelConfig(
        use_sentence_transformers=False,  # We'll use raw feature vectors
        use_quantum_methods=True,
        quantum_amplitude_encoding=True,
        similarity_metric='quantum',
        cache_type='lru',
        cache_size=1000
    )
    kernel = get_kernel(config)
    
    # Convert feature vectors to text-like format for kernel
    # We'll use the feature values as a "document"
    X_train_text = [f"f0:{x[0]:.3f} f1:{x[1]:.3f} f2:{x[2]:.3f} f3:{x[3]:.3f}" 
                    for x in X_train]
    X_test_text = [f"f0:{x[0]:.3f} f1:{x[1]:.3f} f2:{x[2]:.3f} f3:{x[3]:.3f}" 
                   for x in X_test]
    
    # Build kernel matrix using quantum similarity
    print("Building quantum kernel matrix...")
    train_kernel = np.zeros((len(X_train), len(X_train)))
    for i in range(len(X_train)):
        for j in range(len(X_train)):
            sim = kernel.similarity(X_train_text[i], X_train_text[j])
            train_kernel[i, j] = sim
    
    test_kernel = np.zeros((len(X_test), len(X_train)))
    for i in range(len(X_test)):
        for j in range(len(X_train)):
            sim = kernel.similarity(X_test_text[i], X_train_text[j])
            test_kernel[i, j] = sim
    
    # Use SVM with precomputed kernel
    svm = SVC(kernel='precomputed', random_state=42)
    svm.fit(train_kernel, y_train)
    
    y_pred = svm.predict(test_kernel)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return accuracy


def quantum_feature_svm(X_train, X_test, y_train, y_test):
    """SVM using quantum-inspired feature embeddings"""
    print("\n" + "="*60)
    print("Quantum-Inspired Feature SVM")
    print("="*60)
    
    # Initialize quantum kernel for embeddings
    config = KernelConfig(
        use_sentence_transformers=True,  # Use sentence transformers for embeddings
        use_quantum_methods=True,
        quantum_amplitude_encoding=True,
        similarity_metric='quantum',
        cache_type='lru',
        cache_size=1000
    )
    kernel = get_kernel(config)
    
    # Convert to text format
    X_train_text = [f"f0:{x[0]:.3f} f1:{x[1]:.3f} f2:{x[2]:.3f} f3:{x[3]:.3f}" 
                    for x in X_train]
    X_test_text = [f"f0:{x[0]:.3f} f1:{x[1]:.3f} f2:{x[2]:.3f} f3:{x[3]:.3f}" 
                   for x in X_test]
    
    print("Generating quantum-inspired embeddings...")
    # Get embeddings using quantum kernel
    X_train_emb = np.array([kernel._create_embedding(text) for text in X_train_text])
    X_test_emb = np.array([kernel._create_embedding(text) for text in X_test_text])
    
    # Use SVM with RBF kernel on quantum embeddings
    svm = SVC(kernel='rbf', gamma='scale', random_state=42)
    svm.fit(X_train_emb, y_train)
    
    y_pred = svm.predict(X_test_emb)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Embedding dimension: {X_train_emb.shape[1]}")
    
    return accuracy


def main():
    """Main function to run comparisons"""
    print("\n" + "="*60)
    print("Toy Quantum-Inspired SVM Comparison")
    print("Dataset: Iris (2 classes)")
    print("="*60)
    
    # Prepare data
    X_train, X_test, y_train, y_test, scaler = prepare_data()
    print(f"\nDataset shape: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    
    # Run comparisons
    acc_classical = classical_svm(X_train, X_test, y_train, y_test)
    acc_quantum_kernel = quantum_kernel_svm(X_train, X_test, y_train, y_test)
    acc_quantum_feature = quantum_feature_svm(X_train, X_test, y_train, y_test)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Classical SVM (RBF):           {acc_classical:.4f}")
    print(f"Quantum Kernel SVM:            {acc_quantum_kernel:.4f}")
    print(f"Quantum Feature SVM:           {acc_quantum_feature:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
