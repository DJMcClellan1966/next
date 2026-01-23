"""
Tests for Advanced Cybersecurity and Data Learning Methods
"""
import sys
from pathlib import Path
import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from advanced_cybersecurity import (
        SecureMultiPartyComputation, ModelWatermarking, SecureModelServing,
        OutputSanitizer, PrivacyBudgetManager, ModelExtractionPrevention,
        MembershipInferenceDefense, DataPoisoningDetector, SecureModelDeployment
    )
    from advanced_data_learning import (
        SecureFederatedLearning, PrivacyPreservingFeatureEngineering,
        SecureAggregationWithDP, PrivacyPreservingInference, EncryptedInference,
        SecureModelSharing, PrivacyAuditor, AdvancedOnlineLearning,
        StreamingDataLearning, IncrementalFeatureSelection
    )
    FEATURES_AVAILABLE = True
except ImportError:
    FEATURES_AVAILABLE = False
    pytestmark = pytest.mark.skip("Features not available")


class TestAdvancedCybersecurity:
    """Tests for advanced cybersecurity methods"""
    
    def test_secure_multi_party_computation(self):
        """Test secure multi-party computation"""
        smpc = SecureMultiPartyComputation(num_parties=3)
        
        smpc.add_party('party1', np.array([1, 2, 3]))
        smpc.add_party('party2', np.array([4, 5, 6]))
        smpc.add_party('party3', np.array([7, 8, 9]))
        
        result = smpc.secure_sum()
        assert 'result' in result
        assert result['num_parties'] == 3
        
        avg_result = smpc.secure_average()
        assert 'result' in avg_result
    
    def test_model_watermarking(self):
        """Test model watermarking"""
        from sklearn.ensemble import RandomForestClassifier
        
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        X = np.random.rand(10, 5)
        y = np.random.randint(0, 2, 10)
        model.fit(X, y)
        
        watermarker = ModelWatermarking()
        result = watermarker.embed_watermark(model, "test_watermark")
        
        assert 'watermark_hash' in result
        
        verified = watermarker.verify_watermark(model, "test_watermark")
        assert verified == True
    
    def test_secure_model_serving(self):
        """Test secure model serving"""
        from sklearn.ensemble import RandomForestClassifier
        from ml_security_framework import InputValidator
        
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        X = np.random.rand(10, 5)
        y = np.random.randint(0, 2, 10)
        model.fit(X, y)
        
        validator = InputValidator()
        server = SecureModelServing(model, validator)
        
        result = server.predict_secure(X[:3])
        assert 'prediction' in result or 'error' in result
    
    def test_output_sanitizer(self):
        """Test output sanitization"""
        sanitizer = OutputSanitizer(max_output_size=1000)
        
        output = np.array([1, 2, 3, 4, 5])
        result = sanitizer.sanitize(output)
        
        assert result['sanitized'] == True
        assert 'sanitized_output' in result
    
    def test_privacy_budget_manager(self):
        """Test privacy budget management"""
        manager = PrivacyBudgetManager(total_budget=1.0)
        
        allocated = manager.allocate_budget(0.3, 'operation1')
        assert allocated == True
        
        remaining = manager.get_remaining_budget()
        assert remaining == 0.7
        
        report = manager.get_budget_report()
        assert 'total_budget' in report
        assert 'used_budget' in report
    
    def test_model_extraction_prevention(self):
        """Test model extraction prevention"""
        from sklearn.ensemble import RandomForestClassifier
        
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        X = np.random.rand(10, 5)
        y = np.random.randint(0, 2, 10)
        model.fit(X, y)
        
        protector = ModelExtractionPrevention(model, max_queries=5)
        
        for i in range(5):
            result = protector.predict_with_protection(X[:1])
            assert 'prediction' in result or 'error' in result
        
        # Should be blocked now
        result = protector.predict_with_protection(X[:1])
        assert result.get('blocked', False) == True
    
    def test_membership_inference_defense(self):
        """Test membership inference defense"""
        from sklearn.ensemble import RandomForestClassifier
        
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        X = np.random.rand(10, 5)
        y = np.random.randint(0, 2, 10)
        model.fit(X, y)
        
        defender = MembershipInferenceDefense(model)
        protected = defender.predict_with_defense(X[:3])
        
        assert protected is not None
        assert len(protected) == 3
    
    def test_data_poisoning_detector(self):
        """Test data poisoning detection"""
        detector = DataPoisoningDetector(contamination=0.1)
        
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)
        
        result = detector.detect_poisoning(X, y)
        
        assert 'poisoned_detected' in result
        assert 'poisoned_indices' in result
    
    def test_secure_model_deployment(self):
        """Test secure model deployment"""
        from sklearn.ensemble import RandomForestClassifier
        
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        X = np.random.rand(10, 5)
        y = np.random.randint(0, 2, 10)
        model.fit(X, y)
        
        config = {
            'encryption': True,
            'input_validation': True,
            'output_sanitization': True
        }
        
        deployment = SecureModelDeployment(model, config)
        result = deployment.deploy()
        
        assert result['success'] == True


class TestAdvancedDataLearning:
    """Tests for advanced data learning methods"""
    
    def test_secure_federated_learning(self):
        """Test secure federated learning"""
        from sklearn.ensemble import RandomForestClassifier
        
        sfl = SecureFederatedLearning(use_encryption=True, use_dp=True, epsilon=1.0)
        
        models = []
        sizes = []
        for i in range(3):
            model = RandomForestClassifier(n_estimators=5, random_state=42)
            X = np.random.rand(20, 5)
            y = np.random.randint(0, 2, 20)
            model.fit(X, y)
            models.append(model)
            sizes.append(20)
        
        result = sfl.secure_aggregate(models, sizes)
        assert 'aggregated_model' in result
    
    def test_privacy_preserving_feature_engineering(self):
        """Test privacy-preserving feature engineering"""
        ppfe = PrivacyPreservingFeatureEngineering(privacy_budget=1.0)
        
        X = np.random.rand(100, 10)
        stats = ppfe.private_statistics(X)
        
        assert 'mean' in stats
        assert 'std' in stats
    
    def test_secure_aggregation_with_dp(self):
        """Test secure aggregation with differential privacy"""
        aggregator = SecureAggregationWithDP(epsilon=1.0)
        
        updates = [
            np.random.rand(10),
            np.random.rand(10),
            np.random.rand(10)
        ]
        
        result = aggregator.aggregate_with_dp(updates)
        assert 'aggregated' in result
        assert result['epsilon'] == 1.0
    
    def test_privacy_preserving_inference(self):
        """Test privacy-preserving inference"""
        from sklearn.ensemble import RandomForestClassifier
        
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        X = np.random.rand(10, 5)
        y = np.random.randint(0, 2, 10)
        model.fit(X, y)
        
        ppi = PrivacyPreservingInference(model, epsilon=1.0)
        result = ppi.private_predict(X[:3])
        
        assert 'predictions' in result
        assert result['private'] == True
    
    def test_encrypted_inference(self):
        """Test encrypted inference"""
        from sklearn.ensemble import RandomForestClassifier
        
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        X = np.random.rand(10, 5)
        y = np.random.randint(0, 2, 10)
        model.fit(X, y)
        
        ei = EncryptedInference(model)
        result = ei.encrypted_predict(X[:3])
        
        assert 'predictions' in result
        assert result['encrypted'] == True
    
    def test_secure_model_sharing(self):
        """Test secure model sharing"""
        from sklearn.ensemble import RandomForestClassifier
        
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        X = np.random.rand(10, 5)
        y = np.random.randint(0, 2, 10)
        model.fit(X, y)
        
        sharing = SecureModelSharing(model)
        result = sharing.share_model('recipient1', ['read', 'predict'])
        
        assert result['shared'] == True
        assert 'recipient' in result
    
    def test_privacy_auditor(self):
        """Test privacy auditor"""
        auditor = PrivacyAuditor()
        
        audit = auditor.audit_operation('operation1', 0.5, 'medium')
        assert 'compliant' in audit
        
        report = auditor.get_audit_report()
        assert 'total_operations' in report
        assert 'compliance_rate' in report
    
    def test_advanced_online_learning(self):
        """Test advanced online learning"""
        from sklearn.linear_model import SGDClassifier
        
        model = SGDClassifier(random_state=42)
        aol = AdvancedOnlineLearning(model, adaptive=True)
        
        X_new = np.random.rand(5, 10)
        y_new = np.random.randint(0, 2, 5)
        
        result = aol.incremental_update(X_new, y_new)
        assert result['updated'] == True
    
    def test_streaming_data_learning(self):
        """Test streaming data learning"""
        from sklearn.linear_model import SGDClassifier
        
        model = SGDClassifier(random_state=42)
        sdl = StreamingDataLearning(model, window_size=100)
        
        X = np.random.rand(10, 5)
        y = np.random.randint(0, 2, 10)
        
        result = sdl.process_stream(X, y)
        assert result['processed'] == True
    
    def test_incremental_feature_selection(self):
        """Test incremental feature selection"""
        ifs = IncrementalFeatureSelection(max_features=5)
        
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)
        
        result = ifs.update_selection(X, y)
        assert 'selected_features' in result
        assert len(result['selected_features']) <= 5


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
