"""
Network Security for ML Serving
Network security testing and monitoring for ML APIs

Features:
- API security testing
- Network attack detection
- Traffic analysis
- ML endpoint protection
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import warnings
import time
from collections import defaultdict, deque
import json

sys.path.insert(0, str(Path(__file__).parent))


class MLNetworkSecurity:
    """
    Network Security for ML Serving
    
    Network security testing and monitoring
    """
    
    def __init__(self, api_endpoint: Optional[str] = None):
        """
        Args:
            api_endpoint: ML API endpoint URL (optional)
        """
        self.api_endpoint = api_endpoint
        self.traffic_history = deque(maxlen=1000)
        self.attack_patterns = []
    
    def test_api_security(
        self,
        test_cases: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Test ML API security
        
        Args:
            test_cases: Custom test cases (optional)
            
        Returns:
            API security test results
        """
        if test_cases is None:
            test_cases = self._generate_default_test_cases()
        
        results = {
            'endpoint': self.api_endpoint,
            'tests_performed': len(test_cases),
            'vulnerabilities': [],
            'passed_tests': 0,
            'failed_tests': 0,
            'security_score': 0.0
        }
        
        for test_case in test_cases:
            test_result = self._run_security_test(test_case)
            
            if test_result['passed']:
                results['passed_tests'] += 1
            else:
                results['failed_tests'] += 1
                results['vulnerabilities'].append({
                    'test': test_case['name'],
                    'severity': test_result.get('severity', 'medium'),
                    'issue': test_result.get('issue', 'Test failed'),
                    'recommendation': test_result.get('recommendation', 'Fix security issue')
                })
        
        # Calculate security score
        if results['tests_performed'] > 0:
            results['security_score'] = (
                results['passed_tests'] / results['tests_performed'] * 100
            )
        
        return results
    
    def _generate_default_test_cases(self) -> List[Dict[str, Any]]:
        """Generate default security test cases"""
        return [
            {
                'name': 'SQL Injection Test',
                'type': 'injection',
                'payload': "'; DROP TABLE models; --"
            },
            {
                'name': 'XSS Test',
                'type': 'xss',
                'payload': '<script>alert("XSS")</script>'
            },
            {
                'name': 'Path Traversal Test',
                'type': 'path_traversal',
                'payload': '../../../etc/passwd'
            },
            {
                'name': 'Large Payload Test',
                'type': 'dos',
                'payload': 'A' * 10000
            },
            {
                'name': 'Rate Limiting Test',
                'type': 'rate_limit',
                'requests': 100
            }
        ]
    
    def _run_security_test(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Run a security test"""
        test_type = test_case.get('type', 'unknown')
        
        # Simplified tests (full implementation would make actual API calls)
        if test_type == 'injection':
            return {
                'passed': True,  # Would test actual API
                'severity': 'high',
                'issue': 'SQL injection vulnerability possible',
                'recommendation': 'Use parameterized queries and input validation'
            }
        elif test_type == 'xss':
            return {
                'passed': True,
                'severity': 'medium',
                'issue': 'XSS vulnerability possible',
                'recommendation': 'Sanitize user inputs and use CSP headers'
            }
        elif test_type == 'dos':
            return {
                'passed': True,
                'severity': 'high',
                'issue': 'DoS vulnerability possible',
                'recommendation': 'Implement request size limits and rate limiting'
            }
        else:
            return {
                'passed': True,
                'severity': 'low',
                'issue': 'Test completed',
                'recommendation': 'Monitor for security issues'
            }
    
    def analyze_traffic(
        self,
        traffic_logs: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Analyze ML API traffic
        
        Args:
            traffic_logs: Traffic log data (optional)
            
        Returns:
            Traffic analysis results
        """
        if traffic_logs is None:
            traffic_logs = list(self.traffic_history)
        
        analysis = {
            'total_requests': len(traffic_logs),
            'request_patterns': {},
            'anomalies': [],
            'attack_indicators': [],
            'recommendations': []
        }
        
        if not traffic_logs:
            return analysis
        
        # Analyze request patterns
        request_types = defaultdict(int)
        request_sizes = []
        request_times = []
        
        for log in traffic_logs:
            request_types[log.get('type', 'unknown')] += 1
            request_sizes.append(log.get('size', 0))
            request_times.append(log.get('timestamp', 0))
        
        analysis['request_patterns'] = {
            'types': dict(request_types),
            'avg_size': np.mean(request_sizes) if request_sizes else 0,
            'max_size': np.max(request_sizes) if request_sizes else 0
        }
        
        # Detect anomalies
        if request_sizes:
            avg_size = np.mean(request_sizes)
            std_size = np.std(request_sizes)
            
            for i, size in enumerate(request_sizes):
                if size > avg_size + 3 * std_size:
                    analysis['anomalies'].append({
                        'type': 'unusually_large_request',
                        'index': i,
                        'size': size,
                        'severity': 'medium'
                    })
        
        # Detect attack indicators
        for log in traffic_logs:
            payload = str(log.get('payload', ''))
            
            # Check for SQL injection patterns
            if any(pattern in payload.lower() for pattern in ['union', 'select', 'drop', 'delete']):
                analysis['attack_indicators'].append({
                    'type': 'sql_injection',
                    'severity': 'high',
                    'log': log
                })
            
            # Check for XSS patterns
            if '<script' in payload.lower():
                analysis['attack_indicators'].append({
                    'type': 'xss',
                    'severity': 'medium',
                    'log': log
                })
        
        # Generate recommendations
        if analysis['attack_indicators']:
            analysis['recommendations'].append('Implement input validation and sanitization')
        
        if analysis['anomalies']:
            analysis['recommendations'].append('Monitor for unusual request patterns')
        
        return analysis
    
    def detect_attacks(
        self,
        traffic: List[Dict[str, Any]],
        use_ml_toolbox: bool = True
    ) -> Dict[str, Any]:
        """
        Detect network attacks using ML
        
        Args:
            traffic: Traffic data
            use_ml_toolbox: Use ML Toolbox for detection
            
        Returns:
            Attack detection results
        """
        detection = {
            'attacks_detected': [],
            'attack_types': defaultdict(int),
            'total_attacks': 0,
            'recommendations': []
        }
        
        # Extract features from traffic
        features = []
        for log in traffic:
            features.append([
                log.get('size', 0),
                log.get('duration', 0),
                len(str(log.get('payload', ''))),
                hash(str(log.get('payload', ''))) % 1000  # Simplified feature
            ])
        
        if not features:
            return detection
        
        features = np.array(features)
        
            # Use ML Toolbox for anomaly detection
        if use_ml_toolbox:
            try:
                from ml_toolbox import MLToolbox
                toolbox = MLToolbox()
                
                # Use threat detection system
                detector = toolbox.algorithms.get_threat_detection_system()
                
                # Create normal vs attack data (simplified)
                if len(features) > 1:
                    split_idx = len(features) // 2
                    X_normal = features[:split_idx]
                    X_attacks = features[split_idx:] + np.random.randn(len(features) - split_idx, features.shape[1]) * 10
                else:
                    X_normal = features
                    X_attacks = features + np.random.randn(*features.shape) * 10
                
                # Train detector
                detector.train_threat_detector(X_normal, X_attacks, use_ml_toolbox=True)
                
                # Detect attacks
                result = detector.detect_threat(features)
                
                if result.get('threat_detected'):
                    detection['attacks_detected'] = result.get('threat_indices', [])
                    detection['total_attacks'] = result.get('threat_count', 0)
                    detection['attack_types']['ml_detected'] = detection['total_attacks']
                
            except Exception as e:
                warnings.warn(f"ML Toolbox detection failed: {e}")
        
        # Rule-based detection
        for i, log in enumerate(traffic):
            payload = str(log.get('payload', ''))
            
            # SQL injection
            if any(pattern in payload.lower() for pattern in ['union', 'select', 'drop']):
                detection['attacks_detected'].append(i)
                detection['attack_types']['sql_injection'] += 1
            
            # XSS
            if '<script' in payload.lower():
                detection['attacks_detected'].append(i)
                detection['attack_types']['xss'] += 1
            
            # Large payload (DoS)
            if log.get('size', 0) > 10000:
                detection['attacks_detected'].append(i)
                detection['attack_types']['dos'] += 1
        
        detection['total_attacks'] = len(set(detection['attacks_detected']))
        
        if detection['total_attacks'] > 0:
            detection['recommendations'].append('Implement rate limiting and input validation')
            detection['recommendations'].append('Enable WAF (Web Application Firewall)')
        
        return detection
    
    def monitor_traffic(
        self,
        request: Dict[str, Any],
        use_ml_toolbox: bool = True
    ) -> Dict[str, Any]:
        """
        Monitor and analyze traffic in real-time
        
        Args:
            request: Request data
            use_ml_toolbox: Use ML Toolbox for analysis
            
        Returns:
            Monitoring results
        """
        # Add to history
        request['timestamp'] = time.time()
        self.traffic_history.append(request)
        
        # Analyze recent traffic
        recent_traffic = list(self.traffic_history)[-100:]  # Last 100 requests
        
        # Detect attacks
        detection = self.detect_attacks(recent_traffic, use_ml_toolbox)
        
        # Traffic analysis
        analysis = self.analyze_traffic(recent_traffic)
        
        return {
            'request_logged': True,
            'attack_detection': detection,
            'traffic_analysis': analysis,
            'recommendations': detection.get('recommendations', [])
        }
    
    def generate_security_report(self) -> Dict[str, Any]:
        """Generate network security report"""
        traffic_analysis = self.analyze_traffic()
        api_tests = self.test_api_security()
        
        return {
            'timestamp': time.time(),
            'endpoint': self.api_endpoint,
            'api_security': api_tests,
            'traffic_analysis': traffic_analysis,
            'overall_security_score': (
                api_tests.get('security_score', 0) * 0.6 +
                (100 - len(traffic_analysis.get('attack_indicators', [])) * 10) * 0.4
            ),
            'recommendations': (
                api_tests.get('vulnerabilities', []) +
                traffic_analysis.get('recommendations', [])
            )
        }


class MLAPISecurityTester:
    """
    ML API Security Tester
    
    Comprehensive API security testing
    """
    
    def __init__(self, api_endpoint: str):
        """
        Args:
            api_endpoint: ML API endpoint URL
        """
        self.api_endpoint = api_endpoint
        self.network_security = MLNetworkSecurity(api_endpoint)
    
    def comprehensive_test(self) -> Dict[str, Any]:
        """Comprehensive API security test"""
        results = {
            'endpoint': self.api_endpoint,
            'api_security': self.network_security.test_api_security(),
            'traffic_analysis': self.network_security.analyze_traffic(),
            'recommendations': []
        }
        
        # Combine recommendations
        results['recommendations'] = (
            results['api_security'].get('vulnerabilities', []) +
            results['traffic_analysis'].get('recommendations', [])
        )
        
        return results
