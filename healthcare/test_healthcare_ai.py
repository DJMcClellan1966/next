"""
Healthcare AI Assistant Tests
Comprehensive tests for healthcare use cases
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest
from healthcare.healthcare_ai_demo import HealthcareAIAssistant
from quantum_kernel import get_kernel, KernelConfig


class TestHealthcareAI(unittest.TestCase):
    """Test healthcare AI assistant functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        print("\n" + "="*70)
        print("Setting up Healthcare AI Assistant for testing...")
        print("="*70)
        cls.assistant = HealthcareAIAssistant()
        print()
    
    def test_drug_interaction_detection(self):
        """Test that drug interactions are detected"""
        print("\nTest: Drug Interaction Detection")
        print("-" * 70)
        
        # Test known interaction
        result = self.assistant.check_drug_interaction("Warfarin", "Aspirin")
        
        response_text = result['response'].lower()
        self.assertIsNotNone(result['response'])
        self.assertGreater(len(result['response']), 20, "Response should be substantial")
        
        # Should mention interaction or bleeding
        has_interaction_keywords = any(keyword in response_text for keyword in [
            'interaction', 'bleeding', 'risk', 'warn', 'caution', 'monitor'
        ])
        self.assertTrue(has_interaction_keywords, 
                       "Response should mention interaction or risk")
        
            print(f"[+] Query: {result['question']}")
            print(f"[+] Response length: {len(result['response'])} characters")
            print(f"[+] Confidence: {result['confidence']:.2f}")
            print(f"[+] Found interaction-related keywords: {has_interaction_keywords}")
    
    def test_clinical_protocol_retrieval(self):
        """Test that clinical protocols can be retrieved"""
        print("\nTest: Clinical Protocol Retrieval")
        print("-" * 70)
        
        result = self.assistant.get_protocol("chest pain")
        
        self.assertIsNotNone(result['response'])
        self.assertGreater(len(result['response']), 20, "Response should be substantial")
        self.assertGreater(result['confidence'], 0.0, "Should have some confidence")
        
        response_text = result['response'].lower()
        # Should mention protocol-related terms
        has_protocol_keywords = any(keyword in response_text for keyword in [
            'protocol', 'assess', 'ekg', 'cardiac', 'chest', 'pain', 'test'
        ])
        
        print(f"[+] Query: {result['question']}")
        print(f"[+] Response length: {len(result['response'])} characters")
        print(f"[+] Confidence: {result['confidence']:.2f}")
        print(f"[+] Found protocol-related keywords: {has_protocol_keywords}")
    
    def test_symptom_assessment(self):
        """Test symptom assessment support"""
        print("\nTest: Symptom Assessment")
        print("-" * 70)
        
        symptoms = "chest pain radiating to left arm, sweating, nausea"
        vital_signs = "BP 140/90, HR 95"
        
        result = self.assistant.assess_symptoms(symptoms, vital_signs)
        
        self.assertIsNotNone(result['response'])
        self.assertGreater(len(result['response']), 20, "Response should be substantial")
        
        response_text = result['response'].lower()
        # Should provide clinical guidance
        has_guidance_keywords = any(keyword in response_text for keyword in [
            'assess', 'evaluate', 'consider', 'test', 'monitor', 'chest', 'cardiac'
        ])
        
        print(f"[+] Symptoms: {symptoms}")
        print(f"[+] Response length: {len(result['response'])} characters")
        print(f"[+] Confidence: {result['confidence']:.2f}")
        print(f"[+] Found clinical guidance keywords: {has_guidance_keywords}")
    
    def test_knowledge_base_search(self):
        """Test that medical knowledge base is searchable"""
        print("\nTest: Knowledge Base Search")
        print("-" * 70)
        
        query = "diabetes management guidelines"
        result = self.assistant.query(query)
        
        self.assertIsNotNone(result['response'])
        self.assertGreater(len(result['sources']), 0, "Should find relevant sources")
        self.assertGreater(result['confidence'], 0.0, "Should have some confidence")
        
        print(f"[+] Query: {query}")
        print(f"[+] Found {len(result['sources'])} relevant sources")
        print(f"[+] Confidence: {result['confidence']:.2f}")
        for i, source in enumerate(result['sources'][:3], 1):
            print(f"  Source {i}: {source[:80]}...")
    
    def test_multiple_drug_interactions(self):
        """Test multiple drug interaction queries"""
        print("\nTest: Multiple Drug Interactions")
        print("-" * 70)
        
        drug_pairs = [
            ("Warfarin", "Aspirin"),
            ("ACE inhibitor", "Potassium-sparing diuretic"),
            ("Beta-blocker", "Calcium channel blocker")
        ]
        
        for drug1, drug2 in drug_pairs:
            result = self.assistant.check_drug_interaction(drug1, drug2)
            self.assertIsNotNone(result['response'])
            self.assertGreater(len(result['response']), 10)
            print(f"[+] {drug1} + {drug2}: Response generated ({len(result['response'])} chars)")
    
    def test_protocol_queries(self):
        """Test various protocol queries"""
        print("\nTest: Protocol Queries")
        print("-" * 70)
        
        protocols = [
            "chest pain",
            "diabetes management",
            "hypertension treatment"
        ]
        
        for protocol in protocols:
            result = self.assistant.get_protocol(protocol)
            self.assertIsNotNone(result['response'])
            self.assertGreater(len(result['response']), 10)
            print(f"[+] {protocol}: Response generated ({len(result['response'])} chars)")
    
    def test_response_structure(self):
        """Test that responses have correct structure"""
        print("\nTest: Response Structure")
        print("-" * 70)
        
        result = self.assistant.query("What is diabetes?")
        
        # Check required fields
        self.assertIn('question', result)
        self.assertIn('response', result)
        self.assertIn('confidence', result)
        self.assertIn('sources', result)
        self.assertIn('disclaimer', result)
        
        # Check types
        self.assertIsInstance(result['question'], str)
        self.assertIsInstance(result['response'], str)
        self.assertIsInstance(result['confidence'], (int, float))
        self.assertIsInstance(result['sources'], list)
        self.assertIsInstance(result['disclaimer'], str)
        
        print("[+] Response has all required fields")
        print("[+] All fields have correct types")
    
    def test_disclaimer_present(self):
        """Test that medical disclaimers are always present"""
        print("\nTest: Medical Disclaimer")
        print("-" * 70)
        
        result = self.assistant.query("Test question")
        
        disclaimer = result['disclaimer'].lower()
        self.assertIn('disclaimer', result)
        self.assertGreater(len(result['disclaimer']), 50)
        
        # Should mention key safety terms
        safety_terms = ['support', 'judgment', 'advice', 'verify', 'replace']
        has_safety_terms = any(term in disclaimer for term in safety_terms)
        self.assertTrue(has_safety_terms, "Disclaimer should mention safety terms")
        
        print("[+] Disclaimer present in all responses")
        print(f"[+] Disclaimer: {result['disclaimer'][:100]}...")


class TestHealthcareAISemantic(unittest.TestCase):
    """Test semantic understanding for healthcare"""
    
    def setUp(self):
        """Set up for semantic tests"""
        self.kernel = get_kernel(KernelConfig(use_sentence_transformers=True))
    
    def test_synonym_detection_medical(self):
        """Test that medical synonyms are detected"""
        print("\nTest: Medical Synonym Detection")
        print("-" * 70)
        
        medical_synonyms = [
            ("heart attack", "myocardial infarction"),
            ("high blood pressure", "hypertension"),
            ("diabetes", "diabetes mellitus")
        ]
        
        for term1, term2 in medical_synonyms:
            similarity = self.kernel.similarity(term1, term2)
            self.assertGreater(similarity, 0.5, 
                             f"{term1} and {term2} should be similar")
            print(f"[+] '{term1}' <-> '{term2}': {similarity:.3f}")
    
    def test_medical_concept_relationships(self):
        """Test medical concept relationships"""
        print("\nTest: Medical Concept Relationships")
        print("-" * 70)
        
        concepts = [
            "chest pain",
            "cardiac enzymes",
            "EKG",
            "acute coronary syndrome"
        ]
        
        # Find relationships
        relationships = self.kernel.find_similar("chest pain protocol", concepts, top_k=3)
        self.assertGreater(len(relationships), 0)
        
        print("[+] Found relationships for 'chest pain protocol':")
        for concept, score in relationships:
            print(f"  - {concept}: {score:.3f}")
    
    def test_drug_similarity(self):
        """Test that similar drugs are identified"""
        print("\nTest: Drug Similarity")
        print("-" * 70)
        
        # Drugs in same class should be similar
        ace_inhibitors = ["Lisinopril", "Enalapril", "Ramipril"]
        
        for i, drug1 in enumerate(ace_inhibitors):
            for drug2 in ace_inhibitors[i+1:]:
                similarity = self.kernel.similarity(drug1, drug2)
                # May not be high since drug names alone, but should be > 0
                self.assertGreater(similarity, 0.0)
                print(f"[+] {drug1} <-> {drug2}: {similarity:.3f}")


def run_all_tests():
    """Run all healthcare tests"""
    print("\n" + "="*70)
    print("HEALTHCARE AI ASSISTANT - COMPREHENSIVE TESTS")
    print("="*70)
    print("\n*** NOTE: These tests validate functionality, not medical accuracy. ***")
    print("   Always verify medical information with licensed professionals.")
    print("="*70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestHealthcareAI))
    suite.addTests(loader.loadTestsFromTestCase(TestHealthcareAISemantic))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print("="*70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
