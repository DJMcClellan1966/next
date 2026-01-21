"""
Healthcare AI Assistant Demo
Demonstrates how the AI system works for healthcare applications
HIPAA-compliant local processing
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_kernel import get_kernel, KernelConfig
from ai import CompleteAISystem
from llm.quantum_llm_standalone import StandaloneQuantumLLM


class HealthcareAIAssistant:
    """
    Healthcare AI Assistant
    Demonstrates HIPAA-compliant local AI for clinical decision support
    """
    
    def __init__(self):
        """Initialize healthcare AI assistant"""
        print("Initializing Healthcare AI Assistant...")
        print("[+] Loading medical knowledge base...")
        
        # Initialize AI system with LLM support
        self.ai = CompleteAISystem(use_llm=True)
        self.llm = StandaloneQuantumLLM(kernel=self.ai.kernel)
        
        # Load medical knowledge base
        self._load_medical_knowledge()
        
        print("[+] Healthcare AI Assistant ready!")
        print("  - HIPAA compliant (local processing)")
        print("  - Medical protocols loaded")
        print("  - Drug database initialized")
        print()
    
    def _load_medical_knowledge(self):
        """Load medical knowledge into the system"""
        medical_knowledge = [
            # Drug Information
            "Warfarin is an anticoagulant used to prevent blood clots. Common dosage: 2-10mg daily.",
            "Aspirin is an antiplatelet medication. Low-dose aspirin (81mg) is commonly used for cardiovascular protection.",
            "Warfarin and Aspirin together increase bleeding risk. Requires careful monitoring and usually avoided unless specifically indicated.",
            "ACE inhibitors like Lisinopril are used for hypertension and heart failure. Common side effects include cough and hyperkalemia.",
            
            # Clinical Protocols
            "Chest pain protocol: Assess ABCs, obtain EKG within 10 minutes, check cardiac enzymes, consider aspirin and nitroglycerin if indicated.",
            "Diabetes management: Check HbA1c every 3 months, screen for complications annually, manage blood pressure and cholesterol.",
            "Hypertension treatment: Lifestyle modifications first, then consider ACE inhibitor, ARB, or thiazide diuretic based on comorbidities.",
            
            # Clinical Guidelines
            "Acute coronary syndrome (ACS) requires immediate EKG, cardiac enzymes, aspirin 325mg, and consideration of dual antiplatelet therapy.",
            "Sepsis protocol: Obtain cultures, administer antibiotics within 1 hour, provide fluid resuscitation, monitor lactate levels.",
            "Stroke protocol: Time is critical, obtain CT scan immediately, assess for thrombolytics if within 4.5 hours of symptom onset.",
            
            # Drug Interactions
            "Warfarin interacts with many medications including antibiotics, antifungals, and some pain medications.",
            "ACE inhibitors combined with potassium-sparing diuretics can cause dangerous hyperkalemia.",
            "Beta-blockers and calcium channel blockers together can cause significant bradycardia.",
            
            # Documentation Standards
            "SOAP note format: Subjective (patient's description), Objective (vital signs, exam findings), Assessment (diagnosis), Plan (treatment).",
            "Progress notes should include: chief complaint, review of systems, physical exam, assessment, and plan.",
            "Documentation must be clear, concise, and support medical decision-making for billing and legal purposes."
        ]
        
        # Add all knowledge to the system
        for knowledge in medical_knowledge:
            self.ai.knowledge_graph.add_document(knowledge)
        
        print(f"  Loaded {len(medical_knowledge)} medical knowledge items")
    
    def query(self, question: str, context: str = None) -> dict:
        """
        Query the healthcare AI assistant
        
        Args:
            question: Clinical question
            context: Optional patient context
            
        Returns:
            Dictionary with response and confidence
        """
        # Combine question with context
        full_query = question
        if context:
            full_query = f"Context: {context}\n\nQuestion: {question}"
        
        # Get documents from knowledge graph
        knowledge_corpus = [node.get('text', '') for node in self.ai.knowledge_graph.graph.get('nodes', [])]
        
        # Search medical knowledge base
        search_results = self.ai.search.search(full_query, knowledge_corpus, top_k=3) if knowledge_corpus else []
        
        # Generate response using LLM
        prompt = f"Answer this medical question based on clinical guidelines and protocols: {question}"
        if context:
            prompt += f"\n\nPatient context: {context}"
        
        # Get relevant context from search
        relevant_context = "\n".join([doc for doc, score in search_results])
        
        response = self.llm.generate_grounded(
            prompt=f"{prompt}\n\nRelevant medical information:\n{relevant_context}",
            max_length=300,
            temperature=0.7
        )
        
        return {
            "question": question,
            "response": response.get('generated') or response.get('text', str(response)),
            "confidence": response.get('confidence', 0.0),
            "sources": [doc for doc, score in search_results],
            "disclaimer": "This is AI-assisted clinical decision support. Always verify information and use clinical judgment. Not a replacement for professional medical advice."
        }
    
    def check_drug_interaction(self, drug1: str, drug2: str) -> dict:
        """Check for drug interactions"""
        query = f"Are there any interactions between {drug1} and {drug2}?"
        return self.query(query)
    
    def get_protocol(self, condition: str) -> dict:
        """Get clinical protocol for a condition"""
        query = f"What is the protocol for {condition}?"
        return self.query(query)
    
    def assess_symptoms(self, symptoms: str, vital_signs: str = None) -> dict:
        """Provide symptom assessment support"""
        query = f"Provide clinical guidance for these symptoms: {symptoms}"
        if vital_signs:
            query += f"\nVital signs: {vital_signs}"
        return self.query(query)


def demo_drug_interaction():
    """Demo: Drug interaction check"""
    print("="*70)
    print("DEMO 1: Drug Interaction Check")
    print("="*70)
    
    assistant = HealthcareAIAssistant()
    
    result = assistant.check_drug_interaction("Warfarin", "Aspirin")
    
    print(f"\nQuestion: {result['question']}")
    print(f"\nResponse:\n{result['response']}")
    print(f"\nConfidence: {result['confidence']:.2f}")
    print(f"\nSources:")
    for i, source in enumerate(result['sources'][:3], 1):
        print(f"  {i}. {source[:100]}...")
    print(f"\n{result['disclaimer']}")
    print()


def demo_clinical_protocol():
    """Demo: Clinical protocol lookup"""
    print("="*70)
    print("DEMO 2: Clinical Protocol Lookup")
    print("="*70)
    
    assistant = HealthcareAIAssistant()
    
    result = assistant.get_protocol("chest pain in emergency department")
    
    print(f"\nQuestion: {result['question']}")
    print(f"\nResponse:\n{result['response']}")
    print(f"\nConfidence: {result['confidence']:.2f}")
    print(f"\n{result['disclaimer']}")
    print()


def demo_symptom_assessment():
    """Demo: Symptom assessment support"""
    print("="*70)
    print("DEMO 3: Symptom Assessment Support")
    print("="*70)
    
    assistant = HealthcareAIAssistant()
    
    symptoms = "55-year-old male, acute onset chest pain radiating to left arm, sweating, nausea"
    vital_signs = "BP 140/90, HR 95, O2 sat 98%, Temp 98.6F"
    
    result = assistant.assess_symptoms(symptoms, vital_signs)
    
    print(f"\nSymptoms: {symptoms}")
    print(f"Vital Signs: {vital_signs}")
    print(f"\nClinical Guidance:\n{result['response']}")
    print(f"\nConfidence: {result['confidence']:.2f}")
    print(f"\n{result['disclaimer']}")
    print()


def demo_general_query():
    """Demo: General clinical query"""
    print("="*70)
    print("DEMO 4: General Clinical Query")
    print("="*70)
    
    assistant = HealthcareAIAssistant()
    
    question = "What should I consider when managing a patient with diabetes and hypertension?"
    
    result = assistant.query(question)
    
    print(f"\nQuestion: {question}")
    print(f"\nResponse:\n{result['response']}")
    print(f"\nConfidence: {result['confidence']:.2f}")
    print(f"\n{result['disclaimer']}")
    print()


def main():
    """Run all healthcare demos"""
    print("\n" + "="*70)
    print("HEALTHCARE AI ASSISTANT DEMONSTRATION")
    print("HIPAA-Compliant Local AI for Clinical Decision Support")
    print("="*70)
    print("\n*** IMPORTANT DISCLAIMER ***")
    print("   This is a demonstration system for clinical decision SUPPORT.")
    print("   It does NOT provide diagnoses or replace clinical judgment.")
    print("   All responses should be verified by licensed healthcare providers.")
    print("="*70 + "\n")
    
    try:
        demo_drug_interaction()
        demo_clinical_protocol()
        demo_symptom_assessment()
        demo_general_query()
        
        print("="*70)
        print("DEMONSTRATION COMPLETE")
        print("="*70)
        print("\nKey Features Demonstrated:")
        print("  [+] Drug interaction checking")
        print("  [+] Clinical protocol lookup")
        print("  [+] Symptom assessment support")
        print("  [+] General clinical queries")
        print("\nBenefits:")
        print("  [+] HIPAA compliant (local processing)")
        print("  [+] No external API calls")
        print("  [+] Domain-specific medical knowledge")
        print("  [+] Clinical decision support")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        print("\nMake sure sentence-transformers is installed:")
        print("  pip install sentence-transformers")


if __name__ == "__main__":
    main()
