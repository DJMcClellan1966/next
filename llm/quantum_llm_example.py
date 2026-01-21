"""
Example usage of Quantum Tokenizer and LLM
"""
from quantum_tokenizer import QuantumTokenizer
from quantum_llm import QuantumLLM, QuantumLLMTrainer
import torch

def main():
    # Sample training texts (in practice, use a large corpus)
    training_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "In the beginning was the Word, and the Word was with God.",
        "Love your neighbor as yourself.",
        "Faith, hope, and love, but the greatest of these is love.",
        "For God so loved the world that he gave his one and only Son.",
        # Add more texts for better training
    ] * 100  # Repeat for demonstration
    
    print("=" * 60)
    print("Quantum Tokenizer & LLM Example")
    print("=" * 60)
    
    # Step 1: Train tokenizer
    print("\n1. Training Quantum Tokenizer...")
    tokenizer = QuantumTokenizer(vocab_size=1000, dimension=128)
    tokenizer.train(training_texts, min_frequency=2)
    
    # Test tokenization
    test_text = "The Word was with God"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"Original: {test_text}")
    print(f"Encoded: {encoded[:10]}...")  # Show first 10 tokens
    print(f"Decoded: {decoded}")
    
    # Test quantum features
    print("\n2. Testing Quantum Features...")
    if "word" in tokenizer.vocab:
        measurement = tokenizer.measure_token("word")
        print(f"Token 'word' measurement: {measurement['probability']:.4f}")
        
        entangled = tokenizer.get_entangled_tokens("word", top_k=5)
        print(f"Entangled tokens with 'word': {entangled[:3]}")
    
    # Step 2: Create and train LLM
    print("\n3. Creating Quantum LLM...")
    vocab_size = len(tokenizer.vocab)
    model = QuantumLLM(
        vocab_size=vocab_size,
        d_model=256,
        n_heads=4,
        n_layers=3,
        d_ff=1024,
        max_seq_length=128
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Step 3: Train the model
    print("\n4. Training Quantum LLM...")
    trainer = QuantumLLMTrainer(model, tokenizer, learning_rate=1e-3)
    trainer.train(training_texts, epochs=5, batch_size=8)
    
    # Step 4: Generate text
    print("\n5. Generating Text...")
    prompt = "The Word"
    generated = model.generate(
        tokenizer,
        prompt,
        max_length=20,
        temperature=0.8,
        top_k=10
    )
    
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated}")
    
    # Save tokenizer
    print("\n6. Saving tokenizer...")
    tokenizer.save("quantum_tokenizer.json")
    print("Tokenizer saved!")
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
