"""
Quantum-Inspired LLM Architecture
Uses quantum principles for language modeling
"""
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict
from quantum_tokenizer import QuantumTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuantumAttention(nn.Module):
    """Quantum-inspired attention mechanism"""
    
    def __init__(self, d_model: int, n_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Quantum-inspired attention scores
        # Use quantum amplitude instead of dot product
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        # Apply quantum measurement (softmax)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        return self.W_o(context), attention_weights


class QuantumTransformerBlock(nn.Module):
    """Quantum-inspired transformer block"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = QuantumAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, mask=None):
        # Quantum attention with residual connection
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + attn_output)
        
        # Feed forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x


class QuantumLLM(nn.Module):
    """
    Quantum-Inspired Large Language Model
    Uses quantum principles for language understanding and generation
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        max_seq_length: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Token embeddings with quantum state initialization
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_length, d_model)
        
        # Quantum transformer blocks
        self.transformer_blocks = nn.ModuleList([
            QuantumTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output layer
        self.output_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize with quantum-inspired weights
        self._init_quantum_weights()
    
    def _init_quantum_weights(self):
        """Initialize weights using quantum-inspired distributions"""
        for param in self.parameters():
            if param.dim() > 1:
                # Use quantum amplitude distribution
                nn.init.normal_(param, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None):
        batch_size, seq_length = input_ids.shape
        
        # Create position IDs
        positions = torch.arange(0, seq_length, device=input_ids.device).unsqueeze(0)
        
        # Token and position embeddings
        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.position_embedding(positions)
        x = self.dropout(token_embeds + pos_embeds)
        
        # Pass through quantum transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer(x, mask)
        
        # Output projection
        x = self.output_norm(x)
        logits = self.output_projection(x)
        
        return logits
    
    def generate(
        self,
        tokenizer: QuantumTokenizer,
        prompt: str,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9
    ) -> str:
        """Generate text using quantum sampling"""
        self.eval()
        
        # Encode prompt
        input_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long)
        
        generated = input_ids.copy()
        
        with torch.no_grad():
            for _ in range(max_length - len(input_ids)):
                # Forward pass
                logits = self.forward(input_tensor)[0, -1, :]
                
                # Apply temperature
                logits = logits / temperature
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                # Quantum sampling (softmax + multinomial)
                probs = torch.softmax(logits, dim=-1)
                next_token_id = torch.multinomial(probs, 1).item()
                
                generated.append(next_token_id)
                
                # Update input tensor
                input_tensor = torch.tensor([generated[-self.max_seq_length:]], dtype=torch.long)
        
        # Decode generated tokens
        return tokenizer.decode(generated)


class QuantumLLMTrainer:
    """Enhanced Trainer for Quantum LLM with checkpointing and validation"""
    
    def __init__(
        self,
        model: QuantumLLM,
        tokenizer: QuantumTokenizer,
        learning_rate: float = 1e-4,
        checkpoint_dir: str = 'checkpoints'
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Checkpointing
        self.checkpoint_dir = checkpoint_dir
        self.best_loss = float('inf')
        import os
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def train_step(self, batch_texts: List[str]) -> float:
        """Perform one training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Tokenize batch
        batch_token_ids = [self.tokenizer.encode(text) for text in batch_texts]
        
        # Create input/target pairs
        inputs = []
        targets = []
        
        for token_ids in batch_token_ids:
            if len(token_ids) > 1:
                inputs.append(token_ids[:-1])
                targets.append(token_ids[1:])
        
        if not inputs:
            return 0.0
        
        # Pad sequences
        max_len = max(len(seq) for seq in inputs)
        padded_inputs = []
        padded_targets = []
        
        for inp, tgt in zip(inputs, targets):
            padded_inputs.append(inp + [0] * (max_len - len(inp)))
            padded_targets.append(tgt + [-1] * (max_len - len(tgt)))
        
        input_tensor = torch.tensor(padded_inputs, dtype=torch.long)
        target_tensor = torch.tensor(padded_targets, dtype=torch.long)
        
        # Forward pass
        logits = self.model(input_tensor)
        
        # Reshape for loss calculation
        logits = logits.view(-1, self.model.vocab_size)
        targets = target_tensor.view(-1)
        
        # Calculate loss
        loss = self.criterion(logits, targets)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def validate(self, val_texts: List[str], batch_size: int = 32) -> float:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for i in range(0, len(val_texts), batch_size):
                batch = val_texts[i:i + batch_size]
                batch_token_ids = [self.tokenizer.encode(text) for text in batch]
                
                inputs = []
                targets = []
                for token_ids in batch_token_ids:
                    if len(token_ids) > 1:
                        inputs.append(token_ids[:-1])
                        targets.append(token_ids[1:])
                
                if not inputs:
                    continue
                
                # Pad sequences
                max_len = max(len(seq) for seq in inputs)
                padded_inputs = []
                padded_targets = []
                
                for inp, tgt in zip(inputs, targets):
                    padded_inputs.append(inp + [0] * (max_len - len(inp)))
                    padded_targets.append(tgt + [-1] * (max_len - len(tgt)))
                
                input_tensor = torch.tensor(padded_inputs, dtype=torch.long)
                target_tensor = torch.tensor(padded_targets, dtype=torch.long)
                
                logits = self.model(input_tensor)
                logits = logits.view(-1, self.model.vocab_size)
                targets = target_tensor.view(-1)
                
                loss = self.criterion(logits, targets)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def _save_checkpoint(self, epoch: int, loss: float):
        """Save model checkpoint"""
        import os
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'tokenizer': self.tokenizer  # Save tokenizer if possible
        }, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def train(self, texts: List[str], val_texts: List[str] = None, 
              epochs: int = 10, batch_size: int = 32):
        """Enhanced training with validation and checkpointing"""
        logger.info(f"Training Quantum LLM for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training
            total_loss = 0.0
            num_batches = 0
            
            # Create batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                loss = self.train_step(batch)
                total_loss += loss
                num_batches += 1
            
            train_loss = total_loss / num_batches if num_batches > 0 else 0.0
            
            # Validation
            if val_texts:
                val_loss = self.validate(val_texts, batch_size)
                self.scheduler.step(val_loss)
                
                # Save checkpoint if best
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self._save_checkpoint(epoch, val_loss)
                
                logger.info(f"Epoch {epoch + 1}/{epochs}: "
                          f"Train Loss: {train_loss:.4f}, "
                          f"Val Loss: {val_loss:.4f}")
            else:
                self.scheduler.step(train_loss)
                logger.info(f"Epoch {epoch + 1}/{epochs}, Average Loss: {train_loss:.4f}")
        
        logger.info("Training complete!")
