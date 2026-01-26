"""
Neural Lace - Direct Neural Interface

Inspired by: Iain M. Banks' "Culture" series, "The Matrix"

Implements:
- Direct Model-Data Interface
- Real-Time Streaming Learning
- Adaptive Neural Connections
- Bidirectional Communication
- Neural Plasticity
"""
import numpy as np
from typing import List, Dict, Any, Optional, Callable, Iterator
import logging
from collections import deque
import threading
import time

logger = logging.getLogger(__name__)


class NeuralThread:
    """
    Neural Thread - Direct connection between model and data source
    """
    
    def __init__(
        self,
        data_source: Any,
        model: Any,
        connection_strength: float = 1.0,
        plasticity_rate: float = 0.1
    ):
        """
        Initialize neural thread
        
        Args:
            data_source: Data source (iterator, generator, or callable)
            model: Model to connect
            connection_strength: Initial connection strength
            plasticity_rate: Rate of connection adaptation
        """
        self.data_source = data_source
        self.model = model
        self.connection_strength = connection_strength
        self.plasticity_rate = plasticity_rate
        self.connection_history = []
        self.data_flow = deque(maxlen=1000)
        self.active = False
    
    def connect(self):
        """Establish connection"""
        self.active = True
        self.connection_history.append({
            'timestamp': time.time(),
            'action': 'connected',
            'strength': self.connection_strength
        })
    
    def disconnect(self):
        """Disconnect thread"""
        self.active = False
        self.connection_history.append({
            'timestamp': time.time(),
            'action': 'disconnected'
        })
    
    def adapt_connection(self, performance: float):
        """
        Adapt connection strength based on performance (plasticity)
        
        Args:
            performance: Performance metric (higher = better)
        """
        # Hebbian-like plasticity: strengthen if performance good
        if performance > 0.8:
            self.connection_strength = min(1.0, self.connection_strength + self.plasticity_rate)
        elif performance < 0.5:
            self.connection_strength = max(0.1, self.connection_strength - self.plasticity_rate)
        
        self.connection_history.append({
            'timestamp': time.time(),
            'action': 'adapted',
            'strength': self.connection_strength,
            'performance': performance
        })
    
    def read_data(self, n_samples: int = 1) -> List[Any]:
        """Read data from source"""
        if not self.active:
            return []
        
        data = []
        try:
            if hasattr(self.data_source, '__iter__'):
                for _ in range(n_samples):
                    try:
                        item = next(self.data_source)
                        data.append(item)
                        self.data_flow.append({
                            'timestamp': time.time(),
                            'direction': 'in',
                            'data': item
                        })
                    except StopIteration:
                        break
            elif callable(self.data_source):
                for _ in range(n_samples):
                    item = self.data_source()
                    data.append(item)
                    self.data_flow.append({
                        'timestamp': time.time(),
                        'direction': 'in',
                        'data': item
                    })
        except Exception as e:
            logger.warning(f"Error reading data: {e}")
        
        return data
    
    def write_data(self, data: Any):
        """Write data back to source (bidirectional)"""
        if not self.active:
            return False
        
        try:
            if hasattr(self.data_source, 'write'):
                self.data_source.write(data)
            elif hasattr(self.data_source, 'append'):
                self.data_source.append(data)
            
            self.data_flow.append({
                'timestamp': time.time(),
                'direction': 'out',
                'data': data
            })
            return True
        except Exception as e:
            logger.warning(f"Error writing data: {e}")
            return False


class NeuralLace:
    """
    Neural Lace - Network of neural threads connecting models to data
    """
    
    def __init__(self):
        """Initialize neural lace"""
        self.threads = {}
        self.thread_network = {}  # Thread connections
        self.streaming_active = False
        self.streaming_thread = None
    
    def create_thread(
        self,
        thread_id: str,
        data_source: Any,
        model: Any,
        connection_strength: float = 1.0
    ) -> NeuralThread:
        """
        Create a neural thread
        
        Args:
            thread_id: Unique thread identifier
            data_source: Data source
            model: Model to connect
            connection_strength: Initial connection strength
        
        Returns:
            Created neural thread
        """
        thread = NeuralThread(data_source, model, connection_strength)
        self.threads[thread_id] = thread
        self.thread_network[thread_id] = []
        return thread
    
    def connect_threads(self, thread_id1: str, thread_id2: str):
        """Connect two threads (inter-thread communication)"""
        if thread_id1 in self.thread_network and thread_id2 in self.thread_network:
            self.thread_network[thread_id1].append(thread_id2)
            self.thread_network[thread_id2].append(thread_id1)
    
    def activate_all(self):
        """Activate all threads"""
        for thread in self.threads.values():
            thread.connect()
    
    def deactivate_all(self):
        """Deactivate all threads"""
        for thread in self.threads.values():
            thread.disconnect()
    
    def stream_learn(
        self,
        thread_id: str,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        max_iterations: Optional[int] = None
    ):
        """
        Real-time streaming learning
        
        Args:
            thread_id: Thread to use for learning
            learning_rate: Learning rate
            batch_size: Batch size for learning
            max_iterations: Maximum iterations (None for infinite)
        """
        if thread_id not in self.threads:
            raise ValueError(f"Thread {thread_id} not found")
        
        thread = self.threads[thread_id]
        thread.connect()
        
        iteration = 0
        while (max_iterations is None or iteration < max_iterations) and thread.active:
            # Read data batch
            data_batch = thread.read_data(batch_size)
            
            if not data_batch:
                time.sleep(0.1)  # Wait for data
                continue
            
            # Prepare data
            if isinstance(data_batch[0], tuple):
                X = np.array([d[0] for d in data_batch])
                y = np.array([d[1] for d in data_batch])
            else:
                X = np.array(data_batch)
                y = None
            
            # Learn from data
            try:
                if hasattr(thread.model, 'partial_fit'):
                    if y is not None:
                        thread.model.partial_fit(X, y)
                    else:
                        thread.model.partial_fit(X)
                elif hasattr(thread.model, 'fit'):
                    # For non-streaming models, use small batches
                    if y is not None:
                        thread.model.fit(X, y)
                    else:
                        thread.model.fit(X)
                
                # Adapt connection based on performance
                if hasattr(thread.model, 'score') and y is not None:
                    score = thread.model.score(X, y)
                    thread.adapt_connection(score)
            except Exception as e:
                logger.warning(f"Learning error: {e}")
            
            iteration += 1
    
    def start_streaming(self, thread_id: str, **kwargs):
        """Start streaming learning in background thread"""
        self.streaming_active = True
        self.streaming_thread = threading.Thread(
            target=self.stream_learn,
            args=(thread_id,),
            kwargs=kwargs,
            daemon=True
        )
        self.streaming_thread.start()
    
    def stop_streaming(self):
        """Stop streaming learning"""
        self.streaming_active = False
        if self.streaming_thread:
            self.streaming_thread.join(timeout=1.0)
    
    def get_thread_status(self, thread_id: str) -> Dict[str, Any]:
        """Get status of a thread"""
        if thread_id not in self.threads:
            return {}
        
        thread = self.threads[thread_id]
        return {
            'active': thread.active,
            'connection_strength': thread.connection_strength,
            'data_flow_count': len(thread.data_flow),
            'connection_history_count': len(thread.connection_history)
        }
    
    def get_lace_status(self) -> Dict[str, Any]:
        """Get status of entire neural lace"""
        return {
            'threads': len(self.threads),
            'active_threads': sum(1 for t in self.threads.values() if t.active),
            'streaming_active': self.streaming_active,
            'thread_statuses': {
                tid: self.get_thread_status(tid)
                for tid in self.threads.keys()
            }
        }


class DirectNeuralInterface:
    """
    Direct Neural Interface - Seamless model-data connection
    """
    
    def __init__(self, model: Any, data_source: Any):
        """
        Initialize direct neural interface
        
        Args:
            model: Model to interface
            data_source: Data source
        """
        self.model = model
        self.data_source = data_source
        self.lace = NeuralLace()
        self.thread = self.lace.create_thread('main', data_source, model)
        self.thread.connect()
    
    def predict_stream(self, n_samples: int = 1) -> List[Any]:
        """Predict on streaming data"""
        data = self.thread.read_data(n_samples)
        if not data:
            return []
        
        X = np.array([d[0] if isinstance(d, tuple) else d for d in data])
        predictions = self.model.predict(X)
        return predictions.tolist()
    
    def learn_stream(self, n_samples: int = 1):
        """Learn from streaming data"""
        data = self.thread.read_data(n_samples)
        if not data:
            return
        
        if isinstance(data[0], tuple):
            X = np.array([d[0] for d in data])
            y = np.array([d[1] for d in data])
            
            if hasattr(self.model, 'partial_fit'):
                self.model.partial_fit(X, y)
            else:
                self.model.fit(X, y)
        else:
            X = np.array(data)
            if hasattr(self.model, 'partial_fit'):
                self.model.partial_fit(X)
            else:
                self.model.fit(X)
    
    def feedback_loop(self, n_iterations: int = 100):
        """
        Bidirectional feedback loop: read → predict → write back
        
        Args:
            n_iterations: Number of iterations
        """
        for i in range(n_iterations):
            # Read data
            data = self.thread.read_data(1)
            if not data:
                continue
            
            X = np.array([data[0][0] if isinstance(data[0], tuple) else data[0]])
            
            # Predict
            prediction = self.model.predict(X)[0]
            
            # Write prediction back (bidirectional)
            self.thread.write_data(prediction)
            
            # Learn if labels available
            if isinstance(data[0], tuple):
                y = np.array([data[0][1]])
                if hasattr(self.model, 'partial_fit'):
                    self.model.partial_fit(X, y)
    
    def get_interface_status(self) -> Dict[str, Any]:
        """Get interface status"""
        return {
            'thread_status': self.lace.get_thread_status('main'),
            'model_type': type(self.model).__name__,
            'data_source_type': type(self.data_source).__name__
        }
