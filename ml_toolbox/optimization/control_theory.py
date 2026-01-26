"""
Control Theory & Cybernetics - Inspired by Norbert Wiener

Implements:
- PID Controllers for adaptive learning rates
- Feedback loops for training stability
- System stability analysis
- Adaptive control for hyperparameter tuning
"""
import numpy as np
from typing import Dict, Any, Optional, List, Callable, Tuple
import logging

logger = logging.getLogger(__name__)


class PIDController:
    """
    PID (Proportional-Integral-Derivative) Controller
    
    Used for adaptive learning rate control and system regulation
    """
    
    def __init__(
        self,
        Kp: float = 1.0,  # Proportional gain
        Ki: float = 0.1,  # Integral gain
        Kd: float = 0.01,  # Derivative gain
        setpoint: float = 0.0,  # Target value
        output_limits: Optional[Tuple[float, float]] = None,
        sample_time: float = 1.0
    ):
        """
        Initialize PID Controller
        
        Args:
            Kp: Proportional gain
            Ki: Integral gain
            Kd: Derivative gain
            setpoint: Target value to maintain
            output_limits: (min, max) output limits
            sample_time: Time between updates
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.output_limits = output_limits
        self.sample_time = sample_time
        
        self._integral = 0.0
        self._last_error = 0.0
        self._last_output = 0.0
        self._last_time = None
    
    def update(self, measured_value: float, current_time: Optional[float] = None) -> float:
        """
        Update PID controller and return control output
        
        Args:
            measured_value: Current measured value
            current_time: Current time (for adaptive sample time)
        
        Returns:
            Control output
        """
        error = self.setpoint - measured_value
        
        # Calculate time delta
        if current_time is not None and self._last_time is not None:
            dt = current_time - self._last_time
        else:
            dt = self.sample_time
        
        # Proportional term
        P = self.Kp * error
        
        # Integral term
        self._integral += error * dt
        I = self.Ki * self._integral
        
        # Derivative term
        if dt > 0:
            derivative = (error - self._last_error) / dt
        else:
            derivative = 0.0
        D = self.Kd * derivative
        
        # Calculate output
        output = P + I + D
        
        # Apply output limits
        if self.output_limits:
            output = np.clip(output, self.output_limits[0], self.output_limits[1])
        
        # Update state
        self._last_error = error
        self._last_output = output
        if current_time is not None:
            self._last_time = current_time
        
        return output
    
    def reset(self):
        """Reset controller state"""
        self._integral = 0.0
        self._last_error = 0.0
        self._last_output = 0.0
        self._last_time = None


class AdaptiveLearningRateController:
    """
    PID-based adaptive learning rate controller
    
    Adjusts learning rate based on training feedback
    """
    
    def __init__(
        self,
        initial_lr: float = 0.01,
        target_loss: float = 0.0,
        Kp: float = 0.1,
        Ki: float = 0.01,
        Kd: float = 0.001,
        lr_bounds: Tuple[float, float] = (1e-6, 1.0)
    ):
        """
        Initialize adaptive learning rate controller
        
        Args:
            initial_lr: Initial learning rate
            target_loss: Target loss value
            Kp, Ki, Kd: PID gains
            lr_bounds: (min, max) learning rate bounds
        """
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.target_loss = target_loss
        self.lr_bounds = lr_bounds
        
        self.pid = PIDController(
            Kp=Kp,
            Ki=Ki,
            Kd=Kd,
            setpoint=target_loss,
            output_limits=lr_bounds
        )
        
        self.loss_history = []
        self.lr_history = [initial_lr]
    
    def update(self, current_loss: float) -> float:
        """
        Update learning rate based on current loss
        
        Args:
            current_loss: Current training loss
        
        Returns:
            New learning rate
        """
        self.loss_history.append(current_loss)
        
        # PID controller adjusts learning rate to minimize loss
        lr_adjustment = self.pid.update(current_loss)
        
        # Update learning rate (PID output is adjustment, not absolute)
        # Use proportional adjustment
        if len(self.loss_history) > 1:
            loss_change = self.loss_history[-1] - self.loss_history[-2]
            if loss_change > 0:  # Loss increased
                self.current_lr *= 0.9  # Decrease LR
            else:  # Loss decreased
                self.current_lr *= 1.05  # Increase LR
        
        # Apply bounds
        self.current_lr = np.clip(self.current_lr, self.lr_bounds[0], self.lr_bounds[1])
        self.lr_history.append(self.current_lr)
        
        return self.current_lr
    
    def get_learning_rate(self) -> float:
        """Get current learning rate"""
        return self.current_lr


class TrainingStabilityMonitor:
    """
    Monitor training stability using control theory concepts
    """
    
    def __init__(
        self,
        stability_window: int = 10,
        divergence_threshold: float = 1e6,
        oscillation_threshold: float = 0.1
    ):
        """
        Initialize stability monitor
        
        Args:
            stability_window: Window size for stability checks
            divergence_threshold: Loss value indicating divergence
            oscillation_threshold: Coefficient of variation threshold for oscillation
        """
        self.stability_window = stability_window
        self.divergence_threshold = divergence_threshold
        self.oscillation_threshold = oscillation_threshold
        
        self.loss_history = []
        self.stability_status = 'stable'
    
    def check_stability(self, current_loss: float) -> Dict[str, Any]:
        """
        Check training stability
        
        Args:
            current_loss: Current training loss
        
        Returns:
            Stability status and metrics
        """
        self.loss_history.append(current_loss)
        
        # Check divergence
        if current_loss > self.divergence_threshold:
            self.stability_status = 'diverging'
            return {
                'status': 'diverging',
                'reason': 'loss_exceeded_threshold',
                'loss': current_loss
            }
        
        # Check oscillation
        if len(self.loss_history) >= self.stability_window:
            recent_losses = self.loss_history[-self.stability_window:]
            mean_loss = np.mean(recent_losses)
            std_loss = np.std(recent_losses)
            cv = std_loss / (mean_loss + 1e-10)  # Coefficient of variation
            
            if cv > self.oscillation_threshold:
                self.stability_status = 'oscillating'
                return {
                    'status': 'oscillating',
                    'reason': 'high_variation',
                    'coefficient_of_variation': cv,
                    'mean_loss': mean_loss,
                    'std_loss': std_loss
                }
        
        # Check monotonic improvement
        if len(self.loss_history) >= self.stability_window:
            recent_losses = self.loss_history[-self.stability_window:]
            if all(recent_losses[i] <= recent_losses[i+1] for i in range(len(recent_losses)-1)):
                self.stability_status = 'degrading'
                return {
                    'status': 'degrading',
                    'reason': 'monotonic_increase',
                    'trend': 'increasing'
                }
        
        self.stability_status = 'stable'
        return {
            'status': 'stable',
            'loss': current_loss
        }
    
    def get_recommendations(self) -> List[str]:
        """Get recommendations based on stability status"""
        recommendations = []
        
        if self.stability_status == 'diverging':
            recommendations.extend([
                "Reduce learning rate",
                "Check for gradient explosion",
                "Add gradient clipping",
                "Reduce model complexity"
            ])
        elif self.stability_status == 'oscillating':
            recommendations.extend([
                "Reduce learning rate",
                "Increase batch size",
                "Add momentum",
                "Use learning rate scheduling"
            ])
        elif self.stability_status == 'degrading':
            recommendations.extend([
                "Reduce learning rate",
                "Check data quality",
                "Verify labels",
                "Add regularization"
            ])
        
        return recommendations


class AdaptiveHyperparameterTuner:
    """
    Adaptive hyperparameter tuning using control theory
    """
    
    def __init__(
        self,
        hyperparameter_ranges: Dict[str, Tuple[float, float]],
        objective_function: Callable,
        target_metric: float = 0.0
    ):
        """
        Initialize adaptive hyperparameter tuner
        
        Args:
            hyperparameter_ranges: Dict of {param_name: (min, max)}
            objective_function: Function to minimize (takes hyperparams dict)
            target_metric: Target metric value
        """
        self.hyperparameter_ranges = hyperparameter_ranges
        self.objective_function = objective_function
        self.target_metric = target_metric
        
        # Initialize PID controllers for each hyperparameter
        self.controllers = {}
        for param_name in hyperparameter_ranges:
            self.controllers[param_name] = PIDController(
                Kp=0.1,
                Ki=0.01,
                Kd=0.001,
                setpoint=target_metric
            )
        
        self.history = []
    
    def tune(self, n_iterations: int = 100) -> Dict[str, Any]:
        """
        Run adaptive hyperparameter tuning
        
        Args:
            n_iterations: Number of tuning iterations
        
        Returns:
            Best hyperparameters and history
        """
        # Initialize hyperparameters to middle of ranges
        current_params = {}
        for param_name, (min_val, max_val) in self.hyperparameter_ranges.items():
            current_params[param_name] = (min_val + max_val) / 2
        
        best_params = current_params.copy()
        best_metric = float('inf')
        
        for iteration in range(n_iterations):
            # Evaluate current hyperparameters
            metric = self.objective_function(current_params)
            
            # Update best
            if metric < best_metric:
                best_metric = metric
                best_params = current_params.copy()
            
            # Update each hyperparameter using PID controller
            for param_name, controller in self.controllers.items():
                # PID controller adjusts based on metric
                adjustment = controller.update(metric)
                
                # Apply adjustment to hyperparameter
                min_val, max_val = self.hyperparameter_ranges[param_name]
                current_val = current_params[param_name]
                
                # Adjust proportionally
                new_val = current_val * (1 + adjustment * 0.1)
                current_params[param_name] = np.clip(new_val, min_val, max_val)
            
            self.history.append({
                'iteration': iteration,
                'params': current_params.copy(),
                'metric': metric
            })
        
        return {
            'best_params': best_params,
            'best_metric': best_metric,
            'history': self.history
        }
