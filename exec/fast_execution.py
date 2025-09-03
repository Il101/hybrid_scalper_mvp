"""
Ultra-low latency execution system for scalping
"""
import asyncio
import time
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
import logging

@dataclass
class FastOrder:
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    order_type: str  # 'market', 'limit', 'post_only'
    price: Optional[float] = None
    client_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

@dataclass
class ExecutionResult:
    success: bool
    order_id: Optional[str] = None
    fill_price: Optional[float] = None
    filled_qty: float = 0.0
    error_message: Optional[str] = None
    latency_ms: float = 0.0

class FastExecutor:
    """
    Ultra-low latency execution engine for scalping
    Pre-calculates orders and maintains persistent connections
    """
    
    def __init__(self):
        self.prepared_orders: Dict[str, FastOrder] = {}
        self.connection_pool: Dict[str, Any] = {}
        self.execution_stats = {
            'orders_sent': 0,
            'orders_filled': 0,
            'avg_latency_ms': 0.0,
            'error_count': 0
        }
        self.logger = logging.getLogger(__name__)
    
    def prepare_order_template(self, symbol: str, side: str, base_quantity: float):
        """
        Pre-calculate order parameters for ultra-fast execution
        """
        order_key = f"{symbol}_{side}"
        self.prepared_orders[order_key] = FastOrder(
            symbol=symbol,
            side=side,
            quantity=base_quantity,
            order_type='market'  # Default to market for speed
        )
    
    async def fire_and_forget(self, symbol: str, side: str, 
                             quantity: Optional[float] = None,
                             price: Optional[float] = None) -> ExecutionResult:
        """
        Ultra-low latency execution without waiting for full confirmation
        Returns immediately after sending order
        """
        start_time = time.perf_counter()
        order_key = f"{symbol}_{side}"
        
        try:
            # Use prepared template if available
            if order_key in self.prepared_orders:
                template = self.prepared_orders[order_key]
                order = FastOrder(
                    symbol=template.symbol,
                    side=template.side,
                    quantity=quantity or template.quantity,
                    order_type=template.order_type,
                    price=price
                )
            else:
                order = FastOrder(symbol, side, quantity or 0.0, 'market', price)
            
            # Fire order asynchronously
            order_id = await self._send_order_async(order)
            
            latency = (time.perf_counter() - start_time) * 1000
            self._update_stats(latency, True)
            
            return ExecutionResult(
                success=True,
                order_id=order_id,
                latency_ms=latency
            )
            
        except Exception as e:
            latency = (time.perf_counter() - start_time) * 1000
            self._update_stats(latency, False)
            self.logger.error(f"Fast execution failed: {e}")
            
            return ExecutionResult(
                success=False,
                error_message=str(e),
                latency_ms=latency
            )
    
    async def batch_execute(self, orders: list[FastOrder]) -> list[ExecutionResult]:
        """
        Execute multiple orders simultaneously
        """
        tasks = [self.fire_and_forget(order.symbol, order.side, 
                                    order.quantity, order.price) 
                for order in orders]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to failed results
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append(ExecutionResult(
                    success=False,
                    error_message=str(result)
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _send_order_async(self, order: FastOrder) -> str:
        """
        Actual order sending logic - to be implemented with exchange API
        This is a placeholder that simulates order sending
        """
        # Simulate network latency
        await asyncio.sleep(0.01)  # 10ms simulated latency
        
        # Generate mock order ID
        order_id = f"order_{int(time.time() * 1000000)}"
        
        # In real implementation, this would call exchange API
        # Example: await exchange_client.create_order(order)
        
        return order_id
    
    def _update_stats(self, latency_ms: float, success: bool):
        """Update execution statistics"""
        self.execution_stats['orders_sent'] += 1
        
        if success:
            self.execution_stats['orders_filled'] += 1
        else:
            self.execution_stats['error_count'] += 1
        
        # Update rolling average latency
        current_avg = self.execution_stats['avg_latency_ms']
        total_orders = self.execution_stats['orders_sent']
        
        self.execution_stats['avg_latency_ms'] = (
            (current_avg * (total_orders - 1) + latency_ms) / total_orders
        )
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get current execution statistics"""
        stats = self.execution_stats.copy()
        if stats['orders_sent'] > 0:
            stats['fill_rate'] = stats['orders_filled'] / stats['orders_sent']
            stats['error_rate'] = stats['error_count'] / stats['orders_sent']
        else:
            stats['fill_rate'] = 0.0
            stats['error_rate'] = 0.0
        
        return stats
    
    def reset_stats(self):
        """Reset execution statistics"""
        self.execution_stats = {
            'orders_sent': 0,
            'orders_filled': 0,
            'avg_latency_ms': 0.0,
            'error_count': 0
        }

class LatencyOptimizer:
    """
    Optimize execution latency through various techniques
    """
    
    def __init__(self):
        self.latency_history = []
        self.connection_quality = {}
    
    def record_latency(self, exchange: str, latency_ms: float):
        """Record latency measurement"""
        self.latency_history.append({
            'exchange': exchange,
            'latency_ms': latency_ms,
            'timestamp': time.time()
        })
        
        # Keep only recent history
        cutoff_time = time.time() - 300  # 5 minutes
        self.latency_history = [
            record for record in self.latency_history 
            if record['timestamp'] > cutoff_time
        ]
    
    def get_avg_latency(self, exchange: str, window_sec: int = 60) -> float:
        """Get average latency for an exchange over time window"""
        cutoff_time = time.time() - window_sec
        
        relevant_records = [
            record for record in self.latency_history
            if record['exchange'] == exchange and record['timestamp'] > cutoff_time
        ]
        
        if not relevant_records:
            return 50.0  # Default estimate
        
        return sum(r['latency_ms'] for r in relevant_records) / len(relevant_records)
    
    def recommend_connection_optimization(self, exchange: str) -> Dict[str, str]:
        """Provide recommendations for latency optimization"""
        avg_latency = self.get_avg_latency(exchange)
        recommendations = {}
        
        if avg_latency > 100:
            recommendations['latency'] = "HIGH - Consider co-location or better network"
        elif avg_latency > 50:
            recommendations['latency'] = "MEDIUM - Optimize network routing"
        else:
            recommendations['latency'] = "GOOD - Latency within acceptable range"
        
        # Additional recommendations based on patterns
        recent_records = [
            r for r in self.latency_history
            if r['exchange'] == exchange and r['timestamp'] > time.time() - 60
        ]
        
        if len(recent_records) > 5:
            latencies = [r['latency_ms'] for r in recent_records]
            std_dev = (sum((l - avg_latency)**2 for l in latencies) / len(latencies))**0.5
            
            if std_dev > 20:
                recommendations['stability'] = "UNSTABLE - High latency variance"
            else:
                recommendations['stability'] = "STABLE - Consistent latency"
        
        return recommendations

# Global executor instance for reuse
_global_executor = None

def get_fast_executor() -> FastExecutor:
    """Get singleton fast executor instance"""
    global _global_executor
    if _global_executor is None:
        _global_executor = FastExecutor()
    return _global_executor

async def execute_immediately(symbol: str, side: str, quantity: float) -> ExecutionResult:
    """
    Convenience function for immediate execution
    """
    executor = get_fast_executor()
    return await executor.fire_and_forget(symbol, side, quantity)
