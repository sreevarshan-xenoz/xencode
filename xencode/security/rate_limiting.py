"""
Rate limiting utilities for Xencode
Provides rate limiting functionality to prevent abuse
"""
import time
import threading
from collections import defaultdict, deque
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta


class RateLimiter:
    """A sliding window rate limiter to prevent API abuse"""
    
    def __init__(self, max_requests: int = 10, window_size: int = 60):
        """
        Initialize the rate limiter.
        
        Args:
            max_requests: Maximum number of requests allowed in the window
            window_size: Time window size in seconds
        """
        self.max_requests = max_requests
        self.window_size = window_size
        self.requests: Dict[str, deque] = defaultdict(lambda: deque())
        self.lock = threading.Lock()
    
    def is_allowed(self, identifier: str) -> Tuple[bool, Optional[int]]:
        """
        Check if a request from the given identifier is allowed.
        
        Args:
            identifier: Unique identifier for the requester (e.g., IP address, user ID)
            
        Returns:
            Tuple of (is_allowed, seconds_to_wait)
        """
        with self.lock:
            now = time.time()
            requests = self.requests[identifier]
            
            # Remove requests that are outside the current window
            while requests and requests[0] <= now - self.window_size:
                requests.popleft()
            
            # Check if we've exceeded the limit
            if len(requests) >= self.max_requests:
                # Calculate when the next request will be allowed
                oldest_request = requests[0] if requests else now
                next_allowed_time = oldest_request + self.window_size
                seconds_to_wait = int(next_allowed_time - now) + 1
                return False, seconds_to_wait
            
            # Add the current request
            requests.append(now)
            return True, None
    
    def get_remaining_requests(self, identifier: str) -> int:
        """
        Get the number of remaining requests for the identifier in the current window.
        
        Args:
            identifier: Unique identifier for the requester
            
        Returns:
            Number of remaining requests
        """
        with self.lock:
            now = time.time()
            requests = self.requests[identifier]
            
            # Remove requests that are outside the current window
            while requests and requests[0] <= now - self.window_size:
                requests.popleft()
            
            return self.max_requests - len(requests)
    
    def reset(self, identifier: str) -> None:
        """
        Reset the rate limit for a specific identifier.
        
        Args:
            identifier: Unique identifier to reset
        """
        with self.lock:
            if identifier in self.requests:
                del self.requests[identifier]


class TokenBucketRateLimiter:
    """Token bucket rate limiter implementation"""
    
    def __init__(self, capacity: int, refill_rate: float):
        """
        Initialize the token bucket rate limiter.
        
        Args:
            capacity: Maximum number of tokens in the bucket
            refill_rate: Number of tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.buckets: Dict[str, Dict] = defaultdict(lambda: {
            'tokens': capacity,
            'last_refill': time.time()
        })
        self.lock = threading.Lock()
    
    def is_allowed(self, identifier: str) -> bool:
        """
        Check if a request from the given identifier is allowed.
        
        Args:
            identifier: Unique identifier for the requester
            
        Returns:
            True if allowed, False otherwise
        """
        with self.lock:
            bucket = self.buckets[identifier]
            now = time.time()
            
            # Calculate tokens to add based on time passed
            time_passed = now - bucket['last_refill']
            tokens_to_add = time_passed * self.refill_rate
            
            # Update bucket
            bucket['tokens'] = min(self.capacity, bucket['tokens'] + tokens_to_add)
            bucket['last_refill'] = now
            
            # Check if we can consume a token
            if bucket['tokens'] >= 1:
                bucket['tokens'] -= 1
                return True
            
            return False
    
    def get_reset_time(self, identifier: str) -> float:
        """
        Get the time when the next request will be allowed.
        
        Args:
            identifier: Unique identifier for the requester
            
        Returns:
            Unix timestamp when the next request will be allowed
        """
        with self.lock:
            bucket = self.buckets[identifier]
            now = time.time()
            
            # Calculate tokens to add based on time passed
            time_passed = now - bucket['last_refill']
            tokens_to_add = time_passed * self.refill_rate
            
            # Update bucket conceptually
            current_tokens = min(self.capacity, bucket['tokens'] + tokens_to_add)
            
            # If we have enough tokens, next request is allowed now
            if current_tokens >= 1:
                return now
            
            # Otherwise, calculate when we'll have enough tokens
            tokens_needed = 1 - current_tokens
            time_needed = tokens_needed / self.refill_rate
            return now + time_needed


class RateLimitMiddleware:
    """Middleware to integrate rate limiting into API flows"""
    
    def __init__(self, rate_limiter: RateLimiter):
        """
        Initialize the middleware.
        
        Args:
            rate_limiter: Rate limiter instance to use
        """
        self.rate_limiter = rate_limiter
    
    def check_request(self, identifier: str, endpoint: str = "/") -> Tuple[bool, Optional[Dict]]:
        """
        Check if a request should be allowed.
        
        Args:
            identifier: Unique identifier for the requester
            endpoint: API endpoint being accessed
            
        Returns:
            Tuple of (is_allowed, response_data_if_limited)
        """
        # Create a composite key for endpoint-specific rate limiting
        composite_key = f"{identifier}:{endpoint}"
        
        allowed, wait_time = self.rate_limiter.is_allowed(composite_key)
        
        if not allowed:
            return False, {
                "error": "Rate limit exceeded",
                "message": f"Too many requests. Please try again in {wait_time} seconds.",
                "retry_after": wait_time,
                "code": 429
            }
        
        return True, None


# Global rate limiter instances
default_rate_limiter = RateLimiter(max_requests=10, window_size=60)  # 10 requests per minute
strict_rate_limiter = RateLimiter(max_requests=5, window_size=60)   # 5 requests per minute for sensitive endpoints
token_bucket_limiter = TokenBucketRateLimiter(capacity=10, refill_rate=1.0)  # 10 tokens, refilled at 1 per second


def get_default_rate_limiter() -> RateLimiter:
    """Get the default rate limiter instance"""
    return default_rate_limiter


def get_strict_rate_limiter() -> RateLimiter:
    """Get the strict rate limiter instance"""
    return strict_rate_limiter


def get_token_bucket_limiter() -> TokenBucketRateLimiter:
    """Get the token bucket rate limiter instance"""
    return token_bucket_limiter


def check_rate_limit(identifier: str, endpoint: str = "/", 
                    rate_limiter: RateLimiter = None) -> Tuple[bool, Optional[Dict]]:
    """
    Convenience function to check rate limits.
    
    Args:
        identifier: Unique identifier for the requester
        endpoint: API endpoint being accessed
        rate_limiter: Specific rate limiter to use (uses default if None)
        
    Returns:
        Tuple of (is_allowed, response_data_if_limited)
    """
    if rate_limiter is None:
        rate_limiter = get_default_rate_limiter()
    
    middleware = RateLimitMiddleware(rate_limiter)
    return middleware.check_request(identifier, endpoint)


def get_remaining_requests(identifier: str, endpoint: str = "/") -> int:
    """
    Get the number of remaining requests for an identifier.
    
    Args:
        identifier: Unique identifier for the requester
        endpoint: API endpoint being accessed
        
    Returns:
        Number of remaining requests
    """
    # Use the same composite key as in check_request
    composite_key = f"{identifier}:{endpoint}"
    return default_rate_limiter.get_remaining_requests(composite_key)