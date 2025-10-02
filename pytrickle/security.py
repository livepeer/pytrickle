"""
Security middleware for pytrickle servers to handle malformed requests and improve security posture.
"""

import logging
import time
from typing import Dict, Set, Optional, List
from aiohttp import web, hdrs
from aiohttp.http_exceptions import BadHttpMessage
import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SecurityConfig:
    """Configuration for security middleware."""
    
    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 50
    rate_limit_window: int = 60
    
    # Request validation
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    blocked_user_agents: Set[str] = None
    allowed_methods: Set[str] = None
    
    # Security headers
    security_headers_enabled: bool = True
    custom_security_headers: Dict[str, str] = None
    
    # Logging
    log_suspicious_requests: bool = True
    log_all_requests: bool = False
    
    def __post_init__(self):
        if self.blocked_user_agents is None:
            self.blocked_user_agents = set()
        if self.allowed_methods is None:
            self.allowed_methods = {
                'GET', 'POST', 'PUT', 'DELETE', 'HEAD', 'OPTIONS', 'PATCH'
            }
        if self.custom_security_headers is None:
            self.custom_security_headers = {}


class SecurityMiddleware:
    """Middleware to handle security concerns and malformed requests."""
    
    def __init__(self, config: SecurityConfig = None):
        self.config = config or SecurityConfig()
        
        # Rate limiting storage
        self.request_counts: Dict[str, deque] = defaultdict(deque)
        
    async def __call__(self, request: web.Request, handler):
        """Main middleware handler."""
        try:
            # Check rate limiting
            if (self.config.rate_limit_enabled and 
                not await self._check_rate_limit(request)):
                logger.warning(f"Rate limit exceeded for {request.remote}")
                return web.Response(
                    status=429,
                    text="Rate limit exceeded",
                    headers=self._get_security_headers()
                )
            
            # Check request method
            if request.method not in self.config.allowed_methods:
                logger.warning(f"Blocked unsupported method {request.method} from {request.remote}")
                return web.Response(
                    status=405,
                    text="Method not allowed",
                    headers=self._get_security_headers()
                )
            
            # Check user agent blocking
            user_agent = request.headers.get('User-Agent', '')
            if any(blocked in user_agent.lower() for blocked in self.config.blocked_user_agents):
                logger.warning(f"Blocked user agent: {user_agent} from {request.remote}")
                return web.Response(
                    status=403,
                    text="Forbidden",
                    headers=self._get_security_headers()
                )
            
            # Check content length
            content_length = request.headers.get('Content-Length')
            if content_length and int(content_length) > self.config.max_request_size:
                logger.warning(f"Request too large: {content_length} bytes from {request.remote}")
                return web.Response(
                    status=413,
                    text="Request entity too large",
                    headers=self._get_security_headers()
                )
            
            # Process the request
            response = await handler(request)
            
            # Add security headers to response
            if self.config.security_headers_enabled:
                for header, value in self._get_security_headers().items():
                    response.headers[header] = value
                
            return response
            
        except BadHttpMessage as e:
            # Handle HTTP/2 and malformed request errors gracefully
            logger.info(f"Malformed HTTP request from {request.remote}: {e}")
            return web.Response(
                status=400,
                text="Bad Request",
                headers=self._get_security_headers()
            )
        except Exception as e:
            logger.error(f"Security middleware error: {e}")
            return web.Response(
                status=500,
                text="Internal Server Error",
                headers=self._get_security_headers()
            )
    
    async def _check_rate_limit(self, request: web.Request) -> bool:
        """Check if request is within rate limits."""
        client_ip = request.remote
        now = time.time()
        
        # Clean old entries
        window_start = now - self.config.rate_limit_window
        while (self.request_counts[client_ip] and 
               self.request_counts[client_ip][0] < window_start):
            self.request_counts[client_ip].popleft()
        
        # Check current count
        if len(self.request_counts[client_ip]) >= self.config.rate_limit_requests:
            return False
        
        # Add current request
        self.request_counts[client_ip].append(now)
        return True
    
    def _get_security_headers(self) -> Dict[str, str]:
        """Get security headers to add to responses."""
        headers = {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'Content-Security-Policy': "default-src 'self'",
            'Server': 'pytrickle'  # Hide server version
        }
        
        # Add custom headers
        headers.update(self.config.custom_security_headers)
        
        return headers


@web.middleware
async def error_handling_middleware(request: web.Request, handler):
    """Middleware to handle aiohttp protocol errors gracefully."""
    try:
        return await handler(request)
    except BadHttpMessage as e:
        # Log the malformed request attempt
        logger.info(f"Malformed HTTP request from {request.remote}: {str(e)[:100]}")
        return web.Response(
            status=400,
            text="Bad Request - Invalid HTTP protocol",
            headers={
                'Content-Type': 'text/plain',
                'Connection': 'close'
            }
        )
    except asyncio.CancelledError:
        # Handle cancellation gracefully
        logger.debug(f"Request cancelled from {request.remote}")
        raise
    except Exception as e:
        # Log unexpected errors
        logger.error(f"Unexpected error handling request from {request.remote}: {e}")
        return web.Response(
            status=500,
            text="Internal Server Error",
            headers={'Content-Type': 'text/plain'}
        )


def create_logging_middleware(log_all_requests: bool = False, log_suspicious: bool = True):
    """Create logging middleware with configurable options."""
    
    @web.middleware
    async def logging_middleware(request: web.Request, handler):
        """Enhanced logging middleware for security monitoring."""
        start_time = time.time()
        
        # Log suspicious patterns if enabled
        if log_suspicious:
            suspicious_patterns = [
                'PRI *',  # HTTP/2 connection preface
                '\\x',    # Binary data in URL
                '../',    # Path traversal
                'script', # Potential XSS
                'union',  # Potential SQL injection
                'select', # Potential SQL injection
            ]
            
            path_lower = request.path_qs.lower()
            if any(pattern in path_lower for pattern in suspicious_patterns):
                logger.warning(f"Suspicious request pattern from {request.remote}: {request.method} {request.path_qs}")
        
        try:
            response = await handler(request)
            
            # Log the request if enabled
            if log_all_requests:
                duration = time.time() - start_time
                logger.info(
                    f"{request.remote} - {request.method} {request.path_qs} "
                    f"-> {response.status} ({duration:.3f}s)"
                )
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"{request.remote} - {request.method} {request.path_qs} "
                f"-> ERROR: {e} ({duration:.3f}s)"
            )
            raise
    
    return logging_middleware


def create_security_middleware_stack(
    config: Optional[SecurityConfig] = None,
    enable_error_handling: bool = True,
    enable_security_middleware: bool = True,
    enable_logging: bool = True
) -> List:
    """Create a complete security middleware stack for pytrickle servers.
    
    Args:
        config: Security configuration. Uses defaults if None.
        enable_error_handling: Whether to include error handling middleware
        enable_security_middleware: Whether to include security middleware
        enable_logging: Whether to include logging middleware
        
    Returns:
        List of middleware functions in correct order
    """
    middleware_stack = []
    
    if enable_logging:
        security_config = config or SecurityConfig()
        logging_middleware = create_logging_middleware(
            log_all_requests=security_config.log_all_requests,
            log_suspicious=security_config.log_suspicious_requests
        )
        middleware_stack.append(logging_middleware)
    
    if enable_error_handling:
        middleware_stack.append(error_handling_middleware)
    
    if enable_security_middleware:
        security_middleware = SecurityMiddleware(config)
        middleware_stack.append(security_middleware)
    
    return middleware_stack


# Convenience function for backward compatibility
def create_default_security_middleware():
    """Create default security middleware stack with reasonable defaults."""
    return create_security_middleware_stack()
