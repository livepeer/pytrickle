# Security Features in Pytrickle

## Overview

Pytrickle includes built-in security middleware to protect servers against malformed requests, security scans, and various attack vectors. The security features are enabled by default and can be customized through configuration.

## Features

### 1. Protocol Error Handling
- **HTTP/2 Protocol Errors**: Gracefully handles `BadHttpMessage` errors that occur when scanners probe for HTTP/2 support
- **Malformed Request Handling**: Catches and properly responds to malformed HTTP requests
- **Clean Error Responses**: Returns appropriate HTTP status codes without exposing internal server details

### 2. Rate Limiting
- **Per-IP Rate Limiting**: Configurable requests per minute per IP address (default: 50/minute)
- **Sliding Window**: Uses efficient sliding window algorithm for accurate rate limiting
- **Memory Management**: Automatic cleanup of old request records

### 3. Request Validation
- **HTTP Method Filtering**: Only allows standard HTTP methods (GET, POST, PUT, DELETE, HEAD, OPTIONS, PATCH)
- **Content-Length Validation**: Prevents oversized requests (default: 10MB limit)
- **User-Agent Blocking**: Configurable blocking of specific user agent patterns
- **Suspicious Pattern Detection**: Logs requests with suspicious patterns (path traversal, XSS, SQL injection attempts)

### 4. Security Headers
All responses include comprehensive security headers:
- `X-Content-Type-Options: nosniff` - Prevents MIME type sniffing
- `X-Frame-Options: DENY` - Prevents clickjacking attacks
- `X-XSS-Protection: 1; mode=block` - Enables XSS protection
- `Strict-Transport-Security` - Enforces HTTPS connections
- `Referrer-Policy: strict-origin-when-cross-origin` - Controls referrer information
- `Content-Security-Policy: default-src 'self'` - Restricts resource loading
- `Server: pytrickle` - Hides detailed server version information

### 5. Enhanced Logging
- **Security Event Logging**: Logs suspicious requests and security violations
- **Performance Monitoring**: Tracks request duration and response codes
- **IP-based Tracking**: Associates all security events with client IP addresses

## Usage

### Basic Usage (Default Security)

```python
from pytrickle import StreamProcessor

# Security is enabled by default with reasonable settings
processor = StreamProcessor(
    video_processor=my_video_processor,
    port=8000
)
```

### Custom Security Configuration

```python
from pytrickle import StreamProcessor, SecurityConfig

# Create custom security configuration
security_config = SecurityConfig(
    rate_limit_requests=100,  # 100 requests per minute
    rate_limit_window=60,     # 1 minute window
    max_request_size=5 * 1024 * 1024,  # 5MB limit
    blocked_user_agents={'badbot', 'scanner'},
    log_all_requests=True,    # Log all requests (verbose)
    log_suspicious_requests=True
)

processor = StreamProcessor(
    video_processor=my_video_processor,
    port=8000,
    security_config=security_config
)
```

### Disabling Security (Not Recommended)

```python
from pytrickle import StreamProcessor

# Disable security middleware (not recommended for production)
processor = StreamProcessor(
    video_processor=my_video_processor,
    port=8000,
    enable_security=False
)
```

### Using StreamServer Directly

```python
from pytrickle import StreamServer, SecurityConfig

security_config = SecurityConfig(
    rate_limit_requests=25,  # Stricter rate limiting
    custom_security_headers={
        'X-Custom-Header': 'MyValue'
    }
)

server = StreamServer(
    frame_processor=my_processor,
    port=8000,
    security_config=security_config,
    enable_security=True
)
```

## Configuration Options

### SecurityConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rate_limit_enabled` | bool | `True` | Enable/disable rate limiting |
| `rate_limit_requests` | int | `50` | Max requests per window per IP |
| `rate_limit_window` | int | `60` | Rate limit window in seconds |
| `max_request_size` | int | `10MB` | Maximum request size in bytes |
| `blocked_user_agents` | Set[str] | `set()` | User agent patterns to block |
| `allowed_methods` | Set[str] | Standard HTTP | Allowed HTTP methods |
| `security_headers_enabled` | bool | `True` | Enable security headers |
| `custom_security_headers` | Dict[str,str] | `{}` | Additional custom headers |
| `log_suspicious_requests` | bool | `True` | Log suspicious patterns |
| `log_all_requests` | bool | `False` | Log all requests (verbose) |

### StreamProcessor/StreamServer Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `security_config` | SecurityConfig | `None` | Custom security config (uses defaults if None) |
| `enable_security` | bool | `True` | Enable/disable security middleware |

## Security Event Logging

The security middleware provides detailed logging:

```
# Rate limit violations
2025-10-01 16:26:09 [WARNING] Rate limit exceeded for 192.168.1.100

# Suspicious request patterns  
2025-10-01 16:26:10 [WARNING] Suspicious request pattern from 192.168.1.100: GET /../../etc/passwd

# Malformed HTTP requests
2025-10-01 16:26:11 [INFO] Malformed HTTP request from 192.168.1.100: BadHttpMessage(400, message: Pause on PRI/Upgrade)

# Blocked user agents
2025-10-01 16:26:12 [WARNING] Blocked user agent: BadBot/1.0 from 192.168.1.100
```

## Performance Impact

The security middleware is designed for minimal performance impact:
- **Efficient Data Structures**: Uses deque for O(1) rate limiting operations
- **Memory Management**: Automatic cleanup prevents memory leaks
- **Non-blocking**: All operations are asynchronous and non-blocking
- **Minimal Overhead**: < 1ms overhead for normal requests

## Integration with Existing Applications

For applications already using pytrickle, security is automatically enabled when upgrading. To maintain existing behavior:

```python
# Maintain existing behavior (no security)
processor = StreamProcessor(
    video_processor=my_processor,
    enable_security=False  # Explicitly disable
)
```

## Best Practices

1. **Keep Security Enabled**: Always use security middleware in production
2. **Monitor Logs**: Regularly review security event logs
3. **Tune Rate Limits**: Adjust rate limits based on your application's needs
4. **Custom Headers**: Add application-specific security headers as needed
5. **Update Regularly**: Keep pytrickle updated for latest security improvements

## Advanced Usage

### Custom Middleware Stack

```python
from pytrickle.security import create_security_middleware_stack, SecurityConfig

# Create custom middleware stack
security_config = SecurityConfig(rate_limit_requests=25)
middleware_stack = create_security_middleware_stack(
    config=security_config,
    enable_error_handling=True,
    enable_security_middleware=True,
    enable_logging=True
)

# Use with StreamServer
server = StreamServer(
    frame_processor=my_processor,
    middleware=middleware_stack,
    enable_security=False  # Disable default, use custom
)
```

### Blocking Specific Patterns

```python
security_config = SecurityConfig(
    blocked_user_agents={
        'badbot',
        'scanner',
        'crawler'
    },
    # Block requests with suspicious patterns
    log_suspicious_requests=True
)
```

## Migration from Custom Security

If you were using custom security middleware:

```python
# Old approach (custom middleware)
from my_security import create_security_middleware_stack
processor = StreamProcessor(
    video_processor=my_processor,
    middleware=create_security_middleware_stack()
)

# New approach (built-in security)
from pytrickle import SecurityConfig
security_config = SecurityConfig(
    rate_limit_requests=50,
    log_all_requests=False
)
processor = StreamProcessor(
    video_processor=my_processor,
    security_config=security_config
)
```

## Troubleshooting

### Common Issues

1. **Rate Limiting Too Strict**: Increase `rate_limit_requests` or `rate_limit_window`
2. **Legitimate Requests Blocked**: Review `blocked_user_agents` and suspicious pattern detection
3. **Performance Issues**: Disable verbose logging (`log_all_requests=False`)
4. **Security Scan Errors**: These are now handled gracefully and logged appropriately

### Debug Logging

Enable debug logging to see detailed security middleware operation:

```python
import logging
logging.getLogger('pytrickle.security').setLevel(logging.DEBUG)
```
