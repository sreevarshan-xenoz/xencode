# Xencode API Documentation

## Overview

The Xencode API provides comprehensive endpoints for AI-powered development tools, workspace management, analytics, monitoring, and plugin systems. This documentation covers all available endpoints, authentication, and usage examples.

## Base URL

```
https://api.xencode.dev/api/v1
```

## Authentication

All API endpoints require authentication using JWT tokens:

```http
Authorization: Bearer <your_jwt_token>
```

## API Endpoints

### Analytics API

#### GET /analytics/overview
Get system analytics overview with key metrics and insights.

**Response:**
```json
{
  "total_events": 15420,
  "active_users": 156,
  "system_health_score": 0.96,
  "performance_metrics": {
    "avg_response_time_ms": 45.3,
    "requests_per_second": 125.8,
    "error_rate": 0.012
  }
}
```

#### POST /analytics/metrics
Record performance metrics for monitoring and analysis.

**Request:**
```json
{
  "name": "api_response_time",
  "value": 150.5,
  "metric_type": "gauge",
  "labels": {
    "endpoint": "/api/v1/workspaces",
    "method": "GET"
  }
}
```

#### GET /analytics/events
Retrieve analytics events with filtering and pagination.

**Query Parameters:**
- `event_type` (string): Filter by event type
- `user_id` (string): Filter by user ID
- `start_date` (string): Start date (ISO format)
- `end_date` (string): End date (ISO format)
- `limit` (integer): Number of results (default: 100)

### Monitoring API

#### GET /monitoring/health
Get comprehensive system health status.

**Response:**
```json
{
  "overall_status": "healthy",
  "health_score": 0.96,
  "components": {
    "database": "healthy",
    "cache": "healthy",
    "external_apis": "degraded"
  },
  "uptime_hours": 24.5
}
```

#### GET /monitoring/resources
Get system resource utilization metrics.

**Response:**
```json
[
  {
    "resource_type": "cpu",
    "utilization_percent": 23.7,
    "available_cores": 8,
    "load_average": [1.2, 1.5, 1.8]
  },
  {
    "resource_type": "memory",
    "utilization_percent": 68.2,
    "used_gb": 12.5,
    "total_gb": 32.0
  }
]
```

### Plugin API

#### GET /plugins
List all available plugins with their status and metadata.

**Response:**
```json
[
  {
    "id": "file-operations",
    "name": "File Operations Plugin",
    "version": "1.2.0",
    "status": "enabled",
    "description": "Advanced file system operations",
    "capabilities": ["read", "write", "search", "analyze"]
  }
]
```

#### POST /plugins/{plugin_id}/execute
Execute a plugin method with specified parameters.

**Request:**
```json
{
  "method": "analyze_file",
  "args": ["/path/to/file.py"],
  "kwargs": {
    "include_metrics": true,
    "depth": "detailed"
  }
}
```

### Workspace API

#### POST /workspaces
Create a new collaborative workspace.

**Request:**
```json
{
  "name": "ML Project Workspace",
  "description": "Machine learning project collaboration",
  "settings": {
    "auto_save_enabled": true,
    "real_time_sync": true
  },
  "collaborators": ["user1@example.com", "user2@example.com"]
}
```

#### GET /workspaces/{workspace_id}
Get detailed workspace information including files and collaboration status.

#### WebSocket /workspaces/{workspace_id}/ws
Real-time collaboration WebSocket endpoint for live synchronization.

### Code Analysis API

#### POST /code/analyze
Analyze code for quality, security, and performance issues.

**Request:**
```json
{
  "code": "def hello_world():\n    print('Hello, World!')",
  "language": "python",
  "analysis_types": ["syntax", "style", "security", "performance"]
}
```

### Document Processing API

#### POST /documents/process
Process documents and extract structured content.

**Request:**
```json
{
  "document_url": "https://example.com/document.pdf",
  "processing_options": {
    "extract_text": true,
    "extract_images": false,
    "ocr_enabled": true
  }
}
```

## Error Handling

All API endpoints return consistent error responses:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input parameters",
    "details": {
      "field": "email",
      "issue": "Invalid email format"
    }
  }
}
```

### HTTP Status Codes

- `200` - Success
- `201` - Created
- `400` - Bad Request
- `401` - Unauthorized
- `403` - Forbidden
- `404` - Not Found
- `422` - Validation Error
- `429` - Rate Limited
- `500` - Internal Server Error

## Rate Limiting

API requests are rate limited to ensure fair usage:

- **Standard endpoints**: 1000 requests per hour
- **Analytics endpoints**: 5000 requests per hour
- **WebSocket connections**: 10 concurrent connections per user

Rate limit headers are included in responses:
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1640995200
```

## SDK Examples

### Python SDK

```python
from xencode_sdk import XencodeClient

client = XencodeClient(api_key="your_api_key")

# Create workspace
workspace = client.workspaces.create({
    "name": "My Project",
    "description": "Project workspace"
})

# Analyze code
analysis = client.code.analyze(
    code="print('hello')",
    language="python"
)

# Get analytics
metrics = client.analytics.get_overview()
```

### JavaScript SDK

```javascript
import { XencodeClient } from '@xencode/sdk';

const client = new XencodeClient({ apiKey: 'your_api_key' });

// Create workspace
const workspace = await client.workspaces.create({
  name: 'My Project',
  description: 'Project workspace'
});

// Real-time collaboration
const ws = client.workspaces.connect(workspace.id);
ws.on('change', (change) => {
  console.log('Workspace updated:', change);
});
```

## Webhooks

Configure webhooks to receive real-time notifications:

### Webhook Events

- `workspace.created`
- `workspace.updated`
- `plugin.executed`
- `analysis.completed`
- `alert.triggered`

### Webhook Payload

```json
{
  "event": "workspace.created",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "workspace_id": "ws_123",
    "name": "New Workspace",
    "created_by": "user_456"
  }
}
```

## Performance Optimization

### Caching

- API responses are cached for 5 minutes
- Use `Cache-Control: no-cache` header to bypass cache
- ETags are supported for conditional requests

### Pagination

Large result sets are paginated:

```json
{
  "data": [...],
  "pagination": {
    "page": 1,
    "per_page": 100,
    "total": 1500,
    "total_pages": 15
  }
}
```

### Filtering and Sorting

Most list endpoints support filtering and sorting:

```http
GET /api/v1/analytics/events?event_type=user_action&sort=timestamp&order=desc
```

## Security

### API Security Features

- JWT-based authentication
- Rate limiting and DDoS protection
- Input validation and sanitization
- SQL injection prevention
- XSS protection
- CORS configuration

### Best Practices

1. **Store API keys securely** - Never commit API keys to version control
2. **Use HTTPS only** - All API calls must use HTTPS
3. **Validate responses** - Always validate API responses in your application
4. **Handle errors gracefully** - Implement proper error handling and retry logic
5. **Monitor usage** - Track API usage and set up alerts for unusual activity

## Support and Resources

- **API Status**: https://status.xencode.dev
- **Developer Portal**: https://developers.xencode.dev
- **Community Forum**: https://community.xencode.dev
- **Support Email**: api-support@xencode.dev

## Changelog

### v3.0.0 (2024-01-15)
- Added comprehensive analytics API
- Implemented real-time collaboration WebSockets
- Enhanced security and monitoring features
- Added plugin marketplace integration

### v2.1.0 (2023-12-01)
- Added code analysis endpoints
- Improved workspace management
- Enhanced error handling and validation

### v2.0.0 (2023-10-15)
- Major API restructure
- Added authentication and authorization
- Implemented rate limiting
- Added webhook support