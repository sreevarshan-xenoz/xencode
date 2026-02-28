# Security Audit Report

**Generated:** 2026-02-28T13:08:13.684340

**Scan Path:** demo_vulnerable.py


## Summary

- **Total Vulnerabilities:** 3
- **Code Vulnerabilities:** 3
- **Dependency Vulnerabilities:** 0

### By Risk Level

- 🔴 **Critical:** 2
- 🟠 **High:** 0
- 🟡 **Medium:** 0
- 🟢 **Low:** 1
- ℹ️  **Info:** 0

## Code Vulnerabilities


### Hardcoded password
- **Risk Level:** CRITICAL
- **Type:** hardcoded_secrets
- **Location:** demo_vulnerable.py:2
- **Code:** `password = "secret123"`
- **Fix:** Store secrets in environment variables or a secure vault. Example: password = os.environ.get('DB_PASSWORD')

### Hardcoded API key
- **Risk Level:** CRITICAL
- **Type:** hardcoded_secrets
- **Location:** demo_vulnerable.py:3
- **Code:** `api_key = "1234567890"`
- **Fix:** Store secrets in environment variables or a secure vault. Example: password = os.environ.get('DB_PASSWORD')

### Possible hardcoded password: 'secret123'
- **Risk Level:** LOW
- **Type:** security_misconfiguration
- **Location:** .\demo_vulnerable.py:2
- **Code:** `1 
2 password = "secret123"
3 api_key = "1234567890"`
- **Fix:** https://bandit.readthedocs.io/en/1.8.6/plugins/b105_hardcoded_password_string.html

## Recommendations

- ⚠️  URGENT: 2 critical vulnerabilities found. Address these immediately before deploying to production.
- 🔑 Found 2 hardcoded secrets. Move all secrets to environment variables or a secure vault.
- 📚 Review OWASP Top 10 guidelines: https://owasp.org/www-project-top-ten/
- 🔍 Consider implementing automated security testing in your CI/CD pipeline.