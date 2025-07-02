#!/usr/bin/env bash

set -e

# UPS Microservice Configuration with DigitalOcean Vault Integration
# This file provides centralized configuration for the UPS microservice
# with support for DigitalOcean's managed vault service

# Create config directories
mkdir -p "./shared/config"
mkdir -p ./config/generated

# DigitalOcean Vault Configuration
# DO Vault typically uses their managed secrets service
export APP_CONFIG_DO_VAULT_ENABLED="${DO_VAULT_ENABLED:-true}"
export APP_CONFIG_DO_VAULT_PROJECT_ID="${DO_VAULT_PROJECT_ID:-}"
export APP_CONFIG_DO_VAULT_API_TOKEN="${DO_VAULT_API_TOKEN:-}"
export APP_CONFIG_DO_VAULT_REGION="${DO_VAULT_REGION:-nyc3}"
export APP_CONFIG_DO_VAULT_ENDPOINT="${DO_VAULT_ENDPOINT:-https://api.digitalocean.com/v2/projects}"

# Traditional Vault fallback (for local development)
export APP_CONFIG_UPS_VAULT_ADDR="${VAULT_ADDR:-}"
export APP_CONFIG_UPS_VAULT_PATH="${VAULT_PATH:-secret/ups-microservice}"
export APP_CONFIG_UPS_VAULT_TOKEN="${VAULT_TOKEN:-}"

# Export configuration as environment variables
export APP_CONFIG_UPS_MICROSERVICE_URL="${UPS_MICROSERVICE_URL:-http://localhost:8000}"
export APP_CONFIG_UPS_MICROSERVICE_HEALTH_URL="${UPS_MICROSERVICE_URL:-http://localhost:8000}/health"
export APP_CONFIG_UPS_MICROSERVICE_TIMEOUT="${UPS_MICROSERVICE_TIMEOUT:-30}"
export APP_CONFIG_UPS_MICROSERVICE_MAX_RETRIES="${UPS_MICROSERVICE_MAX_RETRIES:-3}"

# Environment-specific settings
export APP_CONFIG_UPS_ENVIRONMENT="${UPS_ENVIRONMENT:-production}"

# API Configuration (will be fetched from DO Vault if enabled)
export APP_CONFIG_UPS_API_KEY="${APP_CONFIG_UPS_API_KEY:-${API_KEY:-}}"
export APP_CONFIG_UPS_CLIENT_ID="${APP_CONFIG_UPS_CLIENT_ID:-${UPS_CLIENT_ID:-}}"
export APP_CONFIG_UPS_CLIENT_SECRET="${APP_CONFIG_UPS_CLIENT_SECRET:-${UPS_CLIENT_SECRET:-}}"

# Logging configuration
export APP_CONFIG_UPS_LOG_LEVEL="${LOG_LEVEL:-INFO}"
export APP_CONFIG_UPS_DEBUG="${DEBUG:-false}"

# Service discovery and load balancing
export APP_CONFIG_UPS_SERVICE_NAME="ups-microservice"
export APP_CONFIG_UPS_SERVICE_VERSION="1.0.0"

# Health check configuration
export APP_CONFIG_UPS_HEALTH_CHECK_INTERVAL="${HEALTH_CHECK_INTERVAL:-30}"
export APP_CONFIG_UPS_HEALTH_CHECK_TIMEOUT="${HEALTH_CHECK_TIMEOUT:-5}"

# Rate limiting
export APP_CONFIG_UPS_RATE_LIMIT="${UPS_RATE_LIMIT:-100}"

# Generate UPS API configuration file
cat > ./config/generated/ups_api.yml << EOF
production:
  client_id: ${UPS_CLIENT_ID}
  client_secret: ${UPS_CLIENT_SECRET}
  environment: ${UPS_ENVIRONMENT:-sandbox}
  oauth_url: ${UPS_OAUTH_URL}
  rate_url: ${UPS_RATE_URL}
  request_timeout: ${UPS_REQUEST_TIMEOUT:-30}
  max_retries: ${UPS_MAX_RETRIES:-3}
  rate_limit: ${UPS_RATE_LIMIT:-100}
EOF

# Generate service configuration file
cat > ./config/generated/service.yml << EOF
production:
  microservice_url: ${UPS_MICROSERVICE_URL:-http://localhost:8000}
  health_check_interval: ${HEALTH_CHECK_INTERVAL:-30}
  health_check_timeout: ${HEALTH_CHECK_TIMEOUT:-5}
  log_level: ${LOG_LEVEL:-INFO}
  debug: ${DEBUG:-false}
  api_key: ${API_KEY}
EOF

# Generate DigitalOcean vault configuration file
cat > ./config/generated/vault.yml << EOF
production:
  # DigitalOcean Vault Configuration
  do_vault:
    enabled: ${DO_VAULT_ENABLED:-true}
    project_id: ${DO_VAULT_PROJECT_ID}
    api_token: ${DO_VAULT_API_TOKEN}
    region: ${DO_VAULT_REGION:-nyc3}
    endpoint: ${DO_VAULT_ENDPOINT:-https://api.digitalocean.com/v2/projects}
  # Traditional Vault fallback
  vault_addr: ${VAULT_ADDR}
  vault_path: ${VAULT_PATH:-secret/ups-microservice}
  vault_token: ${VAULT_TOKEN}
  secret_refresh_interval: ${SECRET_REFRESH_INTERVAL:-3600}
EOF

echo "UPS Microservice configuration loaded with DigitalOcean Vault:"
echo "  URL: $APP_CONFIG_UPS_MICROSERVICE_URL"
echo "  Environment: $APP_CONFIG_UPS_ENVIRONMENT"
echo "  DO Vault Enabled: ${DO_VAULT_ENABLED:-true}"
echo "  DO Vault Region: ${DO_VAULT_REGION:-nyc3}"
echo "  Traditional Vault: ${APP_CONFIG_UPS_VAULT_ADDR:-'not configured'}"
echo "  Log Level: $APP_CONFIG_UPS_LOG_LEVEL"