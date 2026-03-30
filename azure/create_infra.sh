#!/usr/bin/env bash
set -euo pipefail

RESOURCE_GROUP="${1:?Usage: create_infra.sh <resource-group> <location> <acr-name> <containerapp-env>}"
LOCATION="${2:?Usage: create_infra.sh <resource-group> <location> <acr-name> <containerapp-env>}"
ACR_NAME="${3:?Usage: create_infra.sh <resource-group> <location> <acr-name> <containerapp-env>}"
CONTAINERAPP_ENV="${4:?Usage: create_infra.sh <resource-group> <location> <acr-name> <containerapp-env>}"

az extension add --name containerapp --upgrade >/dev/null
az provider register --namespace Microsoft.App >/dev/null
az provider register --namespace Microsoft.OperationalInsights >/dev/null

az group create --name "$RESOURCE_GROUP" --location "$LOCATION" >/dev/null
az acr create --name "$ACR_NAME" --resource-group "$RESOURCE_GROUP" --sku Basic --admin-enabled true >/dev/null
az containerapp env create --name "$CONTAINERAPP_ENV" --resource-group "$RESOURCE_GROUP" --location "$LOCATION" >/dev/null

SUBSCRIPTION_ID="$(az account show --query id -o tsv)"

echo "[INFO] ACR credentials (用于 GitHub Secrets AZURE_ACR_USERNAME / AZURE_ACR_PASSWORD):"
az acr credential show --name "$ACR_NAME" --query '{username:username,password:passwords[0].value}' -o json

echo

echo "[INFO] 创建 GitHub Actions 用的 Service Principal，请把输出 JSON 存为 GitHub Secret: AZURE_CREDENTIALS"
az ad sp create-for-rbac \
  --name "github-${RESOURCE_GROUP}-deploy" \
  --role contributor \
  --scopes "/subscriptions/${SUBSCRIPTION_ID}/resourceGroups/${RESOURCE_GROUP}" \
  --sdk-auth

echo

echo "[INFO] 建议把这些 GitHub Variables 也配置好:"
echo "  AZURE_RESOURCE_GROUP=${RESOURCE_GROUP}"
echo "  AZURE_CONTAINER_APP_NAME=churn-prediction-api"
echo "  AZURE_CONTAINER_APP_ENVIRONMENT=${CONTAINERAPP_ENV}"
echo "  AZURE_ACR_NAME=${ACR_NAME}"
