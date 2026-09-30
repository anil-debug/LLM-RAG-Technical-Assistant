# Kubernetes and Helm

Two layouts are in the repository:

- Raw manifests in `deployment/kubernetes/`
- Helm chart in `deployment/helm/technical-rag-assistant/`

The chart is the one to install. The raw manifests are the same shape with fixed names, for reading without Helm.

## What the chart creates

- ServiceAccount
- ConfigMap for non-secret settings (embedding model, LLM base URL, reranker flag)
- Secret for `DATABASE_URL` and `LLM_API_KEY` (placeholders)
- API Deployment and Service
- UI Deployment and Service, unless `ui.enabled=false`
- Ingress only when `ingress.enabled=true`

The API container probes `/health` (liveness) and `/ready` (readiness). Readiness pings the vector store. With the default `vectorBackend=memory`, readiness does not need RDS. Switch to `postgres` and set `secret.databaseUrl` before treating the chart as a deployment.

Resource requests and limits are set on both containers. `allowPrivilegeEscalation` is false. Uploads use an `emptyDir` at `/app/data`.

GPU scheduling is off by default. `gpu.enabled=true` adds an `nvidia.com/gpu` limit and optional node selector and tolerations. That block was rendered with `helm template`. No GPU pod was scheduled.

## Render

```bash
helm template technical-rag-assistant deployment/helm/technical-rag-assistant
helm template technical-rag-assistant deployment/helm/technical-rag-assistant \
  --set ingress.enabled=true \
  --set gpu.enabled=true
```

Both commands succeeded in this environment. `helm install` was not run. No cluster was contacted. The image `technical-rag-assistant:0.1.0` was not pushed to a registry.

## Ingress hostnames

When ingress is enabled, the UI is `rag.example.com` and the API is `api.rag.example.com`. The API is not mounted under an `/api` prefix, because the FastAPI routes are `/documents`, `/search`, `/chat`, `/health`, and `/ready`.

## Apply the raw manifests

```bash
kubectl apply -f deployment/kubernetes/configmap.yaml
kubectl apply -f deployment/kubernetes/secret.yaml
kubectl apply -f deployment/kubernetes/api-deployment.yaml
kubectl apply -f deployment/kubernetes/api-service.yaml
kubectl apply -f deployment/kubernetes/ui-deployment.yaml
kubectl apply -f deployment/kubernetes/ui-service.yaml
```

Replace the secret before applying it. The ingress file is optional and points at `alb` as the class.
