# Kubernetes

Kubernetes is a separate path from Docker Compose. You do not start Compose in order to install the chart. Both paths run the same image, `technical-rag-assistant:0.1.0`, and the same environment variable names (`VECTOR_BACKEND`, `DATABASE_URL`, `LLM_BASE_URL`, and the rest of `core/settings.py`).

```text
Docker Compose     docker-compose.yml on one machine
Kubernetes         Helm chart or raw manifests, including Postgres
AWS EKS            same Helm chart, values-eks.yaml, RDS instead of the StatefulSet
```

## What gets installed

Default `values.yaml` and `values-minikube.yaml`:

- ServiceAccount
- ConfigMap for non-secret settings
- Secret for `DATABASE_URL` and `LLM_API_KEY`
- PostgreSQL 16 + pgvector StatefulSet and Service
- API Deployment and Service, liveness `/health`, readiness `/ready`
- UI Deployment and Service, TCP probes on 8501
- Resource requests and limits on each container
- Ingress only when `ingress.enabled=true`

`values-eks.yaml` turns the StatefulSet off. `secret.databaseUrl` must be the RDS URL. The chart fails the render if Postgres is disabled and that URL is empty.

The local database password in the chart and in `deployment/kubernetes/secret.yaml` is `rag`. That matches Compose and is not an AWS secret. Replace it before any shared cluster.

GPU scheduling stays off unless `gpu.enabled=true`. That flag adds `nvidia.com/gpu` on the API container. It does not create a node group.

## Build the image once

```bash
docker build -t technical-rag-assistant:0.1.0 .
```

Compose produces the same tag when you build `docker-compose.yml`. A registry push is required only for a node that cannot see your local Docker images. EKS needs the image in ECR (or another registry the nodes can pull).

## Minikube

```bash
minikube start --driver=docker --cpus=4 --memory=8192
minikube image load technical-rag-assistant:0.1.0
minikube image load pgvector/pgvector:pg16
helm upgrade --install technical-rag-assistant \
  deployment/helm/technical-rag-assistant \
  -f deployment/helm/technical-rag-assistant/values-minikube.yaml
kubectl rollout status statefulset/technical-rag-assistant-postgres
kubectl rollout status deployment/technical-rag-assistant-api
kubectl rollout status deployment/technical-rag-assistant-ui
kubectl port-forward svc/technical-rag-assistant-api 8000:8000
kubectl port-forward svc/technical-rag-assistant-ui 8501:8501
```

`values-minikube.yaml` sets `image.pullPolicy=Never` so a missing load fails instead of trying a registry. Open a second terminal for the UI port-forward. Then:

```bash
curl -sS http://127.0.0.1:8000/health
curl -sS http://127.0.0.1:8000/ready
curl -sS http://127.0.0.1:8000/documents
```

`/ready` must show `"vector_backend":"postgres"`. `GET /documents` creates the pgvector schema.

The same install without Helm:

```bash
kubectl apply -k deployment/kubernetes
kubectl rollout status statefulset/technical-rag-postgres
kubectl rollout status deployment/technical-rag-api
kubectl rollout status deployment/technical-rag-ui
```

Raw manifests use the image name `technical-rag-assistant:0.1.0` and `imagePullPolicy: IfNotPresent`. Load the image into the cluster first. The ingress file is not part of the kustomization. Apply it only when a controller exists:

```bash
kubectl apply -f deployment/kubernetes/ingress.yaml
```

That ingress expects class `alb` and the hosts `rag.example.com` and `api.rag.example.com`. On Minikube, change the class to the installed controller or use port-forward instead.

## Upgrade

```bash
helm upgrade technical-rag-assistant \
  deployment/helm/technical-rag-assistant \
  -f deployment/helm/technical-rag-assistant/values-minikube.yaml
kubectl rollout status deployment/technical-rag-assistant-api
helm history technical-rag-assistant
```

Raw manifests:

```bash
kubectl apply -k deployment/kubernetes
kubectl rollout status deployment/technical-rag-api
```

## Rollback

Helm keeps revisions:

```bash
helm history technical-rag-assistant
helm rollback technical-rag-assistant 1
kubectl rollout status deployment/technical-rag-assistant-api
```

Replace `1` with the revision you want. Raw manifests have no revision history. Re-apply the previous YAML from git:

```bash
git checkout <previous-commit> -- deployment/kubernetes
kubectl apply -k deployment/kubernetes
```

## Cleanup

```bash
helm uninstall technical-rag-assistant
kubectl delete pvc -l app.kubernetes.io/name=technical-rag-assistant
```

`helm uninstall` deletes the StatefulSet and does not always delete its PVC. The `kubectl delete pvc` removes the database disk. Skip that line if you intend to reinstall onto the same data.

Raw manifests:

```bash
kubectl delete -k deployment/kubernetes
kubectl delete pvc -l app.kubernetes.io/name=technical-rag-assistant
```

Stop Minikube only when you want the node gone. That is independent of this chart:

```bash
minikube stop
minikube delete
```

## AWS EKS

Do not apply `deployment/kubernetes/` on EKS if RDS is the database. That directory starts an in-cluster StatefulSet. Use the chart:

```bash
helm upgrade --install technical-rag-assistant \
  deployment/helm/technical-rag-assistant \
  -f deployment/helm/technical-rag-assistant/values-eks.yaml \
  --set image.repository=<account>.dkr.ecr.<region>.amazonaws.com/technical-rag-assistant \
  --set image.tag=0.1.0 \
  --set secret.databaseUrl='postgresql://USER:PASSWORD@RDS_HOST:5432/rag' \
  --set secret.llmApiKey='REPLACE_ME' \
  --set ingress.host=rag.example.com
```

Pass the real URL from Secrets Manager at install time. Do not write it into `values-eks.yaml` and commit it. The placeholder `REPLACE_ME` host in that file will not connect.

Upgrade, history, and rollback are the same `helm upgrade`, `helm history`, and `helm rollback` commands. Cleanup is `helm uninstall`. It does not delete RDS.

No EKS cluster was created by this repository. Networking, IAM, and the RDS parameter group are in `docs/deployment/aws.md`.

## Render without a cluster

```bash
helm template technical-rag-assistant deployment/helm/technical-rag-assistant
helm template technical-rag-assistant deployment/helm/technical-rag-assistant \
  -f deployment/helm/technical-rag-assistant/values-minikube.yaml
helm template technical-rag-assistant deployment/helm/technical-rag-assistant \
  -f deployment/helm/technical-rag-assistant/values-eks.yaml
helm template technical-rag-assistant deployment/helm/technical-rag-assistant \
  --set ingress.enabled=true --set gpu.enabled=true
kubectl kustomize deployment/kubernetes
```

`make helm-template` runs the four Helm renders. A successful render is not a running cluster.

## Probes and resources

| Container | Liveness | Readiness | Requests | Limits |
| --- | --- | --- | --- | --- |
| API | `GET /health` | `GET /ready` | 250m CPU, 1Gi | 2 CPU, 4Gi |
| UI | TCP 8501 | TCP 8501 | 100m CPU, 256Mi | 500m CPU, 1Gi |
| Postgres | `pg_isready` | `pg_isready` | 100m CPU, 256Mi | 1 CPU, 1Gi |

`/health` only checks the process. `/ready` pings the database, so the API pod stays out of service until Postgres answers. The API does not exit when the database is down, and it does not need the embedding weights to become ready.

When ingress is enabled, the UI host is `rag.example.com` and the API host is `api.rag.example.com`. The API is not under an `/api` prefix. FastAPI serves `/documents`, `/search`, `/chat`, `/health`, and `/ready` at the root.

## Troubleshooting

| Symptom | What to do |
| --- | --- |
| `ImagePullBackOff` or `ErrImageNeverPull` | The node does not have `technical-rag-assistant:0.1.0`. `minikube image load` it, or push it and set `image.pullPolicy` to `IfNotPresent`. |
| API `Ready` stays false | `kubectl logs deploy/technical-rag-assistant-api`. Then `kubectl get pods` for the Postgres pod. `/ready` is 503 until Postgres accepts connections. |
| `secret.databaseUrl is required` | You rendered with `postgres.enabled=false` and an empty URL. Set the RDS URL. |
| PVC pending | The cluster has no default StorageClass. Set `postgres.storageClass`. |
| Chat returns 503 | `LLM_BASE_URL` does not reach a model server. Readiness does not check the LLM. |
| Search returns 503 `model_unavailable` | The pod cannot download `BAAI/bge-base-en-v1.5`. |
| Ingress has no address | No controller for `ingress.className`, or DNS does not point at it. Use port-forward to test the pods. |

## What was executed here

On 2026-09-30 a stale Minikube profile had no container and `minikube start` failed with `K8S_APISERVER_MISSING`. That profile was deleted. A new cluster came up with `minikube start --driver=docker --cpus=4 --memory=8192` (Kubernetes v1.35.0). The node was Ready.

`minikube image load` copied `technical-rag-assistant:0.1.0` and `pgvector/pgvector:pg16` into the node. `helm upgrade --install` with `values-minikube.yaml` reached revision 1. All three pods were `1/1 Running`: API, UI, and `technical-rag-assistant-postgres-0`.

| Check | Result |
| --- | --- |
| `GET /health` inside the API pod | `{"status":"ok"}` |
| `GET /ready` from the UI pod through the API Service DNS | `vector_backend` `postgres` |
| `GET /documents` | `[]` |
| `pg_extension` | `vector` |
| Streamlit inside the UI pod | HTTP 200. No browser session, and host ports 8000 and 8501 were already used by Compose, so this check did not use port-forward |
| `helm upgrade` | Revision 2, status deployed |
| `helm rollback technical-rag-assistant 1` | Revision 3, description `Rollback to 1`. `/ready` still returned postgres afterward |

Chat was not called inside the cluster. EKS was not created. `values-eks.yaml` was only rendered.

The release was left installed. Remove it with `helm uninstall technical-rag-assistant` and `kubectl delete pvc -l app.kubernetes.io/name=technical-rag-assistant`. `minikube stop` stops the node without deleting the release.
