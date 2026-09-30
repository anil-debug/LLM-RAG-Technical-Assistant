# AWS design

This is a design. No AWS account was used. No VPC, cluster, database, bucket, load balancer, or secret was created. Costs below are resource categories, not a bill.

## Layout

```text
Route 53
  -> ALB (ingress)
       -> EKS node group (API, UI)
       -> optional GPU node group (Transformers or vLLM)
  -> RDS PostgreSQL 16 + pgvector (private subnets)
  -> S3 (original uploads)
  -> Secrets Manager (DATABASE_URL, LLM API key)
  -> CloudWatch (container logs, ALB metrics, RDS metrics)
```

The Helm chart is what EKS would run. `vectorBackend` becomes `postgres`. `DATABASE_URL` comes from Secrets Manager, synced into a Kubernetes secret by External Secrets or a CSI driver. The chart's placeholder secret is not the production secret.

## Networking

- A VPC with public subnets for the ALB and private subnets for nodes and RDS.
- Nodes have no public IPs. They reach the ALB through the AWS load balancer controller.
- RDS is not publicly accessible. The node security group is the only inbound source on 5432.
- The API's egress to Ollama is not an AWS pattern. On AWS, generation is either vLLM on the GPU node group or a private endpoint you operate. The provider stays `OpenAICompatibleProvider` with `LLM_BASE_URL` pointed at that service.
- S3 is reached through a gateway endpoint so uploads do not cross a NAT for storage.

## Security

- IAM roles for service accounts. The API role can `s3:PutObject` and `s3:GetObject` on one bucket prefix, and `secretsmanager:GetSecretValue` on the two secret ARNs. It cannot administer RDS or read other buckets.
- The node role is the normal EKS node role. It is not given the application secrets.
- RDS uses a managed master password in Secrets Manager. Application credentials are a separate user that can `SELECT`, `INSERT`, `UPDATE`, and `DELETE` on the RAG tables and can `CREATE EXTENSION` only during bootstrap, not as a standing grant if you prefer a migration job.
- Security groups are least privilege: ALB ingress 443 from the internet or a corporate CIDR, node ingress only from the ALB security group, RDS ingress only from the node security group.
- Kubernetes secrets are mounted as environment variables. They are not committed. The chart value `secret.databaseUrl` must be overridden at install time.
- Uploads are size-limited by the application and should also be limited at the ALB. Filenames are basenames. Retrieved text stays untrusted evidence.

## Storage

- RDS stores chunks, embeddings (`vector(768)`), and conversations. Storage grows with chunk count, not with the UI.
- HNSW indexes are per `model_name`. Adding E5 does not require a new column while the dimension stays 768. A different dimension needs a new table. This build does not create one.
- S3 stores the original file. The database stores the checksum and the S3 key. The in-cluster `emptyDir` is only a scratch space.
- Backups are RDS automated backups. Vectors are in the same database, so they are in the same backup.

## Scaling

- The API deployment scales on CPU when retrieval is in-process BM25 plus pgvector. BM25 is rebuilt from all chunks on each query (`hybrid_retrieve`). That is correct for this corpus and becomes the first bottleneck at tens of thousands of chunks. The next step is to keep BM25 postings in Postgres or OpenSearch, not to scale the UI.
- pgvector HNSW serves the ANN query. Exact search remains available in the memory store for tests.
- The UI is stateless and scales separately. It only calls the API.
- A GPU node group scales from zero when `TransformersLocalProvider` or vLLM is deployed. The API node group stays on CPU. Do not place the GPU limit on the CPU API deployment unless that pod is the one loading weights. The chart's `gpu.enabled` flag is that switch, and it should be a separate deployment in a real cluster so a CPU rollout does not request a GPU.

## Observability

- Application logs already include `request_id`, `query_id`, `document_id`, `embedding_ms`, `bm25_ms`, `reranker_ms`, `llm_ms`, `model_name`, and `top_k`. Ship stdout to CloudWatch Logs.
- ALB target 5xx and target response time show whether `/ready` is failing.
- RDS CPU, free storage, and connections show whether HNSW or the BM25 table scan is the database problem.
- GPU utilization and memory are NVIDIA metrics on the GPU node group only. Do not report them for the Ollama-less CPU API.

## Resource categories

| Piece | Category |
| --- | --- |
| EKS control plane | One cluster |
| CPU node group | Small general-purpose nodes for API and UI |
| GPU node group | Optional, one GPU instance family, scale to zero |
| RDS | PostgreSQL, multi-AZ for a real deployment, single-AZ for a lab |
| S3 | One bucket, versioning on for uploads |
| ALB | One, via the AWS Load Balancer Controller |
| Secrets Manager | Two secrets |
| NAT or VPC endpoints | Endpoints preferred for S3 and the AWS APIs |

No dollar figure is stated because no pricing call was made.

## What was not done

No `terraform apply`, no `eksctl`, no `aws` CLI against an account, no image in ECR, no `helm install`. The commands to do that later, after a cluster exists:

```bash
aws ecr create-repository --repository-name technical-rag-assistant
docker build -t technical-rag-assistant:0.1.0 .
docker tag technical-rag-assistant:0.1.0 <account>.dkr.ecr.<region>.amazonaws.com/technical-rag-assistant:0.1.0
docker push <account>.dkr.ecr.<region>.amazonaws.com/technical-rag-assistant:0.1.0
helm upgrade --install rag deployment/helm/technical-rag-assistant \
  --set image.repository=<account>.dkr.ecr.<region>.amazonaws.com/technical-rag-assistant \
  --set image.tag=0.1.0 \
  --set config.vectorBackend=postgres \
  --set secret.databaseUrl=<from Secrets Manager> \
  --set ingress.enabled=true \
  --set ingress.className=alb \
  --set ingress.host=rag.example.com
```

Those commands were not run.
