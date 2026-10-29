variable "aws_region" {
  description = "AWS region for all resources."
  type        = string
  default     = "us-east-1"
}

variable "aws_account_id" {
  description = "AWS account ID (used to build a globally-unique S3 bucket name)."
  type        = string
  default     = "183088117536"
}

variable "project_prefix" {
  description = "Short prefix for resource names (Amazon Reviews Sentiment Predictor)."
  type        = string
  default     = "arsp"
}

variable "rds_instance_class" {
  description = "RDS instance class -- smallest Graviton burstable class, fine for MLflow's light write load."
  type        = string
  default     = "db.t4g.micro"
}

variable "rds_allocated_storage_gb" {
  description = "RDS storage in GB -- 20GB is the minimum and far more than MLflow's metadata needs."
  type        = number
  default     = 20
}

variable "emr_release_label" {
  description = "EMR Serverless release label. Closest match to the project's pyspark==3.5.3 pin -- confirm/upgrade once real jobs are submitted."
  type        = string
  default     = "emr-7.2.0" # ships Spark 3.5.1
}

variable "ec2_instance_type" {
  description = <<-EOT
    Serving stack instance size. c5.xlarge (4 vCPU, 8GB, compute-optimized,
    non-burstable) -- not t3.medium: api's /predict path runs through an
    embedded Spark local[*] session (JVM task scheduling + Py4J
    serialization per call, not a lightweight inference path), so
    throughput under the load generator's ~1,000-concurrent-peak test
    scales with vCPU count, and a sustained ~17 req/s for the full 20-minute
    test would risk draining a t3's CPU credit balance mid-run and
    contaminating the result -- c5 has no such cliff.
  EOT
  type        = string
  default     = "c5.xlarge"
}

variable "github_repo" {
  description = "owner/repo whose main branch is trusted to assume the GitHub Actions deploy role (cicd.tf)."
  type        = string
  default     = "pdg-stack/amazon_reviews_sentiment_predictor-sparkML"
}
