output "s3_bucket_name" {
  value = aws_s3_bucket.main.bucket
}

output "rds_endpoint" {
  value = aws_db_instance.main.address
}

output "rds_port" {
  value = aws_db_instance.main.port
}

output "rds_database" {
  value = aws_db_instance.main.db_name
}

output "rds_username" {
  value = aws_db_instance.main.username
}

output "rds_password" {
  value     = random_password.rds_master.result
  sensitive = true
}

output "emr_application_id" {
  value = aws_emrserverless_application.spark.id
}

output "emr_execution_role_arn" {
  value = aws_iam_role.emr_serverless_execution.arn
}

output "ecr_repository_url" {
  value = aws_ecr_repository.training.repository_url
}

output "ec2_instance_id" {
  value = aws_instance.serving.id
}

output "ec2_public_ip" {
  value = aws_eip.serving.public_ip
}

output "cloudfront_domain" {
  value = aws_cloudfront_distribution.main.domain_name
}

output "ecr_api_repository_url" {
  value = aws_ecr_repository.api.repository_url
}

output "ecr_log_viewer_repository_url" {
  value = aws_ecr_repository.log_viewer.repository_url
}

output "github_actions_deploy_role_arn" {
  description = "Paste into .github/workflows/deploy.yml's role-to-assume."
  value       = aws_iam_role.github_actions_deploy.arn
}
