# The `app` database (predictions, drift_checks -- Phase 2/3) lives on the
# SAME RDS instance as MLflow's `mlflow` database (not a second instance),
# but as its own database with its own low-privilege user -- avoids
# write-load coupling with MLflow's tables and keeps a clean blast radius.
# The actual CREATE DATABASE/CREATE ROLE/CREATE TABLE DDL runs via SSM from
# the EC2 instance (ec2.tf) -- this instance is the first thing in the VPC
# that can actually reach RDS to run it. See infra/README or the deploy
# notes for the exact command.

resource "random_password" "app_db" {
  length  = 32
  special = false
}

output "app_db_password" {
  value     = random_password.app_db.result
  sensitive = true
}
