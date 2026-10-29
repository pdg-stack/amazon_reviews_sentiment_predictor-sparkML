# MLflow gets its own dedicated database (not just a schema in a shared one) --
# stronger isolation, and avoids needing a network path to run manual schema/role
# DDL against a private RDS instance before any VPC-resident compute (Phase 2's
# EC2) exists to reach it. MLflow's SQLAlchemy backend creates its own tables in
# this database's default `public` schema automatically on first connect.
#
# The `app` schema/database (predictions, drift_checks -- Phase 2) gets created
# later from the EC2 instance, which will actually be able to reach RDS.

resource "random_password" "rds_master" {
  length  = 32
  special = false # avoid characters that need extra escaping in connection strings/.env
}

resource "aws_db_instance" "main" {
  identifier     = "${var.project_prefix}-db"
  engine         = "postgres"
  engine_version = "16"
  instance_class = var.rds_instance_class

  allocated_storage = var.rds_allocated_storage_gb
  storage_type      = "gp3"

  db_name  = "mlflow"
  username = "mlflow_admin"
  password = random_password.rds_master.result

  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.rds.id]
  publicly_accessible    = false
  multi_az               = false # single-AZ -- fine for a POC, halves the cost of multi-AZ

  backup_retention_period = 1
  skip_final_snapshot     = true # POC project, not worth the extra resource/cost tracking a final snapshot needs

  deletion_protection = false
}
