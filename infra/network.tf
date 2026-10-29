# Uses the account's default VPC rather than building a custom one -- simplest,
# cheapest option for a single small RDS instance + EMR Serverless. RDS is not
# publicly accessible and gets no public IP, so it's not internet-reachable even
# though these subnets have a route to an internet gateway (that route only
# matters for resources that actually have a public IP).

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# EMR Serverless doesn't support every AZ in every account (us-east-1e came
# back "currently not supported" here) -- restrict its subnets to the AZs
# that do work, discovered by trial.
data "aws_subnets" "emr_compatible" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
  filter {
    name   = "availability-zone"
    values = ["us-east-1a", "us-east-1b", "us-east-1c", "us-east-1d", "us-east-1f"]
  }
}

resource "aws_db_subnet_group" "main" {
  name       = "${var.project_prefix}-rds-subnets"
  subnet_ids = data.aws_subnets.default.ids
}

resource "aws_security_group" "rds" {
  name        = "${var.project_prefix}-rds-sg"
  description = "Postgres access for MLflows backend store -- inbound only from the EMR Serverless SG (and later, Phase 2s EC2 SG)."
  vpc_id      = data.aws_vpc.default.id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "emr_serverless" {
  name        = "${var.project_prefix}-emr-serverless-sg"
  description = "EMR Serverless application -- egress to RDS + S3, no inbound needed."
  vpc_id      = data.aws_vpc.default.id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group_rule" "rds_from_emr" {
  type                     = "ingress"
  from_port                = 5432
  to_port                  = 5432
  protocol                 = "tcp"
  security_group_id        = aws_security_group.rds.id
  source_security_group_id = aws_security_group.emr_serverless.id
  description               = "MLflow (running as an EMR Serverless job) writing to its Postgres backend store."
}
