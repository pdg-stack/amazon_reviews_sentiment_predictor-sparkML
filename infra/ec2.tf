# Phase 2: the always-on serving stack (api, mlflow-ui, log-viewer, nginx;
# Phase 3 adds grafana) -- one small EC2 instance, not ECS/Fargate, per the
# plan. No SSH: managed entirely via SSM (Session Manager / Run Command),
# same no-open-port pattern as Phase 4's eventual deploys.

data "aws_ami" "al2023" {
  most_recent = true
  owners      = ["amazon"]
  filter {
    # "al2023-ami-2023.*" (not the broader "al2023-ami-*", which also
    # matches "al2023-ami-minimal-2023...") -- the minimal variant doesn't
    # ship SSM Agent, discovered when an instance built from it never
    # registered with SSM no matter how long it ran.
    name   = "name"
    values = ["al2023-ami-2023.*-x86_64"]
  }
}

# Only CloudFront's own origin-facing IP ranges can reach nginx -- the
# instance is never meant to be hit directly. WAF (edge.tf) sits in front of
# CloudFront, not this.
data "aws_ec2_managed_prefix_list" "cloudfront" {
  name = "com.amazonaws.global.cloudfront.origin-facing"
}

resource "aws_security_group" "ec2" {
  name        = "${var.project_prefix}-ec2-sg"
  description = "Serving stack instance -- inbound 80 from CloudFront origin-facing ranges only, no SSH (SSM-managed)."
  vpc_id      = data.aws_vpc.default.id

  ingress {
    from_port       = 80
    to_port         = 80
    protocol        = "tcp"
    prefix_list_ids = [data.aws_ec2_managed_prefix_list.cloudfront.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group_rule" "rds_from_ec2" {
  type                     = "ingress"
  from_port                = 5432
  to_port                  = 5432
  protocol                 = "tcp"
  security_group_id        = aws_security_group.rds.id
  source_security_group_id = aws_security_group.ec2.id
  description              = "api writing to app.predictions/app.drift_checks; mlflow-ui reading the mlflow database."
}

data "aws_iam_policy_document" "ec2_trust" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "ec2" {
  name               = "${var.project_prefix}-ec2-role"
  assume_role_policy = data.aws_iam_policy_document.ec2_trust.json
}

resource "aws_iam_role_policy_attachment" "ec2_ssm" {
  role       = aws_iam_role.ec2.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

data "aws_iam_policy_document" "ec2_access" {
  statement {
    sid       = "EcrPull"
    actions   = ["ecr:GetDownloadUrlForLayer", "ecr:BatchGetImage", "ecr:BatchCheckLayerAvailability", "ecr:GetAuthorizationToken"]
    resources = ["*"] # GetAuthorizationToken is only ever allowed as resource "*"
  }
  statement {
    sid       = "S3DataAndArtifacts"
    actions   = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
    resources = [aws_s3_bucket.main.arn, "${aws_s3_bucket.main.arn}/*"]
  }
}

resource "aws_iam_role_policy" "ec2_access" {
  name   = "${var.project_prefix}-ec2-access"
  role   = aws_iam_role.ec2.id
  policy = data.aws_iam_policy_document.ec2_access.json
}

resource "aws_iam_instance_profile" "ec2" {
  name = "${var.project_prefix}-ec2-profile"
  role = aws_iam_role.ec2.name
}

resource "aws_instance" "serving" {
  ami                    = data.aws_ami.al2023.id
  instance_type          = var.ec2_instance_type
  subnet_id              = data.aws_subnets.default.ids[0]
  vpc_security_group_ids = [aws_security_group.ec2.id]
  iam_instance_profile   = aws_iam_instance_profile.ec2.name

  # Docker + compose plugin only -- the actual stack (docker-compose.cloud.yml)
  # is deployed afterward via SSM, not baked into the AMI/user-data.
  user_data = <<-EOF
    #!/bin/bash
    dnf install -y docker
    systemctl enable --now docker
    usermod -aG docker ec2-user
    mkdir -p /usr/local/lib/docker/cli-plugins
    curl -SL https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64 -o /usr/local/lib/docker/cli-plugins/docker-compose
    chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
  EOF

  tags = {
    Name = "${var.project_prefix}-serving"
  }
}

resource "aws_eip" "serving" {
  instance = aws_instance.serving.id
  domain   = "vpc"
}
