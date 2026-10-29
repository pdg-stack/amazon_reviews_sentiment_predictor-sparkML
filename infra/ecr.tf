# Training image (Phase 1). The always-on stack's images (api, log-viewer --
# Phase 4) get their own repos below: GitHub Actions builds and pushes them
# on merge to main, and the EC2 instance only ever `docker compose pull`s,
# never builds locally (see docker-compose.cloud.yml, cicd.tf).

resource "aws_ecr_repository" "training" {
  name                 = "${var.project_prefix}-training"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

data "aws_iam_policy_document" "training_ecr_pull" {
  statement {
    sid     = "AllowEMRServerlessPull"
    actions = ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer", "ecr:BatchCheckLayerAvailability", "ecr:DescribeImages"]
    principals {
      type        = "Service"
      identifiers = ["emr-serverless.amazonaws.com"]
    }
  }
}

resource "aws_ecr_repository_policy" "training" {
  repository = aws_ecr_repository.training.name
  policy     = data.aws_iam_policy_document.training_ecr_pull.json
}

resource "aws_ecr_lifecycle_policy" "training" {
  repository = aws_ecr_repository.training.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep only the 5 most recent images -- this is a POC training image, not worth unbounded history."
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 5
      }
      action = { type = "expire" }
    }]
  })
}

# api (docker/Dockerfile) and log-viewer (docker/log-viewer.Dockerfile) --
# pulled by the EC2 instance's own IAM role (ec2.tf already grants it
# ecr:GetDownloadUrlForLayer/BatchGetImage/BatchCheckLayerAvailability on "*"),
# pushed by the GitHub Actions role (cicd.tf). No cross-account/service
# principal policy needed here, unlike training's EMR Serverless repo --
# both sides are already-trusted principals within this same account.
resource "aws_ecr_repository" "api" {
  name                 = "${var.project_prefix}-api"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "api" {
  repository = aws_ecr_repository.api.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep only the 10 most recent images."
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 10
      }
      action = { type = "expire" }
    }]
  })
}

resource "aws_ecr_repository" "log_viewer" {
  name                 = "${var.project_prefix}-log-viewer"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "log_viewer" {
  repository = aws_ecr_repository.log_viewer.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep only the 10 most recent images."
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 10
      }
      action = { type = "expire" }
    }]
  })
}
