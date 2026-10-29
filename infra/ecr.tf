# Training image only (Phase 1). Phase 4 adds separate repos/tags for the
# always-on stack's images (api, log-viewer) once that gets built.

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
