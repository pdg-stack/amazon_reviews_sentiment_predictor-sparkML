# Phase 4: GitHub Actions authenticates via OIDC federation, not long-lived
# AWS keys stored as GitHub secrets. The trust policy is scoped to this
# specific repo and branch -- a workflow run from a fork or a different
# branch can't assume this role.

data "tls_certificate" "github_actions" {
  url = "https://token.actions.githubusercontent.com/.well-known/openid-configuration"
}

resource "aws_iam_openid_connect_provider" "github_actions" {
  url             = "https://token.actions.githubusercontent.com"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = [data.tls_certificate.github_actions.certificates[0].sha1_fingerprint]
}

data "aws_iam_policy_document" "github_actions_trust" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github_actions.arn]
    }
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:${var.github_repo}:ref:refs/heads/main"]
    }
  }
}

resource "aws_iam_role" "github_actions_deploy" {
  name               = "${var.project_prefix}-github-actions-deploy"
  assume_role_policy = data.aws_iam_policy_document.github_actions_trust.json
}

# ECR push: GetAuthorizationToken only ever works as resource "*" (same as
# ec2.tf's pull-side policy); the per-repo actions are scoped to just the
# two repos this workflow builds.
data "aws_iam_policy_document" "github_actions_deploy" {
  statement {
    sid       = "EcrAuth"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }
  statement {
    sid = "EcrPush"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:GetDownloadUrlForLayer",
      "ecr:BatchGetImage",
      "ecr:PutImage",
      "ecr:InitiateLayerUpload",
      "ecr:UploadLayerPart",
      "ecr:CompleteLayerUpload",
    ]
    resources = [aws_ecr_repository.api.arn, aws_ecr_repository.log_viewer.arn]
  }
  statement {
    sid       = "DeployViaSsm"
    actions   = ["ssm:SendCommand"]
    resources = [aws_instance.serving.arn, "arn:aws:ssm:${data.aws_region.current.id}::document/AWS-RunShellScript"]
  }
  statement {
    sid       = "ReadSsmCommandResult"
    actions   = ["ssm:GetCommandInvocation"]
    resources = ["*"] # command-invocation ARNs aren't known until SendCommand returns -- no narrower resource to scope to
  }
}

resource "aws_iam_role_policy" "github_actions_deploy" {
  name   = "${var.project_prefix}-github-actions-deploy"
  role   = aws_iam_role.github_actions_deploy.id
  policy = data.aws_iam_policy_document.github_actions_deploy.json
}

data "aws_region" "current" {}
