data "aws_iam_policy_document" "emr_serverless_trust" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["emr-serverless.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "emr_serverless_execution" {
  name               = "${var.project_prefix}-emr-serverless-execution"
  assume_role_policy = data.aws_iam_policy_document.emr_serverless_trust.json
}

data "aws_iam_policy_document" "emr_serverless_access" {
  statement {
    sid = "S3DataAccess"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
      "s3:ListBucket",
    ]
    resources = [
      aws_s3_bucket.main.arn,
      "${aws_s3_bucket.main.arn}/*",
    ]
  }

  statement {
    sid = "CloudWatchLogs"
    actions = [
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:PutLogEvents",
      "logs:DescribeLogGroups",
      "logs:DescribeLogStreams",
    ]
    resources = ["arn:aws:logs:${var.aws_region}:${var.aws_account_id}:*"]
  }
}

resource "aws_iam_role_policy" "emr_serverless_access" {
  name   = "${var.project_prefix}-emr-serverless-access"
  role   = aws_iam_role.emr_serverless_execution.id
  policy = data.aws_iam_policy_document.emr_serverless_access.json
}
