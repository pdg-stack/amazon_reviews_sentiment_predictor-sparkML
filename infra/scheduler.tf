# Phase 3: automates what was, until now, a manually-triggered drift check.
# A scheduled EventBridge rule calls GET /admin/drift-report every 2 minutes
# via an API destination (the standard way EventBridge calls an arbitrary
# HTTPS endpoint with a custom auth header) -- the CloudFront domain is read
# directly from aws_cloudfront_distribution.main rather than hardcoded,
# since that domain has already changed twice this project (distribution
# recreated), and a stale reference here would silently stop drift checks.
#
# NOTE: API destinations are a Rules-targeting feature (aws_cloudwatch_event_rule
# + aws_cloudwatch_event_target below), NOT something EventBridge *Scheduler*
# (aws_scheduler_schedule) can invoke directly -- confirmed by testing:
# aws_scheduler_schedule rejected the api-destination ARN outright
# ("Provided Arn is not in correct format"). These are two separate
# EventBridge products that don't interoperate the way their shared "EventBridge"
# name suggests.

variable "drift_report_api_key" {
  description = "One API key from .env's API_KEYS/API_KEY pool, used only for the EventBridge-triggered drift-report call. Pass via TF_VAR_drift_report_api_key -- never commit a default."
  type        = string
  sensitive   = true
}

resource "aws_cloudwatch_event_connection" "drift_report" {
  name               = "${var.project_prefix}-drift-report"
  authorization_type = "API_KEY"

  auth_parameters {
    api_key {
      key   = "X-API-Key"
      value = var.drift_report_api_key
    }
  }
}

resource "aws_cloudwatch_event_api_destination" "drift_report" {
  name                = "${var.project_prefix}-drift-report"
  invocation_endpoint = "https://${aws_cloudfront_distribution.main.domain_name}/admin/drift-report"
  http_method         = "GET"
  connection_arn      = aws_cloudwatch_event_connection.drift_report.arn
  invocation_rate_limit_per_second = 1
}

data "aws_iam_policy_document" "events_trust" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["events.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "drift_report_scheduler" {
  name               = "${var.project_prefix}-drift-scheduler"
  assume_role_policy = data.aws_iam_policy_document.events_trust.json
}

data "aws_iam_policy_document" "drift_report_scheduler_invoke" {
  statement {
    actions   = ["events:InvokeApiDestination"]
    resources = [aws_cloudwatch_event_api_destination.drift_report.arn]
  }
}

resource "aws_iam_role_policy" "drift_report_scheduler_invoke" {
  name   = "${var.project_prefix}-drift-scheduler-invoke"
  role   = aws_iam_role.drift_report_scheduler.id
  policy = data.aws_iam_policy_document.drift_report_scheduler_invoke.json
}

resource "aws_cloudwatch_event_rule" "drift_report" {
  name                = "${var.project_prefix}-drift-report"
  schedule_expression = "rate(2 minutes)"
}

resource "aws_cloudwatch_event_target" "drift_report" {
  rule     = aws_cloudwatch_event_rule.drift_report.name
  arn      = aws_cloudwatch_event_api_destination.drift_report.arn
  role_arn = aws_iam_role.drift_report_scheduler.arn
}
