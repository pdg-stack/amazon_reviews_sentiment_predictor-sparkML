# One CloudFront distribution in front of everything (api, mlflow-ui,
# log-viewer, grafana -- nginx does the path routing), with WAF's rate-based
# rule attached. Default *.cloudfront.net domain -- no custom domain, so no
# ACM cert to manage; CloudFront's own domain gets a trusted cert
# automatically. See docs/DEPLOYMENT.md for the full reasoning.

resource "aws_wafv2_web_acl" "main" {
  name  = "${var.project_prefix}-waf"
  scope = "CLOUDFRONT" # must be created in us-east-1, which this whole project already targets

  default_action {
    allow {}
  }

  rule {
    name     = "rate-limit-per-ip"
    priority = 1
    action {
      block {}
    }
    statement {
      rate_based_statement {
        # Raised from 2000 for the Phase 2 Modal load test: ~1000 concurrent
        # sessions at peak, each sending ~1 request/minute after their
        # initial burst, works out to ~1000-1250 req/min sustained -- up to
        # ~6000 in a rolling 5-min window -- and Modal's calls come from a
        # small/shared set of egress IPs, not 1000 distinct ones, so this
        # whole load concentrates onto very few source IPs (flagged as a
        # risk, since its calls concentrate onto very few source IPs).
        # Lower this back down once the load test is done.
        limit              = 20000 # requests per 5-minute window per IP
        aggregate_key_type = "IP"
      }
    }
    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "${var.project_prefix}-rate-limit"
      sampled_requests_enabled   = true
    }
  }

  visibility_config {
    cloudwatch_metrics_enabled = true
    metric_name                = "${var.project_prefix}-waf"
    sampled_requests_enabled   = true
  }
}

resource "aws_cloudfront_distribution" "main" {
  enabled         = true
  web_acl_id      = aws_wafv2_web_acl.main.arn
  price_class     = "PriceClass_100" # US/Canada/Europe only -- cheapest tier, fine for a small personal project

  origin {
    origin_id   = "ec2-nginx"
    domain_name = aws_eip.serving.public_dns

    custom_origin_config {
      http_port              = 80
      https_port              = 443
      origin_protocol_policy = "http-only" # nginx has no TLS listener -- CloudFront terminates TLS for viewers, origin fetch stays on the AWS backbone
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }

  default_cache_behavior {
    target_origin_id       = "ec2-nginx"
    viewer_protocol_policy = "redirect-to-https"
    allowed_methods         = ["GET", "HEAD", "OPTIONS", "PUT", "POST", "PATCH", "DELETE"]
    cached_methods          = ["GET", "HEAD"]

    # Dynamic content throughout (API responses, live dashboards) -- caching
    # disabled by default rather than tuned per path right now.
    forwarded_values {
      query_string = true
      headers      = ["*"]
      cookies {
        forward = "all"
      }
    }
    min_ttl     = 0
    default_ttl = 0
    max_ttl     = 0
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    cloudfront_default_certificate = true
  }
}
