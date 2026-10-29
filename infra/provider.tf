provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project   = "amazon-reviews-sentiment-predictor"
      ManagedBy = "opentofu"
    }
  }
}
