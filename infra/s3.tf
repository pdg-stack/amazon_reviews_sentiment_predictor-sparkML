resource "aws_s3_bucket" "main" {
  bucket = "${var.project_prefix}-${var.aws_account_id}-${var.aws_region}"

  # POC project: a `tofu destroy` should be able to tear this down in one
  # shot, not get blocked on emptying it by hand first.
  force_destroy = true
}

resource "aws_s3_bucket_public_access_block" "main" {
  bucket = aws_s3_bucket.main.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "main" {
  bucket = aws_s3_bucket.main.id
  versioning_configuration {
    status = "Disabled" # keep costs flat for a small POC dataset; revisit if this needs recovery guarantees later
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "main" {
  bucket = aws_s3_bucket.main.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}
