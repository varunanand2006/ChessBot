# AWS side: the bucket the pod uploads to + a least-privilege IAM user whose
# access key is injected into the pod. A RunPod pod is not an AWS principal, so
# it cannot assume an instance role (as the all-AWS module does); a scoped user
# with an access key is the standard way for off-AWS compute to reach S3. The
# trade vs the instance role: a long-lived key (held in TF state + the pod env),
# so it is scoped to PutObject/List on this one bucket and nothing else.

locals {
  bucket_name = var.bucket_name != "" ? var.bucket_name : "${var.name_prefix}-${random_id.suffix.hex}"
  tags        = { Project = var.name_prefix, ManagedBy = "terraform" }
}

resource "random_id" "suffix" {
  byte_length = 4
}

# --- Bucket (same posture as the all-AWS module) ---------------------------
resource "aws_s3_bucket" "tb" {
  bucket = local.bucket_name
  tags   = local.tags

  # Let `terraform destroy` delete the bucket even with tables (and versions) in
  # it. Without this, destroy fails "BucketNotEmpty" and you must empty it by hand
  # first. Fine here: the tables are regenerable and downloaded before teardown.
  force_destroy = true
}

resource "aws_s3_bucket_versioning" "tb" {
  bucket = aws_s3_bucket.tb.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_public_access_block" "tb" {
  bucket                  = aws_s3_bucket.tb.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "tb" {
  bucket = aws_s3_bucket.tb.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# --- IAM user the pod uploads as -------------------------------------------
resource "aws_iam_user" "uploader" {
  name = "${var.name_prefix}-uploader"
  tags = local.tags
}

data "aws_iam_policy_document" "uploader" {
  statement {
    sid       = "ListBucket"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.tb.arn]
  }
  statement {
    sid       = "ReadWriteObjects"
    actions   = ["s3:GetObject", "s3:PutObject"]
    resources = ["${aws_s3_bucket.tb.arn}/*"]
  }
}

resource "aws_iam_user_policy" "uploader" {
  name   = "${var.name_prefix}-s3"
  user   = aws_iam_user.uploader.name
  policy = data.aws_iam_policy_document.uploader.json
}

resource "aws_iam_access_key" "uploader" {
  user = aws_iam_user.uploader.name
}
