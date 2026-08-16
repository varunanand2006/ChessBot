# Storage + the data sources everything else hangs off.

locals {
  # Auto-name the bucket when one isn't pinned. random_id keeps it globally unique.
  bucket_name = var.bucket_name != "" ? var.bucket_name : "${var.name_prefix}-${random_id.suffix.hex}"
  tags        = { Project = var.name_prefix, ManagedBy = "terraform" }
}

resource "random_id" "suffix" {
  byte_length = 4
}

# --- Tablebase bucket ------------------------------------------------------
# The single source of truth the workers fill and the engine (later) reads.
resource "aws_s3_bucket" "tb" {
  bucket = local.bucket_name
  tags   = local.tags
}

# Versioning: a re-run overwrites <MATERIAL>.tb with a byte-identical file, but
# versioning means an accidental bad upload is recoverable rather than silently
# clobbering the good table.
resource "aws_s3_bucket_versioning" "tb" {
  bucket = aws_s3_bucket.tb.id
  versioning_configuration {
    status = "Enabled"
  }
}

# The tables are not secret, but there is no reason to expose them: block all
# public access and default-encrypt at rest.
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

# --- Network + image lookups ----------------------------------------------
# Use the account's default VPC/subnets — this is a batch job, not a service, so
# it needs no custom networking, just outbound to S3 and the package mirrors.
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# AWS Deep Learning Base AMI: ships NVIDIA drivers + Docker + the NVIDIA
# container toolkit, so `docker run --gpus all` works with no driver install in
# user_data. The name pattern can drift; adjust if a plan finds no AMI.
data "aws_ami" "dlami" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*"]
  }
  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}
