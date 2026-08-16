# The instance's identity. Least-privilege by design: it can touch exactly one
# bucket and be managed by SSM — nothing else. No long-lived keys anywhere; the
# instance assumes this role via its profile and the SDK/CLI pick it up.

data "aws_iam_policy_document" "assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "worker" {
  name               = "${var.name_prefix}-worker"
  assume_role_policy = data.aws_iam_policy_document.assume.json
  tags               = local.tags
}

# S3: list the bucket (to check "is <MATERIAL>.tb already there?") and read/write
# the objects under it. Scoped to this bucket only; no Delete.
data "aws_iam_policy_document" "s3" {
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

resource "aws_iam_role_policy" "s3" {
  name   = "${var.name_prefix}-s3"
  role   = aws_iam_role.worker.id
  policy = data.aws_iam_policy_document.s3.json
}

# SSM Session Manager: a keyless shell into the box for debugging, so the
# security group needs NO inbound SSH (port 22). This managed policy is the
# minimum the SSM agent needs to register.
resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.worker.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "worker" {
  name = "${var.name_prefix}-worker"
  role = aws_iam_role.worker.name
  tags = local.tags
}
