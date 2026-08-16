# The spot GPU worker itself: no inbound access, spot-priced, self-terminating.

# Egress-only security group. No ingress at all — a shell comes via SSM Session
# Manager (see iam.tf), which needs no open port. That is the stronger security
# posture than an SSH rule, and it is the whole reason SSM is wired up.
resource "aws_security_group" "worker" {
  name        = "${var.name_prefix}-worker"
  description = "Egress-only; management via SSM, no inbound."
  vpc_id      = data.aws_vpc.default.id

  egress {
    description = "All outbound (S3, package mirrors, GitHub)."
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = local.tags
}

resource "aws_instance" "worker" {
  ami                    = data.aws_ami.dlami.id
  instance_type          = var.instance_type
  subnet_id              = data.aws_subnets.default.ids[0]
  iam_instance_profile   = aws_iam_instance_profile.worker.name
  vpc_security_group_ids = [aws_security_group.worker.id]

  # Spot: ~60-70% off on-demand. instance_interruption_behavior must be
  # terminate for a one-shot batch worker (a persistent spot request would try
  # to relaunch; here the next `apply`/instance handles resume via bucket skip).
  instance_market_options {
    market_type = "spot"
    spot_options {
      instance_interruption_behavior = "terminate"
    }
  }

  # `shutdown -h now` at the end of user_data terminates rather than stops, so an
  # idle worker stops billing without manual cleanup.
  instance_initiated_shutdown_behavior = "terminate"

  root_block_device {
    volume_size = var.root_volume_gb
    volume_type = "gp3"
  }

  user_data = templatefile("${path.module}/user_data.sh.tftpl", {
    repo_url  = var.repo_url
    repo_ref  = var.repo_ref
    bucket    = aws_s3_bucket.tb.bucket
    tb_prefix = var.tb_prefix
    materials = join(" ", var.materials)
  })

  tags = merge(local.tags, { Name = "${var.name_prefix}-worker" })
}
