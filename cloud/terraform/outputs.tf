output "bucket" {
  description = "S3 bucket holding the generated tablebases."
  value       = aws_s3_bucket.tb.bucket
}

output "dest_uri" {
  description = "The --dest the worker writes to."
  value       = "s3://${aws_s3_bucket.tb.bucket}/${var.tb_prefix}"
}

output "instance_id" {
  description = "Worker instance id."
  value       = aws_instance.worker.id
}

output "ami_id" {
  description = "Resolved DLAMI id (sanity-check the AMI lookup)."
  value       = data.aws_ami.dlami.id
}

output "ssm_connect" {
  description = "Open a keyless shell on the worker (needs the AWS CLI + Session Manager plugin)."
  value       = "aws ssm start-session --target ${aws_instance.worker.id} --region ${var.region}"
}

output "tail_boot_log" {
  description = "Watch the build+generate progress once connected via SSM."
  value       = "tail -f /var/log/cloud-init-output.log"
}
