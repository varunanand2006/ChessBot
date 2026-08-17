output "bucket" {
  description = "S3 bucket holding the generated tablebases."
  value       = aws_s3_bucket.tb.bucket
}

output "dest_uri" {
  description = "The --dest the pod uploads to."
  value       = "s3://${aws_s3_bucket.tb.bucket}/${var.tb_prefix}"
}

output "pod_id" {
  description = "RunPod pod id."
  value       = runpod_pod.batch.id
}

output "uploader_user" {
  description = "IAM user the pod uploads as (scoped to the bucket)."
  value       = aws_iam_user.uploader.name
}

output "next_steps" {
  description = "How to watch the run and stop billing."
  value = join("\n", [
    "Watch:   ssh into the pod (RunPod console) and `tail -f /workspace/chess/fanout_gpu*.log`",
    "Verify:  aws s3 ls s3://${aws_s3_bucket.tb.bucket}/${var.tb_prefix}/",
    "Stop:    ./tf.sh destroy   (RunPod pods do not self-terminate)",
  ])
}
