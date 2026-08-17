# The multi-GPU batch pod. gpu_count GPUs on one node; the bootstrap fans one
# worker per GPU (cloud/fanout.sh). Config arrives entirely through env, so the
# same image serves any material list without a rebuild.
resource "runpod_pod" "batch" {
  name       = "${var.name_prefix}-batch"
  image_name = var.image_name

  gpu_type_ids = var.gpu_type_ids
  gpu_count    = var.gpu_count
  cloud_type   = var.cloud_type

  # Spot-equivalent when true; safe because the batch skips finished tables.
  interruptible = var.interruptible

  # null lets RunPod pick any data center (best availability); a non-empty list
  # restricts placement.
  data_center_ids = length(var.data_center_ids) > 0 ? var.data_center_ids : null

  container_disk_in_gb = var.container_disk_gb

  # SSH in to watch progress; no inbound service, just management.
  support_public_ip = true
  ports             = ["22/tcp"]

  # Everything the bootstrap reads. The AWS key is created by this same apply and
  # injected here, so there is no manual credential copying. (It lands in TF
  # state + the pod config, which is why the user is scoped to one bucket.)
  env = {
    DEST                  = "s3://${aws_s3_bucket.tb.bucket}/${var.tb_prefix}"
    MATERIALS             = join(" ", var.materials)
    REPO_URL              = var.repo_url
    REPO_REF              = var.repo_ref
    AWS_ACCESS_KEY_ID     = aws_iam_access_key.uploader.id
    AWS_SECRET_ACCESS_KEY = aws_iam_access_key.uploader.secret
    AWS_DEFAULT_REGION    = var.region
  }

  # Bootstrap read from a file so the shell logic stays readable (and testable)
  # instead of an inline HCL string. Its env-var references expand on the pod.
  docker_start_cmd = ["bash", "-lc", file("${path.module}/pod_start.sh")]
}
