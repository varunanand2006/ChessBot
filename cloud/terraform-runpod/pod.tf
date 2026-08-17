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

  # Boot to a stable idle container. Driving a finite batch through a Pod start
  # command crash-loops: RunPod Pods expect a long-lived container, so when the
  # bootstrap script exits (success OR failure) RunPod restarts it, with no
  # visibility. Instead we idle here, then run the batch by hand in the RunPod
  # web terminal with live output. TF still injects DEST/MATERIALS/AWS creds into
  # `env` below, so the manual run just uses them. pod_start.sh holds the exact
  # steps to paste (and is the basis for re-automating later, e.g. via serverless).
  docker_start_cmd = ["sleep", "infinity"]
}
