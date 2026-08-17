variable "region" {
  description = "AWS region for the S3 bucket + IAM user."
  type        = string
  default     = "us-east-1"
}

variable "name_prefix" {
  description = "Prefix for resource names/tags."
  type        = string
  default     = "chess-tb"
}

variable "bucket_name" {
  description = "S3 bucket for the .tb files. Empty auto-generates a unique name; pin to reuse across applies."
  type        = string
  default     = ""
}

variable "tb_prefix" {
  description = "Key prefix within the bucket. The workers' --dest is s3://<bucket>/<tb_prefix>."
  type        = string
  default     = "v1"
}

# --- RunPod pod --------------------------------------------------------------

variable "gpu_type_ids" {
  description = "RunPod GPU type(s), tried in order. e.g. NVIDIA GeForce RTX 4090, NVIDIA RTX A5000."
  type        = list(string)
  default     = ["NVIDIA GeForce RTX 4090"]
}

variable "gpu_count" {
  description = "GPUs on the single pod. The batch fans one worker per GPU, so this is the parallelism."
  type        = number
  default     = 4
}

variable "cloud_type" {
  description = "COMMUNITY (cheaper, less reliable) or SECURE (pricier, more available)."
  type        = string
  default     = "COMMUNITY"
}

variable "interruptible" {
  description = "Spot-equivalent: cheaper but reclaimable. Safe here (the batch is idempotent/resumable), off by default for a clean first run."
  type        = bool
  default     = false
}

variable "data_center_ids" {
  description = "Restrict to specific RunPod data centers (e.g. US-CA-2). Empty lets RunPod choose (better availability)."
  type        = list(string)
  default     = []
}

variable "container_disk_gb" {
  description = "Container disk: holds the build (~2 GB) + each GPU's in-flight .tb (~0.42 GB) + toolchain."
  type        = number
  default     = 40
}

variable "image_name" {
  description = "Base image the pod boots. A CUDA devel image (ships nvcc) so the bootstrap can build the engine."
  type        = string
  default     = "nvidia/cuda:12.6.2-devel-ubuntu24.04"
}

# --- Workload ----------------------------------------------------------------

variable "materials" {
  description = "Canonical materials to generate, fanned across the GPUs."
  type        = list(string)
  default     = ["KQRKR", "KRBKN", "KBNKN", "KQBKR"]
}

variable "repo_url" {
  description = "Git URL the pod clones to build the engine."
  type        = string
  default     = "https://github.com/varunanand2006/ChessBot.git"
}

variable "repo_ref" {
  description = "Branch/tag/commit to check out (must contain cloud/fanout.sh)."
  type        = string
  default     = "master"
}
