# Inputs. Sensible defaults for the tablebase batch; override in terraform.tfvars.

variable "region" {
  description = "AWS region to deploy into."
  type        = string
  default     = "us-east-1"
}

variable "name_prefix" {
  description = "Prefix for all resource names/tags."
  type        = string
  default     = "chess-tb"
}

variable "bucket_name" {
  description = <<-EOT
    S3 bucket for the generated .tb files. S3 names are globally unique; leave
    empty to auto-generate "<name_prefix>-<random>". Set explicitly to reuse a
    bucket across applies (so a destroyed instance's tables persist).
  EOT
  type        = string
  default     = ""
}

variable "tb_prefix" {
  description = "Key prefix (folder) within the bucket, e.g. 'v1'. The worker's --dest is s3://<bucket>/<tb_prefix>."
  type        = string
  default     = "v1"
}

variable "instance_type" {
  description = <<-EOT
    GPU instance type. Default g5.xlarge = 1x A10G (sm_86), 4 vCPU — fits an
    8-vCPU spot quota, and two of them is the multi-instance batch. g5.12xlarge
    (4x A10G, 48 vCPU) needs a larger quota.
  EOT
  type        = string
  default     = "g5.xlarge"
}

variable "materials" {
  description = "Materials for the worker to generate, canonical names (K + white pieces + K + black pieces)."
  type        = list(string)
  default     = ["KQKR", "KRKN", "KQKN", "KRKB"]
}

variable "repo_url" {
  description = "Git URL the instance clones to build the worker image."
  type        = string
  default     = "https://github.com/varunanand/chess.git"
}

variable "repo_ref" {
  description = "Branch/tag/commit the instance checks out."
  type        = string
  default     = "master"
}

variable "root_volume_gb" {
  description = "Root EBS size. Needs room for the CUDA build + ~4.4 GB image + tables."
  type        = number
  default     = 100
}
