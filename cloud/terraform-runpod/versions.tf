# Combo stack: RunPod for the multi-GPU batch compute, AWS S3 for durable
# storage. One apply provisions the bucket + a scoped IAM user, then launches a
# RunPod pod with those credentials injected into its environment.
#
# This is a SEPARATE config from cloud/terraform/ (the all-AWS alternative) so
# that validated module is left untouched. Local state, same as there.

terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    runpod = {
      # Community provider (no official RunPod one). Exposes runpod_pod with
      # gpu_count, so a 4-GPU node is a single resource.
      source  = "decentralized-infrastructure/runpod"
      version = "~> 1.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
  }
}

provider "aws" {
  region = var.region
}

# Reads the RUNPOD_API_KEY environment variable (passed through by tf.sh).
provider "runpod" {}
