# Provider + Terraform version constraints.
#
# State: this v1 uses LOCAL state (a terraform.tfstate file beside the config).
# The remote S3 backend below is intentionally commented out — it is a
# chicken-and-egg with the very bucket this config creates, so the standard
# pattern is: apply once with local state, then create a separate state bucket +
# DynamoDB lock table and migrate. Uncomment and `terraform init -migrate-state`
# when you want shared/remote state.

terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
  }

  # backend "s3" {
  #   bucket         = "chess-tb-tfstate-<youracct>"
  #   key            = "tablebase/terraform.tfstate"
  #   region         = "us-east-1"
  #   dynamodb_table = "chess-tb-tflock"
  #   encrypt        = true
  # }
}

provider "aws" {
  region = var.region
}
