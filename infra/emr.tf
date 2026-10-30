resource "aws_emrserverless_application" "spark" {
  name          = "${var.project_prefix}-training"
  release_label = var.emr_release_label
  type          = "SPARK"

  network_configuration {
    subnet_ids         = data.aws_subnets.emr_compatible.ids
    security_group_ids = [aws_security_group.emr_serverless.id]
  }

  # The training image (docker/emr-serverless.Dockerfile), built FROM this
  # exact release's own EMR Serverless base image with optuna/mlflow/
  # psycopg2-binary/boto3 layered on top -- see that Dockerfile's comment.
  image_configuration {
    image_uri = "${aws_ecr_repository.training.repository_url}:latest"
  }

  # Auto-start/auto-stop are on by default -- no persistent workers, no idle
  # cost between jobs (see the plan: nothing to manually shut down).
}
