import dagshub
dagshub.init(repo_owner='rudratyagi777', repo_name='fast_api_tutorials', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)