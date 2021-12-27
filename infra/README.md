```bash
npm install aws-cdk@1.129.0
```

## Set up

```
pip install -r ./infra/src/requirements.txt
```

## CDK Deploy

### Deploy Repos stack

This stack creates code commit and ECR repo .

This deploys the stack into the aws profile pointed to using the example named profile "default"

```bash
     export PYTHONPATH=./infra/src
     cdk --app "python infra/src/app.py" deploy  bertreverse-Repos  --parameters DockerRepoName=bertreverse --profile default 
```

Make a note of the output arns e.g below to use in the next step

```text
Ppiaimed-Repos.OutputECRARN = arn:aws:ecr:us-west-2:11111:repository/bertreverse
Ppiaimed-Repos.OutputECRARN =  arn:aws:secretsmanager:us-west-2:111:secret:githubauthCD82D024-18om8waIrUBI-JRbHkQ

```

### Deploy CI Pipeline stack

To deploy a continous integration pipeline using AWS codepipeline to build a docker image and push to ecr created in the
previous step

```bash
cdk --app "python infra/src/app.py" deploy  bertreverse-CIPipeline --parameters  GithubUrl="https://github.com/elangovana/bert-reverse"  --parameters BranchName=master  --parameters DockerRepoArn=arn:aws:ecr:us-east-2:111:repository/bertreverse  --parameters secretarn=arn:aws:secretsmanager:us-west-2:111:secret:githubauthCD82D024-18om8waIrUBI-JRbHkQ
```
