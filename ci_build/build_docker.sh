# exit when any command fails
set -e

DOCKER_BASE_DIR=$1
IMAGE_REPO=$2 # e.g. of image repo 111111.dkr.ecr.us-east-2.amazonaws.com/image
# Note: The account needs to match the pytorch release images, see https://github.com/aws/deep-learning-containers/blob/master/available_images.md
# e.g.763104351884.dkr.ecr.us-east-2.amazonaws.com/pytorch-training
PYTORCH_REPO=$3
echo Running with arguments "$@"


ECR_REGION=$(echo $IMAGE_REPO | cut -f 4 -d ".")
IMAGE_REPO_ACCOUNT=$(echo $IMAGE_REPO | cut -f 1 -d ".")
PYTORCH_DOCKER_ACCOUNT=$(echo $PYTORCH_REPO | cut -f 1 -d ".")
echo Building the Docker image $IMAGE_REPO from base ${PYTORCH_DOCKER_ACCOUNT_URL}...

function build_docker(){
  DEVICE=$1
  TAG=$2
  RUN_TEST=$3
  CUDA=$4
  LATEST_TAG=$5

  echo build docker with args "$@"

  # Login to pytorch ECR before build
  # aws ecr get-login-password --region region | docker login --username AWS --password-stdin ${PYTORCH_DOCKER_ACCOUNT_URL}
  $(aws ecr get-login --no-include-email --region $ECR_REGION --registry-ids ${PYTORCH_DOCKER_ACCOUNT} )

  docker build -t $IMAGE_REPO:$TAG   -f docker/Dockerfile $DOCKER_BASE_DIR --build-arg device=$DEVICE --build-arg account_url=${PYTORCH_REPO} --build-arg cuda=${CUDA}
  if [  "$RUN_TEST" == "Y" ]; then
    echo "Running tests..."
    #docker run  --mount type=bind,source="$(pwd)"/tests,target=/temp_source_tests,readonly   --entrypoint  "bash -c pip install -r /temp_source_tests/requirements_test.txt; pytest /temp_source_tests/*"  $IMAGE_REPO:$TAG
    docker run  --mount type=bind,source="$(pwd)"/tests,target=/temp_source_tests,readonly   --entrypoint  "/bin/bash -c echo hi; echo gello; ls /temp_source_tests"  $IMAGE_REPO:$TAG
  fi

  echo Logging in to Amazon ECR...
  $(aws ecr get-login --no-include-email --region $ECR_REGION --registry-ids ${IMAGE_REPO_ACCOUNT})
  echo Pushing the Docker image $IMAGE_REPO:$TAG...
  docker push $IMAGE_REPO:$TAG

  if [ -n "$LATEST_TAG" ]; then
    echo tagging the Docker image $IMAGE_REPO:$LATEST_TAG...
    docker tag $IMAGE_REPO:$TAG $IMAGE_REPO:$LATEST_TAG
    docker push $IMAGE_REPO:$LATEST_TAG

  fi

}

## TODO: Automate version tagging based on datetime for now, ideally should be tied to release tags
VERSION=$(date '+%Y%m%d%H%M')

device=cpu
LATEST_TAG=$device-latest
VERSION_TAG=$device-$VERSION
build_docker $device $VERSION_TAG Y

device=gpu
cuda="-cu113"
LATEST_TAG=$device-latest
VERSION_TAG=$device-$VERSION
build_docker $device $VERSION_TAG  N $cuda $LATEST_TAG

