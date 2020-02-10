GRN='\033[0;32m'
RED='\033[0;31m'
PURPLE='\033[0;34m'
RST='\033[0m'

if [ $# -ne 3 ]
  then
    echo "Usage ./register-image.sh IMAGE_NAME IMAGE_TAG DOCKERFILE_DIRECTORY."
        exit 1
fi

PROJECT_ID=$(gcloud config list project --format "value(core.project)")
IMAGE_REPO_NAME=$1
IMAGE_TAG=$2
DOCKER_DIR=$3
IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG

echo "Registering image called ${GRN}${IMAGE_REPO_NAME}${RST} with tag ${GRN}${IMAGE_TAG}${RST} in project ${GRN}${PROJECT_ID}${RST}. Continue? (y/n)"
read prompt

if [ "$prompt" == "y" ]
    then
        cd $DOCKER_DIR
        docker build -f Dockerfile -t $IMAGE_URI ./
        docker push $IMAGE_URI
    else
        echo "Aborting..."
        exit 1
fi