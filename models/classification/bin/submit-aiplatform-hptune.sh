GRN='\033[0;32m'
RED='\033[0;31m'
PURPLE='\033[0;34m'
RST='\033[0m'

if [ $# -ne 8 ]
  then
    echo "Usage ./submit-mlengine.sh IMAGE_NAME IMAGE_TAG HPTUNING_CONFIG_PATH DATA_DIR BUCKET MODEL_TYPE MODEL_NAME JOB_NAME"
        exit 1
fi

PROJECT_ID=$(gcloud config list project --format "value(core.project)")
IMAGE_REPO_NAME=$1
IMAGE_TAG=$2
IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG
HPTUNING_CONFIG=$3
DATA_DIR=$4
BUCKET=$5
MODEL_TYPE=$6
MODEL_NAME=$7
JOB_NAME=$8

echo "Submitting ${GRN}${JOB_NAME}${RST} with image ${GRN}${IMAGE_URI}${RST}."
echo "Use hyperparameter tuning config at tier set to ${GRN}${HPTUNING_CONFIG_PATH}${RST}."
echo "Data directory set to ${GRN}${DATA_DIR}${RST}" 
echo "Bucket set to ${GRN}${BUCKET}${RST}"
echo "${RED}NOTE: check this script for other job variables related to the model! ${RST}"
echo "Continue? (y/n)"
read prompt

if [ "$prompt" == "y" ]
    then
      gcloud ai-platform jobs submit training $JOB_NAME \
        --stream-logs \
        --master-image-uri $IMAGE_URI \
        --region us-central1 \
        --config $HPTUNING_CONFIG \
        -- \
        --task_name $JOB_NAME \
        --data_dir $DATA_DIR \
        --save_steps 1000 \
        --logging_steps 100 \
        --model_type $MODEL_TYPE \
        --model_name_or_path $MODEL_NAME \
        --bucket $BUCKET \
        --output_dir ./$JOB_NAME \
        --do_train \
        --do_eval \
        --overwrite_output_dir \
        --fp16 \
        --fp16_opt_level=O1 \
        --do_lower_case \
        --slack_channel jobs \
        --seq_len_func rs
    else
        echo "Aborting..."
        exit 1
fi
