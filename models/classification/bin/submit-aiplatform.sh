GRN='\033[0;32m'
RED='\033[0;31m'
PURPLE='\033[0;34m'
RST='\033[0m'

if [ $# -ne 8 ]
  then
    echo "Usage ./submit-aiplatform.sh IMAGE_NAME IMAGE_TAG JOB_NAME MACHINE_CONFIG_PATH DATA_DIR BUCKET MODEL_TYPE MODEL_NAME"
        exit 1
fi

PROJECT_ID=$(gcloud config list project --format "value(core.project)")
IMAGE_REPO_NAME=$1
IMAGE_TAG=$2
IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG
JOB_NAME=$3
MACHINE_CONFIG=$4
DATA_DIR=$5
BUCKET=$6
MODEL_TYPE=$7
MODEL_NAME=$8

echo "Submitting ${GRN}${JOB_NAME}${RST} with image ${GRN}${IMAGE_URI}${RST}."
echo "Use machine type config at tier set to ${GRN}${MACHINE_CONFIG}${RST}."
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
        --config $MACHINE_CONFIG \
        -- \
        --task_name $JOB_NAME \
        --data_dir $DATA_DIR \
        --save_steps 10000 \
        --logging_steps 2000 \
        --num_train_epochs 1 \
        --max_seq_length 512 \
        --model_type $MODEL_TYPE \
        --model_name_or_path $MODEL_NAME \
        --bucket $BUCKET \
        --output_dir /tmp \
        --do_train \
        --do_eval \
        --evaluate_during_training \
        --overwrite_output_dir \
        --seq_len_func rs \
        --seq_func_params 16
    else
        echo "Aborting..."
        exit 1
fi
