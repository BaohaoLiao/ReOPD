#!/bin/bash
# Launch local dense retrieval servers (Search-R1 style: e5 + wiki-18 corpus).
#
# Prerequisites (see local_dense_retriever/download.py for the index/corpus):
#   DATA_DIR      directory containing e5_Flat.index and wiki-18.jsonl
#   RETRIEVER     HF path or local dir of the retriever model (default intfloat/e5-base-v2)
#
# Usage:
#   DATA_DIR=/path/to/search_data bash eval/search/run_retrieval_server.sh
#
# Optionally activate a dedicated conda env first (faiss-gpu etc.):
#   source /opt/conda/etc/profile.d/conda.sh && conda activate retriever

set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

DATA_DIR=${DATA_DIR:?"set DATA_DIR to the directory holding e5_Flat.index and wiki-18.jsonl"}
index_file=${INDEX_FILE:-${DATA_DIR}/e5_Flat.index}
corpus_file=${CORPUS_FILE:-${DATA_DIR}/wiki-18.jsonl}
retriever_name=${RETRIEVER_NAME:-e5}
retriever_path=${RETRIEVER:-intfloat/e5-base-v2}
export CUDA_LAUNCH_BLOCKING=1

# Start 2 retrieval server processes, each using 4 GPUs (0-3 and 4-7), sharding the index across 4 GPUs per server
for group in 0 1; do
  if [ $group -eq 0 ]; then
    gpus="0,1,2,3"
    port=8000
  else
    gpus="4,5,6,7"
    port=8001
  fi
  echo "Starting retrieval server on GPUs $gpus, port $port..."
  CUDA_VISIBLE_DEVICES=$gpus nohup python "${SCRIPT_DIR}/local_dense_retriever/retrieval_server.py" \
    --index_path $index_file \
    --corpus_path $corpus_file \
    --topk 3 \
    --retriever_name $retriever_name \
    --retriever_model $retriever_path \
    --faiss_gpu \
    --port $port > retrieval_server_gpus_${gpus//,/}.log 2>&1 &
done

echo "Both retrieval servers started. Point SEARCH_URL at http://127.0.0.1:8000/retrieve,http://127.0.0.1:8001/retrieve"
