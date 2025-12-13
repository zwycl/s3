#!/bin/bash
# Start all services: retriever, generator, and training
# GPU allocation (8 GPUs):
#   GPUs 0-5: Retriever (FAISS)
#   GPUs 6-7: Generator LLM (tensor-parallel-size=2)
#   GPUs 0-7: Training (8 GPU distributed training)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Configuration
RETRIEVER_PORT=3000
GENERATOR_PORT=8000
LOG_DIR="$PROJECT_DIR/train_logs"
mkdir -p "$LOG_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_port() {
    local port=$1
    if ss -tlnp | grep -q ":$port "; then
        return 0  # Port in use
    else
        return 1  # Port free
    fi
}

wait_for_port() {
    local port=$1
    local service=$2
    local timeout=${3:-120}
    local elapsed=0

    log_info "Waiting for $service on port $port..."
    while ! check_port $port; do
        sleep 2
        elapsed=$((elapsed + 2))
        if [ $elapsed -ge $timeout ]; then
            log_error "$service failed to start within ${timeout}s"
            return 1
        fi
    done
    log_info "$service is ready on port $port"
    return 0
}

# Kill existing processes
log_info "Cleaning up existing processes..."
pkill -9 -f "retrieval_server" 2>/dev/null || true
pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
pkill -9 -f "verl.trainer" 2>/dev/null || true
pkill -9 -f "train_s3" 2>/dev/null || true
sleep 3

# Check GPU memory
log_info "Checking GPU memory..."
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

# ============================================
# Step 1: Start Retriever on GPUs 0-1
# FAISS index + e5 model
# ============================================
log_info "Starting Retriever on GPUs 0-1..."

export CUDA_VISIBLE_DEVICES=0,1
file_path=./data
index_file=$file_path/e5_Flat.index
corpus_file=$file_path/wiki-18.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2

nohup python s3/search/retrieval_server.py \
    --index_path $index_file \
    --corpus_path $corpus_file \
    --topk 12 \
    --retriever_name $retriever_name \
    --retriever_model $retriever_path \
    --port $RETRIEVER_PORT \
    --faiss_gpu \
    > "$LOG_DIR/retriever.log" 2>&1 &

RETRIEVER_PID=$!
log_info "Retriever started with PID $RETRIEVER_PID"

# Wait for retriever to be ready
if ! wait_for_port $RETRIEVER_PORT "Retriever" 180; then
    log_error "Retriever failed to start. Check $LOG_DIR/retriever.log"
    exit 1
fi

# ============================================
# Step 2: Start Generator LLM on GPUs 6-7
# Qwen2.5-14B-GPTQ needs ~15GB per GPU
# ============================================
log_info "Starting Generator LLM on GPUs 6-7..."

export CUDA_VISIBLE_DEVICES=6,7
nohup python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4 \
    --port $GENERATOR_PORT \
    --max-model-len 8192 \
    --tensor-parallel-size 2 \
    > "$LOG_DIR/generator.log" 2>&1 &

GENERATOR_PID=$!
log_info "Generator started with PID $GENERATOR_PID"

# Wait for generator to be ready
if ! wait_for_port $GENERATOR_PORT "Generator" 120; then
    log_error "Generator failed to start. Check $LOG_DIR/generator.log"
    exit 1
fi

# ============================================
# Step 3: Start Training on GPUs 2-5
# Training uses 4 GPUs (avoiding GPUs 0-1 for retriever, GPUs 6-7 for generator)
# ============================================
log_info "Starting Training on GPUs 2-5..."
log_info "Checking GPU memory before training..."
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

export CUDA_VISIBLE_DEVICES=2,3,4,5
export VLLM_ATTENTION_BACKEND=XFORMERS

RANDOM_SEED=42
DATA_DIR=data/nq_hotpotqa_train
BASE_MODEL='Qwen/Qwen2.5-7B-Instruct-1M'
EXPERIMENT_NAME="s3_8_3_3_${RANDOM_SEED}"

nohup python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train_e5_s3.parquet \
    data.val_files=$DATA_DIR/test_e5_s3.parquet \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=256 \
    data.val_batch_size=64 \
    data.max_prompt_length=8000 \
    data.max_response_length=500 \
    data.max_start_length=2000 \
    data.max_obs_length=1400 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=gae \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.state_masking=true \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.optim.lr_warmup_steps_ratio=0.01 \
    critic.model.path=$BASE_MODEL \
    critic.model.enable_gradient_checkpointing=true \
    critic.ppo_micro_batch_size=8 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.no_think_rl=false \
    trainer.critic_warmup=0 \
    trainer.logger=[] \
    +trainer.val_only=false \
    +trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=1500 \
    trainer.project_name=SearchAgent \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=4 \
    trainer.total_training_steps=1500 \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    +data.random_seed=$RANDOM_SEED \
    max_turns=3 \
    +generator_llm="Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4" \
    +output_context_dir="data/output_sequences_s3_8_3_3_new" \
    retriever.url="http://127.0.0.1:3000/retrieve" \
    retriever.topk=8 \
    > "$LOG_DIR/$EXPERIMENT_NAME.log" 2>&1 &

TRAIN_PID=$!
log_info "Training started with PID $TRAIN_PID"

# ============================================
# Summary
# ============================================
echo ""
log_info "=========================================="
log_info "All services started!"
log_info "=========================================="
echo ""
echo "Services:"
echo "  - Retriever:  http://127.0.0.1:$RETRIEVER_PORT (GPUs 0-1)"
echo "  - Generator:  http://127.0.0.1:$GENERATOR_PORT (GPUs 6-7)"
echo "  - Training:   Running on GPUs 2-5 (4 GPU distributed)"
echo ""
echo "Logs:"
echo "  - Retriever:  $LOG_DIR/retriever.log"
echo "  - Generator:  $LOG_DIR/generator.log"
echo "  - Training:   $LOG_DIR/s3_8_3_3_42.log"
echo ""
echo "To monitor training:"
echo "  tail -f $LOG_DIR/s3_8_3_3_42.log"
echo ""
echo "GPU Memory:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
