# export LOG_LEVEL=ERROR
# export DEBUG=Falseexport OH_RUNTIME_SINGULARITY_IMAGE_REPO=/lustre/fs1/portfolios/llmservice/users/shaokunz/Openhands2/OpenHands_internal/singularity_images_v2
export PYTHONPATH=$PWD:$PYTHONPATH
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export OH_RUNTIME_SINGULARITY_IMAGE_REPO=/lustre/fsw/portfolios/llmservice/users/shaokunz/Openhands/OpenHands_internal/singularity_images_v2/
nohup python scripts/start_server.py --max-init-workers 70 --max-run-workers 64 --timeout 1000  > log.txt 2>&1 &
