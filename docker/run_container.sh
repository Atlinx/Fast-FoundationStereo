docker rm -f ffs || true
xhost +local:root || true
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
REPO_DIR=$SCRIPT_DIR/../
docker run --gpus all --runtime nvidia --env NVIDIA_DISABLE_REQUIRE=1 -it --network=host --name ffs --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v $REPO_DIR:/workspace --ipc=host -e DISPLAY=${DISPLAY} -v /tmp/.X11-unix:/tmp/.X11-unix -v /tmp:/tmp -v /home:/home -v /mnt:/mnt ffs bash
