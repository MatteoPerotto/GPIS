# ====================================
# Run commands for deveopement 
# ====================================
xhost +
docker run \
    -it \
    --name $1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    ubuntu:gpis
xhost -
