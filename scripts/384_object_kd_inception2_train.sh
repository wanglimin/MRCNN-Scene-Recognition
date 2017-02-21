#!/usr/bin/env sh

TOOLS=/home/lmwang/code/caffe/cmake_build_kd/install/bin

/usr/local/openmpi/bin/mpirun -n 8 \
$TOOLS/caffe train --solver=models/kd_train/384_object_kd_inception2_solver.prototxt --weights=models/imagenet_384_inception2_v5.caffemodel
    
echo "Done."

