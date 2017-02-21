#!/usr/bin/env bash

# standard trained models, 2 resolutions
wget -O models/places2_standard_256_inception2_v5.caffemodel http://mmlab.siat.ac.cn/MRCNN_model/places2_standard_256_inception2_v5.caffemodel
wget -O models/places2_standard_384_inception2_v5.caffemodel http://mmlab.siat.ac.cn/MRCNN_model/places2_standard_384_inception2_v5.caffemodel

# object kd trained models, 2 resolutions
wget -O models/places2_standard_256_object_kd_inception2_v5.caffemodel http://mmlab.siat.ac.cn/MRCNN_model/places2_standard_256_object_kd_inception2_v5.caffemodel
wget -O models/places2_standard_384_object_kd_inception2_v5.caffemodel http://mmlab.siat.ac.cn/MRCNN_model/places2_standard_384_object_kd_inception2_v5.caffemodel

# scene kd trained models, 2 resolutions
wget -O models/places2_standard_256_scene_kd_inception2_v5.caffemodel http://mmlab.siat.ac.cn/MRCNN_model/places2_standard_256_scene_kd_inception2_v5.caffemodel
wget -O models/places2_standard_384_scene_kd_inception2_v5.caffemodel http://mmlab.siat.ac.cn/MRCNN_model/places2_standard_384_scene_kd_inception2_v5.caffemodel

