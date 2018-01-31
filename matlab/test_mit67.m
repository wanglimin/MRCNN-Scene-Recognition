% demo code of feature extraction on the MIT67 dataset

addpath ../caffe/matlab % path of matcaffe
ind = 1:6700;

scale = 256;
crop_num = 3;
crop_dim = 224;

mkdir('mit67_result/');
file_name = 'mit67_result/256_inception2_feature.mat';
feature = zeros(1024, crop_num*crop_num*2*3, length(ind));

tmp=load('scene67_imagelist.mat');
file_list = tmp.imageList;

d = load('places2_mean.mat');
IMAGE_MEAN = d.mean_data;
if size(IMAGE_MEAN,1) ~= scale || size(IMAGE_MEAN,2) ~=scale
    IMAGE_MEAN = imresize(IMAGE_MEAN,[scale,scale]);
end

model_def_file = '../models/standard_train/256_inception2_deploy.prototxt';
model_file = '../models/places2_standard_256_inception2_v5.caffemodel';
gpu_id = 0;

caffe.reset_all();
caffe.set_mode_gpu();
caffe.set_device(gpu_id);
net = caffe.Net(model_def_file, model_file, 'test');

cnt = 1;
for i = ind
    tic;
    im_data = imread(file_list{i}); % read image
    if size(im_data,3) ~=3
        im_data = cat(3, im_data, im_data, im_data);
    end
    im_data = im_data(:, :, [3, 2, 1]); % convert from RGB to BGR
    im_data = permute(im_data, [2, 1, 3]); % permute width and height
    im_data = single(im_data); % convert to single precision
    
    if size(im_data,1) ~= scale || size(im_data, 2) ~=scale
        im_data = imresize(im_data, [scale, scale], 'bilinear','AntiAliasing',false);
    end
    im_data = im_data - IMAGE_MEAN;
    
    
    crop_num = 3; crop_dim = scale*1;
    crop_data_1 = multi_crop(im_data, crop_num, crop_dim, 224);
    crop_num = 3; crop_dim = round(scale*0.9375);
    crop_data_2 = multi_crop(im_data, crop_num, crop_dim, 224);
    crop_num = 3; crop_dim = round(scale*0.875);
    crop_data_3 = multi_crop(im_data, crop_num, crop_dim, 224);
    crop_data = cat(4, crop_data_1, crop_data_2, crop_data_3);
    
    net.blobs('data').set_data(crop_data);
    net.forward_prefilled();
    prediction = net.blobs('global_pool').get_data();
    feature(:,:,cnt) = squeeze(prediction);
    cnt = cnt + 1;
    toc;
end

save(file_name, 'feature', '-v7.3');

