% demo code of test on the Places2 validation set

addpath ../caffe/matlab

IMAGE_DIM = 256;
CROPPED_DIM = 224;

midir('places2_result/')
file_name = 'places2/val_result.mat';
score = zeros(365, 10, length(ind));
val_file = 'places365_val.txt';
val_path = '/nfs_data/share/data/Places365/High_Resolution/val_large/';
tmp = importdata(val_file);
file_list = tmp.textdata;
ind = 1:36500;

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
    im_data = imread([val_path,file_list{i}]); % read image
    if size(im_data,3) ~=3
        im_data = cat(3, im_data, im_data, im_data);
    end
    im_data = im_data(:, :, [3, 2, 1]); % convert from RGB to BGR
    im_data = permute(im_data, [2, 1, 3]); % permute width and height
    im_data = single(im_data); % convert to single precision
    
    if size(im_data,1) ~= IMAGE_DIM  || size(im_data, 2) ~= IMAGE_DIM 
        im_data = imresize(im_data, [IMAGE_DIM , IMAGE_DIM ], 'bilinear','AntiAliasing',false);
        
    end
    im_data = bsxfun(@minus, im_data, reshape([105, 113, 116], [1,1,3]));
    crops_data = zeros(CROPPED_DIM, CROPPED_DIM, 3, 10, 'single');
    indices = [0 IMAGE_DIM-CROPPED_DIM] + 1;
    n = 1;
    for k = indices
        for j = indices
            crops_data(:, :, :, n) = im_data(k:k+CROPPED_DIM-1, j:j+CROPPED_DIM-1, :);
            crops_data(:, :, :, n+5) = crops_data(end:-1:1, :, :, n);
            n = n + 1;
        end
    end
    center = floor(indices(2) / 2) + 1;
    crops_data(:,:,:,5) = ...
        im_data(center:center+CROPPED_DIM-1,center:center+CROPPED_DIM-1,:);
    crops_data(:,:,:,10) = crops_data(end:-1:1, :, :, 5);
    
    net.blobs('data').set_data(crops_data);
    net.forward_prefilled();
    prediction = net.blobs('fc').get_data();
    score(:,:,cnt) = prediction;
    cnt = cnt + 1;
    toc;
end

score = score(:,:,1:cnt-1);
score = exp(score);
score = bsxfun(@rdivide, score, sum(score, 1));
score = squeeze(mean(score,2));
save(file_name, 'score', '-v7.3');

