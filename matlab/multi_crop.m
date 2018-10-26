function  crop_data = multi_crop(img_data, crop_num, crop_dim, dim)
scale1 = size(img_data, 1);
scale2 = size(img_data, 2);
stride1 = (scale1 - crop_dim)/(crop_num - 1);
stride2 = (scale2 - crop_dim)/(crop_num - 1);
crop_data = zeros(dim, dim, 3, crop_num*crop_num, 'single');
cnt = 1;

for i = 1:crop_num
    for j = 1:crop_num
        start1 = round((i-1)*stride1+1);
        if start1 > scale1-crop_dim+1
            start1 = scale1-crop_dim+1;
        end
        start2 = round((j-1)*stride2+1);
        if start2 > scale2-crop_dim+1
            start2 = scale2-crop_dim+1;
        end
        if crop_dim == dim
            crop_data(:,:,:,cnt) = img_data(start1:start1+crop_dim-1, start2:start2+crop_dim-1,:);
        else
            crop_data(:,:,:,cnt) = imresize(img_data(start1:start1+crop_dim-1, start2:start2+crop_dim-1,:), [dim, dim], 'bilinear');
        end
        cnt = cnt + 1;
    end
end

crop_data = cat(4, crop_data, crop_data(end:-1:1,:,:,:));

end
