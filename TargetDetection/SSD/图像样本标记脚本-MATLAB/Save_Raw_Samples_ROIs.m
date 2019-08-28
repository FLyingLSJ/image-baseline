
%函数说明：读取通过MATLAB Training Image标记并保存的labelingSession数据，将
%这些标记的ROI区域转换成bmp图片并存储。图片的名称采用了时间+序号的存储方式，
%所以多次读取这一函数并将结果存入同一文件夹也不会造成命名冲突
%示例调用方式 Save_Raw_Samples_ROIs('E:\TSY\Code\labelingSession.mat','G:\Samples\2016-3-5');
%**************************************************************************
%第一个参数：通过MATLAB Training Image
%Labeler工具导出的labelingSession数据，为mat格式，这里只需输入其全路径即可，如'C:\labelingSession.mat'
%第二个参数：为待导出的文件存储的文件夹，如'G:\Samples\2016-3-1'
%所有标记过的图像从0开始索引为文件名进行存储
function Save_Raw_Samples_ROIs(labelingsession_data_path,output_path)
roi_info = load(labelingsession_data_path);
image_counts = size(roi_info.labelingSession.ImageSet.ROIBoundingBoxes,2);%struct结构与普通矩阵的索引方式不同，第一个为列，第二个为行
roi_index = 0;
pause(1);%延时1s，确保不会出现重命名
%current_time = datestr(now,'yyyy-mm-dd-HH-MM-SS_');
if ~isdir(output_path) %判断路径是否存在
    mkdir(output_path);
end
fid = fopen('D:\\code\\fasterrcnn\\output_img_label\\pos.txt','wt');
for i = 1:image_counts%对每张图依次处理，提取ROI区域
image_info = roi_info.labelingSession.ImageSet.ROIBoundingBoxes(i);
image_name = image_info.imageFilename;
names=strsplit(image_name,'\');
tmp=size(names)
tmp=tmp(2)
name=names(tmp)
name=char(name)
box_rows = size(image_info.objectBoundingBoxes,1);
    for j = 1:box_rows%对每张图的所有ROI区域进行提取
        box =image_info.objectBoundingBoxes(j,:);
        %cropped_roi = imcrop(image,box);
        %imshow(cropped_roi);
        image_full_name = strcat(output_path,'\',name);
        fprintf(fid,'%s\t%d\t%d\t%d\t%d\n',image_full_name,box(1),box(2),box(1)+box(3),box(2)+box(4));
        %imwrite(cropped_roi,image_full_name);
        roi_index=roi_index + 1;
    end
end
fclose(fid);