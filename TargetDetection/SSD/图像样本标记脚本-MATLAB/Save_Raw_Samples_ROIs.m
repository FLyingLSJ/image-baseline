
%����˵������ȡͨ��MATLAB Training Image��ǲ������labelingSession���ݣ���
%��Щ��ǵ�ROI����ת����bmpͼƬ���洢��ͼƬ�����Ʋ�����ʱ��+��ŵĴ洢��ʽ��
%���Զ�ζ�ȡ��һ���������������ͬһ�ļ���Ҳ�������������ͻ
%ʾ�����÷�ʽ Save_Raw_Samples_ROIs('E:\TSY\Code\labelingSession.mat','G:\Samples\2016-3-5');
%**************************************************************************
%��һ��������ͨ��MATLAB Training Image
%Labeler���ߵ�����labelingSession���ݣ�Ϊmat��ʽ������ֻ��������ȫ·�����ɣ���'C:\labelingSession.mat'
%�ڶ���������Ϊ���������ļ��洢���ļ��У���'G:\Samples\2016-3-1'
%���б�ǹ���ͼ���0��ʼ����Ϊ�ļ������д洢
function Save_Raw_Samples_ROIs(labelingsession_data_path,output_path)
roi_info = load(labelingsession_data_path);
image_counts = size(roi_info.labelingSession.ImageSet.ROIBoundingBoxes,2);%struct�ṹ����ͨ�����������ʽ��ͬ����һ��Ϊ�У��ڶ���Ϊ��
roi_index = 0;
pause(1);%��ʱ1s��ȷ���������������
%current_time = datestr(now,'yyyy-mm-dd-HH-MM-SS_');
if ~isdir(output_path) %�ж�·���Ƿ����
    mkdir(output_path);
end
fid = fopen('D:\\code\\fasterrcnn\\output_img_label\\pos.txt','wt');
for i = 1:image_counts%��ÿ��ͼ���δ�����ȡROI����
image_info = roi_info.labelingSession.ImageSet.ROIBoundingBoxes(i);
image_name = image_info.imageFilename;
names=strsplit(image_name,'\');
tmp=size(names)
tmp=tmp(2)
name=names(tmp)
name=char(name)
box_rows = size(image_info.objectBoundingBoxes,1);
    for j = 1:box_rows%��ÿ��ͼ������ROI���������ȡ
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