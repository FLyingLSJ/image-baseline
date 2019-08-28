session_folder = 'D:\code\fasterrcnn\input_img_mat';%sessionÄ¿Â¼ 
output_folder = 'D:\code\fasterrcnn\output_img_label';%Êä³öÄ¿Â¼ 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fileFolder=fullfile(session_folder);
dirOutput=dir(fullfile(fileFolder,'*.mat'));
fileNames={dirOutput.name};


for fileName = fileNames
    file_full_name = strcat(session_folder,'\',fileName);
    Save_Raw_Samples_ROIs(char(file_full_name),output_folder);
end

