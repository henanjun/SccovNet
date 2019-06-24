clear;clc;

data_set_name =  'NWPU45'; % UCM21; AID30;WHU19;NWPU45;
%img_type =  ['*.jpg','*.png'];% UCM21='.tif', AID30='.jpg';WHU19='.jpg';NWPU45='.jpg';
data_dir = 'data';
subfolders = dir(data_set_name);
count1 = 1; 
count2 = 1;
% begin read the name of  all images
for ii = 1:length(subfolders)
    subname = subfolders(ii).name;
    if ~strcmp(subname, '.') & ~strcmp(subname, '..')
        frames = dir(fullfile(data_set_name, subname));
        
        c_num = length(frames);
        class_name{count2} = subname;
        count2 = count2+1;
        for jj= 1:c_num
            frames_subname = frames(jj).name;
            if ~strcmp(frames_subname, '.') & ~strcmp(frames_subname, '..')
                allsamples_name{count1} = fullfile(data_dir,data_set_name, subname, frames(jj).name);
                count1 = count1+1;
            end
        end
     end
end
imdb_all.allsamples_name = allsamples_name;
imdb_all.class_name = class_name;
name = ['imdb_all',data_set_name];
save(name,'imdb_all');