%%  SCRIPT TO EXAMINE TFR OPM DATA

close all
clear
clc

% Run the study configuration
% p = opm_study_config_NEW();
p = cryo_study_config_gamma();

% Overwrite?
clobber = 0;


%% LOAD IN THE VEs and save out just timeseries


% Iterate over the subjects
for ss = 1:size(p.subject_data, 1)
    ss
    temp = load(p.sourcemodel.grid.ts.ve(p.subject(ss), p.session(ss), p.run(ss), p.task(ss)));
    ts_all = temp.ts;
    ts_cat = reshape(ts_all,size(ts_all,1),[]);
    [coeff,~,latent] = pca(ts_cat);
    pca_comp1 = coeff(:,1)';
    ts = reshape(pca_comp1,size(ts_all,2),size(ts_all,3));
    save([p.directories.sub_dir(p.subject(ss), p.session(ss)) '/' p.subject(ss), '_' p.session(ss) '_ts_pca_5mm.mat'],'ts')

end



