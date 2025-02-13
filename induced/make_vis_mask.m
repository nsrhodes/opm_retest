close all
clear
clc

% Run the study configuration
p = opm_study_config_gamma();%rest_babyeinstein_inscapes();%;%opm_study_config_gamma();

% Overwrite?
clobber = 1;

%% Go through subjects
for ss = 1:size(p.subject_data, 1)
    % check if exists as only need once per subject
    if ~exist([p.directories.anat_dir(ss) '/vismask_5mm.nii.gz']) || clobber
        grid = ft_read_mri([p.directories.anat_dir(ss) '/AAL90_5mm.nii.gz']);

        VOI = (grid.anatomy == 45) | (grid.anatomy == 46);

        SE = strel('sphere',5);
        dilatedVOI = imdilate(VOI,SE) & (grid.anatomy > 0);

        vismask = grid;
        vismask.anatomy = double(dilatedVOI);

        ft_write_mri([p.directories.anat_dir(ss) '/vismask_5mm.nii.gz'],vismask,'dataformat','nifti_gz')
    end
end 