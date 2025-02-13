%%  SCRIPT TO EXAMINE TIMESERIES OPM DATA

close all
clear
clc

% Run the study configuration
p = opm_study_config_faces();

%% LOAD DATA

% Extract time

temp = load(p.sourcemodel.grid.ts.ve(p.subject(1), p.session(1), p.run(1),p.task(1)));
time = temp.time';
% Extract the regions
no_trials = [];
% Load in all the timeseries data
ts = zeros(size(temp.pos_coord,1), length(time), size(p.subject_data, 1));
for ss = 1:size(p.subject_data, 1)
    ss
    % Load and store
    VE = load(p.sourcemodel.grid.ts.ve(p.subject(ss), p.session(ss), p.run(ss), p.task(ss)));
    ts_temp = mean(VE.ts, 3);

    % Get left and right fusiform separately
    ts_l = ts_temp(VE.pos_coord_MNI(:,1)<0,:);
    ts_r = ts_temp(VE.pos_coord_MNI(:,1)>0,:);
    sigcorr_r = [];
    sigcorr_l = [];
    %PCA left
    [coeff,~,latent] = pca(ts_l);
    ts_ffg_l(:,ss) = coeff(:,1)';
    for i = 1:size(ts_l);
        try
        tmp = corrcoef(ts_ffg_l(:,ss)',ts_l(i,:));
        sigcorr_l(i) = tmp(1,2);            
        end
    end
    pos_coord_l = VE.pos_coord_MNI(VE.pos_coord_MNI(:,1)<0,:);
    [~,peak] = max(abs(sigcorr_l));
    peak_loc_l(ss,:) = pos_coord_l(peak,:);
    %PCA right
    [coeff,~,latent] = pca(ts_r);
    ts_ffg_r(:,ss) = coeff(:,1)';
    for i = 1:size(ts_r);l
        try
        tmp = corrcoef(ts_ffg_r(:,ss)',ts_r(i,:));
        sigcorr_r(i) = tmp(1,2);
        end
    end
    pos_coord_r = VE.pos_coord_MNI(VE.pos_coord_MNI(:,1)>0,:);
    [~,peak] = max(abs(sigcorr_r));
    peak_loc_r(ss,:) = pos_coord_r(peak,:);

end

%% z-score
% for ss=1:5
%         ts_ffg_zscore_r(:, ss) = (ts_ffg_r(:,ss) - mean(ts_ffg_r(time > -0.1 & time < 0,ss),1))/std(ts_ffg_r(time > -0.1 & time < 0,ss));
% 
% end 
% 
% figure
% hold on;
% plot(time(ind),ts_ffg_zscore_r(ind,1:5),'LineWidth',2,'Color',[cols(1,:) 0.2])
% plot(time(ind),mean(ts_ffg_zscore_r(ind, 1:5), 2), 'Color', cols(1,:), 'LineWidth', 2)
% xlabel('Time (s)');
% ylabel('Amplitude (z-score)');
% xlim([-0.2,0.5]);
%% FIND MOST CORRELATED

% data = ts;
% num_runs = size(data, 3);
% num_voxels = size(data, 1);
% 
%     best_voxels = zeros(num_voxels, num_runs); % Store best match per run
% 
%     for r1 = 1:num_runs
%         for r2 = 1:num_runs
%             if r1 ~= r2
%                 corr_matrix = corr(data(:, :, r1)', data(:, :, r2)'); % Correlate voxels across runs
%                 [~, best_match] = max(corr_matrix, [], 2); % Find best match for each voxel
%                 best_voxels(:, r1) = best_match; % Store best matches
%             end
%         end
%     end
%% PLOTS OF OCCIPITAL REGIONS
% 
% ffg_r = 1; %right fusiform
% ffg_l = 2; %left fusiform
% flip_r = p.subject_data.flip_r;
% flip_l = p.subject_data.flip_l;

%% FLIP

% Extract the ffg timeseries
flip_r = ones(1,25);
flip_l = ones(1,25);
flip_r([11,12,16,21,24]) = 0;
% Iterate through and flip
for ss = 1:size(p.subject_data, 1)
    if flip_r(ss)
        ts_ffg_r(:, ss) = -ts_ffg_r(:, ss);
    end
    if flip_l(ss)
        ts_ffg_l(:,ss) = -ts_ffg_l(:,ss);
    end

end


%% set up for plotting
set_ylim = 0;

twoi = [-0.2, 1.0];
ind = time >= twoi(1) & time <= twoi(2);
cols = lines(7);

%% Group means


fav = figure('Name','Group average response');
fav.Position = [300,500, 600, 600];
ts_ffg_r_mat = reshape(ts_ffg_r,[2520,5,5]); ts_ffg_l_mat = reshape(ts_ffg_l,[2520,5,5]);
ts_ffg_best_mat = cat(2,ts_ffg_r_mat(:,1:3,:),ts_ffg_l_mat(:,4,:),ts_ffg_r_mat(:,5,:));
day_means_best = mean(ts_ffg_r_mat,3);
hold on
plot(time(ind),day_means_best(ind,:),'LineWidth',2,'Color',[cols(7,:) 0.2])
plot(time(ind),mean(day_means_best(ind,:),2), 'Color', cols(7,:), 'LineWidth', 2)
set(gca, 'FontName', 'Arial', 'FontSize', 16)
xlabel('Time (s)');
ylabel('Amplitude (AU)');
axis square
xlim([-0.2,0.5]);
ylim([-0.08 0.05]);
for ss = 1:size(p.subject_data,1)
    toi = [find(time>0.13&time<0.19)];
    time_toi = time(toi);
    ts_temp = ts_ffg_r(toi,ss);
    [amp_r(ss),lat_ind_r(ss)] = min(ts_temp);
    amp_best(ss) = amp_r(ss);
    lat_ind_best(ss) = lat_ind_r(ss);
    ts_temp = ts_ffg_l(toi,ss);
    [amp_l(ss),lat_ind_l(ss)] = min(ts_temp);
    if ss >15 && ss <21
        amp_best(ss) = amp_r(ss);
        lat_ind_best(ss) = lat_ind_r(ss);
    end

end

lat_r = time_toi(lat_ind_r);
lat_l = time_toi(lat_ind_l);
lat_best = time_toi(lat_ind_best);

submean_lat_r = mean(reshape(lat_r,[5,5]),1);
daymean_lat_r = mean(reshape(lat_r,[5,5]),2);
daymean_lat_best = mean(reshape(lat_best,[5,5]),2);

substd_lat_r = std(reshape(lat_r,[5,5]),1);
daystd_lat_r = std(daymean_lat_r);
daystd_lat_best = std(daymean_lat_best);

submean_amp_r = mean(reshape(amp_r,[5,5]),1);
daymean_amp_r = mean(reshape(amp_r,[5,5]),2);
daymean_amp_best = mean(reshape(amp_best,[5,5]),2);

substd_amp_r = std(reshape(amp_r,[5,5]),1);
daystd_amp_r = std(daymean_amp_r);

submean_lat_l = mean(reshape(lat_l,[5,5]),1);
daymean_lat_l = mean(reshape(lat_l,[5,5]),2);

substd_lat_l = std(reshape(lat_l,[5,5]),1);
daystd_lat_l = std(daymean_lat_l);

submean_amp_l = mean(reshape(amp_l,[5,5]),1);
daymean_amp_l = mean(reshape(amp_l,[5,5]),2);

substd_amp_l = std(reshape(amp_l,[5,5]),1);
daystd_amp_l = std(daymean_amp_l);

addpath('/d/mjt/9/projects/OPM-Analysis/OPM_retest/code/Violinplot-Matlab-master/')

fav2 = figure('Name','Group average M170');
fav2.Position = [200,200, 400, 500];
% First subplot: Amplitude (z-score)
subplot(2,1,1)
violinplot(daymean_amp_best,categorical({'Session average'}),'ViolinColor',cols(7,:));
ylabel('Amplitude (AU)');
ylim([-0.15, 0]); xlim([0.6 1.4])
set(gca, 'FontName', 'Arial', 'FontSize', 14)
axis square
% Second subplot: Latency (s)
subplot(2,1,2)
violinplot(daymean_lat_best, categorical({'Session average'}),'ViolinColor',cols(7,:),'Width',0.2);
ylabel('Latency (s)');
ylim([0.14, 0.19]);xlim([0.6 1.4])
set(gca, 'FontName', 'Arial', 'FontSize', 14)
axis square
%% Plot figures sub1
f1 = figure('Name','Sub-001');
f1.Position = [200,200, 400, 900];

subplot(3,1,1)
%title('Right fusiform')
%figure
hold on;
plot(time(ind),ts_ffg_r(ind,1:5),'LineWidth',2,'Color',[cols(1,:) 0.2])
plot(time(ind),mean(ts_ffg_r(ind, 1:5), 2), 'Color', cols(1,:), 'LineWidth', 2)
xlabel('Time (s)');
ylabel('Amplitude (AU)');
xlim([-0.2,0.5]);
ylim([-0.15, 0.08]);
%xline(0.17,'k--','LineWidth',1)
axis square
set(gca, 'FontName', 'Arial', 'FontSize', 16);

subplot(3,1,2)
%title('M170 Amplitude')
hold on;
violinplot( amp_r(1:5)',categorical({'Right fusiform'}),'ViolinColor',cols(1,:));
ylabel('Amplitude (AU)');
set(gca, 'FontName', 'Arial', 'FontSize', 16);
ylim([-0.15, 0]); xlim([0.6 1.4])
axis square

subplot(3,1,3);
%title('M170 Latency');
violinplot(lat_r(1:5), categorical({'Right fusiform'}),'ViolinColor',cols(1,:));
ylabel('Latency (s)');
set(gca, 'FontName', 'Arial', 'FontSize', 16);
ylim([0.14 0.19]);xlim([0.6 1.4])
axis square

%% Plot figures sub2
f2 = figure('Name','Sub-002');
f2.Position = [200,200, 400, 900];

subplot(3,1,1)
%title('Right fusiform')
hold on;
plot(time(ind),ts_ffg_r(ind,6:10),'LineWidth',2,'Color',[cols(2,:) 0.2])
plot(time(ind),mean(ts_ffg_r(ind, 6:10), 2), 'Color', cols(2,:), 'LineWidth', 2)
xlabel('Time (s)');
ylabel('Amplitude (AU)');
xlim([-0.2,0.5]);
ylim([-0.11, 0.08]);
%xline(0.17,'k--','LineWidth',1)
set(gca, 'FontName', 'Arial', 'FontSize', 16);
axis square

subplot(3,1,2)
%title('M170 Amplitude')
hold on;
violinplot( amp_r(6:10)',categorical({'Right fusiform'}),'ViolinColor',cols(2,:));
ylabel('Amplitude (AU)');
set(gca, 'FontName', 'Arial', 'FontSize', 16);
ylim([-0.15, 0]); xlim([0.6 1.4])
axis square

subplot(3,1,3);
%title('M170 Latency');
violinplot(lat_r(6:10), categorical({'Right fusiform'}),'ViolinColor',cols(2,:));
ylabel('Latency (s)');
set(gca, 'FontName', 'Arial', 'FontSize', 16);
ylim([0.14 0.19]);xlim([0.6 1.4])
axis square

%% Plot figures sub3
f3 = figure('Name','Sub-003');
f3.Position = [200,200, 400, 900];

subplot(3,1,1)
%title('Right fusiform')
hold on;
plot(time(ind),ts_ffg_r(ind,11:15),'LineWidth',2,'Color',[cols(3,:) 0.2])
plot(time(ind),mean(ts_ffg_r(ind, 11:15), 2), 'Color', cols(3,:), 'LineWidth', 2)
xlabel('Time (s)');
ylabel('Amplitude (AU)');
xlim([-0.2,0.5]);
ylim([-0.09, 0.08]);
%xline(0.17,'k--','LineWidth',1)
set(gca, 'FontName', 'Arial', 'FontSize', 16);
axis square

subplot(3,1,2)
%title('M170 Amplitude')
hold on;
violinplot( amp_r(11:15)',categorical({'Right fusiform'}),'ViolinColor',cols(3,:));
ylabel('Amplitude (AU)');
set(gca, 'FontName', 'Arial', 'FontSize', 16);
ylim([-0.15, 0]); xlim([0.6 1.4])
axis square

subplot(3,1,3);
%title('M170 Latency');
violinplot(lat_r(11:15), categorical({'Right fusiform'}),'ViolinColor',cols(3,:));
ylabel('Latency (s)');
set(gca, 'FontName', 'Arial', 'FontSize', 16);
ylim([0.14 0.19]);xlim([0.6 1.4])
axis square

%% Plot figures sub4
f4 = figure('Name','Sub-004');
f4.Position = [200,200, 400, 900];

subplot(3,1,1)
%title('Left fusiform')
hold on;
plot(time(ind),ts_ffg_l(ind,16:20),'LineWidth',2,'Color',[cols(4,:) 0.2])
plot(time(ind),mean(ts_ffg_l(ind, 16:20), 2), 'Color', cols(4,:), 'LineWidth', 2)
xlabel('Time (s)');
ylabel('Amplitude (AU)');
xlim([-0.2,0.5]);
ylim([-0.12, 0.08]);
%xline(0.17,'k--','LineWidth',1)
set(gca, 'FontName', 'Arial', 'FontSize', 16);
axis square

subplot(3,1,2)
%title('M170 Amplitude')
hold on;
violinplot( amp_l(16:20)',categorical({'Left fusiform'}),'ViolinColor',cols(4,:));
ylabel('Amplitude (AU)');
set(gca, 'FontName', 'Arial', 'FontSize', 16);
ylim([-0.15, 0]); xlim([0.6 1.4])
axis square

subplot(3,1,3);
%title('M170 Latency');
violinplot(lat_l(16:20), categorical({'Left fusiform'}),'ViolinColor',cols(4,:));
ylabel('Latency (s)');
set(gca, 'FontName', 'Arial', 'FontSize', 16);
ylim([0.14 0.19]);xlim([0.6 1.4])
axis square

%% Plot figures sub5
f5 = figure('Name','Sub-005');
f5.Position = [200,200, 400, 900];

subplot(3,1,1)
%title('Right fusiform')
hold on;
plot(time(ind),ts_ffg_r(ind,21:25),'LineWidth',2,'Color',[cols(5,:) 0.2])
plot(time(ind),mean(ts_ffg_r(ind, 21:25), 2), 'Color', cols(5,:), 'LineWidth', 2)
xlabel('Time (s)');
ylabel('Amplitude (AU)');
xlim([-0.2,0.5]);
ylim([-0.11, 0.09]);
%xline(0.17,'k--','LineWidth',1)
set(gca, 'FontName', 'Arial', 'FontSize', 16);
axis square

subplot(3,1,2)
%title('M170 Amplitude')
hold on;
violinplot( amp_r(21:25)',categorical({'Right fusiform'}),'ViolinColor',cols(5,:));
ylabel('Amplitude (AU)');
set(gca, 'FontName', 'Arial', 'FontSize', 16);
ylim([-0.15, 0]); xlim([0.6 1.4])
axis square

subplot(3,1,3)
%title('M170 Latency');
violinplot(lat_r(21:25), categorical({'Right fusiform'}),'ViolinColor',cols(5,:));
ylabel('Latency (s)');
set(gca, 'FontName', 'Arial', 'FontSize', 16);
ylim([0.14 0.19]);xlim([0.6 1.4])
axis square

%% Plot ellipsoids of peak locations

load('/d/mjt/9/projects/OPM/opm_pipeline_templates/Adult/meshes.mat')

figpeaks = figure('Name','Peak locations');
ft_plot_mesh(meshes(1:2),'facecolor',[.5 .5 .5],'facealpha',.15,'edgecolor','none')
hold on

peak_loc_r(16:20,:) = peak_loc_l(16:20,:);
for sub_ind = 1:5
    % dip_locs = [mean(peak_loc_l((sub_ind*5)-4:sub_ind*5,:)); std(peak_loc_l((sub_ind*5)-4:sub_ind*5,:))]*1e-3;
    % [x(:,:,sub_ind),y(:,:,sub_ind),z(:,:,sub_ind)] = ellipsoid(dip_locs(1,1),dip_locs(1,2),dip_locs(1,3),...
    %     dip_locs(2,1),dip_locs(2,2),dip_locs(2,3));
    % plot3(dip_locs(1,1),dip_locs(1,2),dip_locs(1,3),'.','MarkerFaceColor',cols(sub_ind,:))
    % surf(x(:,:,sub_ind),y(:,:,sub_ind),z(:,:,sub_ind),'EdgeColor','none','FaceColor',cols(sub_ind,:),'FaceAlpha',0.3);
    dip_locs = [mean(peak_loc_r((sub_ind*5)-4:sub_ind*5,:)); std(peak_loc_r((sub_ind*5)-4:sub_ind*5,:))]*1e-3;
    [x(:,:,sub_ind),y(:,:,sub_ind),z(:,:,sub_ind)] = ellipsoid(dip_locs(1,1),dip_locs(1,2),dip_locs(1,3),...
        dip_locs(2,1),dip_locs(2,2),dip_locs(2,3));
    plot3(dip_locs(1,1),dip_locs(1,2),dip_locs(1,3),'.','MarkerFaceColor',cols(sub_ind,:))
    surf(x(:,:,sub_ind),y(:,:,sub_ind),z(:,:,sub_ind),'EdgeColor','none','FaceColor',cols(sub_ind,:),'FaceAlpha',0.3);
  
    %legend({'','','','sub-001','','sub-002','','sub-003','','sub-004','','sub-005',''})
    set(gca, 'FontName', 'Arial', 'FontSize', 16);
end


%% Repeated measures anova
%reshape amplitude data
amp_r_mat = reshape(amp_r,[5,5]);
amp_l_mat = reshape(amp_l,[5,5]);

%reshape latency data
lat_r_mat = reshape(lat_r,[5,5]);
lat_l_mat = reshape(lat_l,[5,5]);

subjects = [1:5]';
table_amp_l = table(subjects,amp_l_mat(1,:)',amp_l_mat(2,:)',amp_l_mat(3,:)',amp_l_mat(4,:)',amp_l_mat(5,:)','VariableNames',{'Subject','Day1','Day2','Day3','Day4','Day5'})
table_amp_r = table(subjects,amp_r_mat(1,:)',amp_r_mat(2,:)',amp_r_mat(3,:)',amp_r_mat(4,:)',amp_r_mat(5,:)','VariableNames',{'Subject','Day1','Day2','Day3','Day4','Day5'})
table_lat_l = table(subjects,lat_l_mat(1,:)',lat_l_mat(2,:)',lat_l_mat(3,:)',lat_l_mat(4,:)',lat_l_mat(5,:)','VariableNames',{'Subject','Day1','Day2','Day3','Day4','Day5'})
table_lat_r = table(subjects,lat_r_mat(1,:)',lat_r_mat(2,:)',lat_r_mat(3,:)',lat_r_mat(4,:)',lat_r_mat(5,:)','VariableNames',{'Subject','Day1','Day2','Day3','Day4','Day5'})

withinDesign = table([1:5]','VariableNames',{'Days'});
rm_amp_l = fitrm(table_amp_l,'Day1-Day5~1','WithinDesign',withinDesign);
rm_amp_r = fitrm(table_amp_r,'Day1-Day5~1','WithinDesign',withinDesign);
rm_lat_l = fitrm(table_lat_l,'Day1-Day5~1','WithinDesign',withinDesign);
rm_lat_r = fitrm(table_lat_r,'Day1-Day5~1','WithinDesign',withinDesign);


ranova_amp_l = ranova(rm_amp_l)
ranova_amp_r = ranova(rm_amp_r)

ranova_lat_l = ranova(rm_lat_l)
ranova_lat_r = ranova(rm_lat_r)

%% Get best amplitudes and latency

amp_best = [amp_r(1:15),amp_l(16:20),amp_r(21:25)]';
lat_best = [lat_r(1:15)',lat_l(16:20)',lat_r(21:25)']';

peak_loc = reshape(peak_loc_r,[5,5,3]);

amp_best_mat = reshape(amp_best,[5,5]);
lat_best_mat = reshape(lat_best,[5,5,]);
