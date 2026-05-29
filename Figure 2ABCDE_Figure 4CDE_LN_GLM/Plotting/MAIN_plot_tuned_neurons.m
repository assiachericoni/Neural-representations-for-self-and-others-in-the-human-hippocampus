%% Load tuning curves 
clear; close all
load('/Users/assiachericoni/Documents/MATLAB/data/PacMan/CumulativeData13/tuningCurves.mat')
load('/Users/assiachericoni/Documents/MATLAB/data/PacMan/CumulativeData13/GLMPosition_Only/sigNeurons.mat')
load('/Users/assiachericoni/Documents/MATLAB/data/PacMan/CumulativeData13/GLMPosition_Only/NS.mat')
%% Filter out non significant neurons 
NSHpc = NStot(strcmp(NStot.brainRegion, 'hpc'),:);

pts = unique(NSHpc.Patient); 
cellIdx = [];
for k = 1:length(pts)

    l = sum(strcmp(NSHpc.Patient, pts{k}));

    vec = 1:l; 

    cellIdx = [cellIdx, vec];
end

cellIdx = cellIdx';

tuningIdx = logical(sum(sigNeurons == [1 4 5 7], 2)); 
% ptsIdx = NSHpc.Patient(tuningIdx);
cellCount = cellIdx(tuningIdx);
ptCount = ptNums(tuningIdx);

Self = SelfMaps(tuningIdx);
Prey = ChosenPreyMaps(tuningIdx);
Unprey = UnchosenPreyMaps(tuningIdx);

iN = find(cellCount == 4 & ptCount == 2)

dt = 1/60;
for iN = 75:80

SelfMap = Self{iN}/dt;
PreyMap = Prey{iN}/dt;
UnpreyMap = Unprey{iN}/dt;

stp_sze = 0.01; % level of smoothness for spatial maps - smaller value means more fine
[outSelfMap] = smooth_maps(SelfMap, stp_sze);

% figureHandle = figure('Position', [100, 100, 1200, 600]);
figure;
% subplot(1,3,1)
imagesc(outSelfMap); axis off;
% set(gca, 'DataAspectRatio', [1 1 1]);
set(gca,'tickDir','out','color','none','linewidth',1,'fontsize',12); box off;
colorbar; title('Self Pos Map');
xlabel('Screen X'); ylabel('Screen Y');
end
[outPreyMap] = smooth_maps(PreyMap, stp_sze);
% subplot(1,3,2)
figure
imagesc(outPreyMap); axis off;
% set(gca, 'DataAspectRatio', [1 1 1]);
set(gca,'tickDir','out','color','none','linewidth',1,'fontsize',12); box off;
colorbar; title('Prey Pos Map');
xlabel('Screen X'); ylabel('Screen Y');

[outUnpreyMap] = smooth_maps(UnpreyMap, stp_sze);
% subplot(1,3,3)
figure
imagesc(outUnpreyMap); axis off;
% set(gca, 'DataAspectRatio', [1 1 1]);
set(gca,'tickDir','out','color','none','linewidth',1,'fontsize',12); box off;
colorbar; title('Unchosen Prey Pos Map');
xlabel('Screen X'); ylabel('Screen Y');
sgtitle(['Cell ' num2str(cellCount(iN)) ' subj ' ptsIdx{iN}])

end

%% 3d plot

iN = 42;

SelfMap = Self{iN}/dt;
PreyMap = Prey{iN}/dt;
UnpreyMap = Unprey{iN}/dt;

stp_sze = 0.01; % level of smoothness for spatial maps - smaller value means more fine
[outSelfMap] = smooth_maps(SelfMap, stp_sze);
figure; s = surf(outSelfMap, 'FaceAlpha',0.85, 'EdgeColor','none');
grid off;
set(gca,'tickDir','out','color','none','linewidth',1,'fontsize',12); box off;
colorbar; title('Self Pos Map');
xlabel('Screen X'); ylabel('Screen Y');
view(-16.16, 58.8);
%axis off

[outPreyMap] = smooth_maps(PreyMap, stp_sze);
figure; s = surf(outPreyMap, 'FaceAlpha',0.85, 'EdgeColor','none');
grid off;
set(gca,'tickDir','out','color','none','linewidth',1,'fontsize',12); box off;
colorbar; title('Prey Pos Map');
xlabel('Screen X'); ylabel('Screen Y');
view(-16.16, 58.8);
%axis off 

[outUnpreyMap] = smooth_maps(UnpreyMap, stp_sze);
figure; s = surf(outUnpreyMap, 'FaceAlpha',0.85, 'EdgeColor','none');
grid off;
set(gca,'tickDir','out','color','none','linewidth',1,'fontsize',12); box off;
colorbar; title('Unchosen Prey Pos Map');
xlabel('Screen X'); ylabel('Screen Y');
view(-16.16, 58.8);
%axis off 

% colormap parula; % Or 'parula', 'hot', etc.
% grid off;
% colorbar; % 
% shading interp;
% lighting phong; % Or 'gouraud' for a smoother look
% s.EdgeColor = 'none';
% xlabel('X-axis');
% ylabel('Y-axis');
% zlabel('Firing Rate');

%%
iN = find(ptNums == 8 & cellIdx == 19)

sigNeurons(iN)






