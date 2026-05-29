%% Load chuncked data 
clear; close all

% load('/Users/assiachericoni/Documents/MATLAB/codes/PacMan/CCGP/data/chunckForCCGP_40s_6BESTbinsPerClassGAZE.mat')

load("/Users/assiachericoni/Documents/MATLAB/codes/PacManRepo-hippocampus/data/dataCCGP.mat")

ySelf = dataCCGP.ySelf;
yPrey = dataCCGP.yPrey;
yUnprey = dataCCGP.yUnprey;

SelfItems = dataCCGP.SelfItems;
PreyItems = dataCCGP.PreyItems;
UnpreyItems = dataCCGP.UnpreyItems;

%% 1. Get decoding axes for Self, Chosen Prey, Unchosen Prey

w_S = get_decoding_axis(SelfItems,   ySelf);
w_P = get_decoding_axis(PreyItems,   yPrey);
w_U = get_decoding_axis(UnpreyItems, yUnprey);

uS = w_S / norm(w_S);
uP = w_P / norm(w_P);
uU = w_U / norm(w_U);

cos_SP = dot(uS, uP);
cos_SU = dot(uS, uU);
cos_PU = dot(uP, uU);

theta_SP = acosd(cos_SP);
theta_SU = acosd(cos_SU);
theta_PU = acosd(cos_PU);

% decoding axis 
cosM = [ 1      cos_SP  cos_SU;
    cos_SP 1       cos_PU;
    cos_SU cos_PU  1     ];

labels = {'Self','Chosen','Unchosen'};

D = sqrt( 2*(1 - cosM) );   % 4x4 symmetric distance matrix

[Yc, eigvals] = cmdscale(D);   % Y is 4 x 4, but only first 2 cols matter

coords2D = Yc(:,1:2);   

% center

[coeff, ~] = pca(Yc);
Y_rot = Yc * coeff;
figure; hold on; axis equal;
scatter(coords2D(:,1), coords2D(:,2), 80, 'filled');
text(coords2D(:,1)+0.02, coords2D(:,2), labels);



% %% 2. Get decoding axes for Self, Chosen Prey, Unchosen Prey and gaze
% 
% w_S = get_decoding_axis(SelfItems,   ySelf);
% w_P = get_decoding_axis(PreyItems,   yPrey);
% w_U = get_decoding_axis(UnpreyItems, yUnprey);
% 
% uS = w_S / norm(w_S);
% uP = w_P / norm(w_P);
% uU = w_U / norm(w_U);
% 
% cos_SP = dot(uS, uP);
% cos_SU = dot(uS, uU);
% cos_PU = dot(uP, uU);
% 
% theta_SP = acosd(cos_SP);
% theta_SU = acosd(cos_SU);
% theta_PU = acosd(cos_PU);
% 
% % decoding axis for gaze
% w_G = get_decoding_axis(GazeItems, yGaze);
% uG = w_G / norm(w_G);
% 
% % cosines (dot products) between all pairs
% cos_SG = dot(uS, uG);
% cos_PG = dot(uP, uG);
% cos_UG = dot(uU, uG);
% 
% cosM = [ 1      cos_SP  cos_SU  cos_SG;
%          cos_SP 1       cos_PU  cos_PG;
%          cos_SU cos_PU  1       cos_UG;
%          cos_SG cos_PG  cos_UG  1      ];
% 
% labels = {'Self','Chosen','Unchosen','Gaze'};
% 
% D = sqrt( 2*(1 - cosM) );   % 4x4 symmetric distance matrix
% 
% [Y, eigvals] = cmdscale(D);   % Y is 4 x 4, but only first 2 cols matter
% 
% coords2D = Y(:,1:2);   
% 
% % center
% figure; hold on; axis equal;
% scatter(-coords2D(:,1), coords2D(:,2), 80, 'filled');
% text(-coords2D(:,1)+0.02, coords2D(:,2), labels);
% set(gca,'TickDir','out', 'Color', 'None', 'box','off','Fontname','Helvetica', 'FontSize', 12, 'TitleFontWeight' , 'normal');
% xlabel('MDS axis 1'); ylabel('MDS axis 2')
% 
% fprintf('Self–Chosen:   cos = %.3f, angle = %.1f deg\n', cos_SP, theta_SP);
% fprintf('Self–Unchosen: cos = %.3f, angle = %.1f deg\n', cos_SU, theta_SU);
% fprintf('Chosen–Unchosen: cos = %.3f, angle = %.1f deg\n', cos_PU, theta_PU);
% 
% theta_SG = acosd(cos_SG);
% theta_PG = acosd(cos_PG);
% theta_UG = acosd(cos_UG);
% 
% fprintf('Self–Gaze:     cos = %.3f, angle = %.1f deg\n', cos_SG, theta_SG);
% fprintf('Chosen–Gaze:   cos = %.3f, angle = %.1f deg\n', cos_PG, theta_PG);
% fprintf('Unchosen–Gaze: cos = %.3f, angle = %.1f deg\n', cos_UG, theta_UG);

