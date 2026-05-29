%% Load chuncked data 
clear; close all
load('chunckForCCGP_40s_6BESTbinsPerClass.mat')

%% 2. Get decoding axes for Self, Chosen Prey, Unchosen Prey

w_S = get_decoding_axis(SelfItems,   ySelf);
w_P = get_decoding_axis(PreyItems,   yPrey);
w_U = get_decoding_axis(UnpreyItems, yUnprey);

% normalize to unit length
uS = w_S / norm(w_S);
uP = w_P / norm(w_P);
uU = w_U / norm(w_U);


%% 3. Cosines and angles between decoding axes (in neuron space)

cos_SP = dot(uS, uP);
cos_SU = dot(uS, uU);
cos_PU = dot(uP, uU);

theta_SP = acosd(cos_SP);
theta_SU = acosd(cos_SU);
theta_PU = acosd(cos_PU);

fprintf('Angles between SVM decoding axes (neuron space):\n');
fprintf('Self vs Chosen:   cos = %.3f, angle = %.1f°\n', cos_SP, theta_SP);
fprintf('Self vs Unchosen: cos = %.3f, angle = %.1f°\n', cos_SU, theta_SU);
fprintf('Chosen vs Unch.:  cos = %.3f, angle = %.1f°\n', cos_PU, theta_PU);

%%
% uS, uP, uU are unit vectors in neuron space: Nneurons x 1
uS = uS(:);
uP = uP(:);
uU = uU(:);

U = [uS.'; uP.'; uU.'];   % 3 x Nneurons
G = U * U.';              % 3 x 3 Gram matrix, G_ij = dot(u_i, u_j)

% sanity check: these are exactly your cosines
disp('Gram matrix (cosines between decoding axes):');
disp(G);
[V, D] = eig(G);           % G = V * D * V'
[evals, order] = sort(diag(D), 'descend');
V = V(:, order);
D = diag(evals);

% coordinates in an isometric embedding
coords3 = V * sqrt(D);     % 3 x 3; rows = S, P, U
coords2 = coords3(:,1:2);  % take the top 2 dims -> 3 x 2

coord_S = coords2(1,:);
coord_P = coords2(2,:);
coord_U = coords2(3,:);

coord_S = coord_S / norm(coord_S);
coord_P = coord_P / norm(coord_P);
coord_U = coord_U / norm(coord_U);

phi = atan2(coord_S(2), coord_S(1));
R = [cos(-phi) -sin(-phi); 
     sin(-phi)  cos(-phi)];

coord_Sr = (R * coord_S.').';
coord_Pr = (R * coord_P.').';
coord_Ur = (R * coord_U.').';

cos2_SP = dot(coord_Sr, coord_Pr);
cos2_SU = dot(coord_Sr, coord_Ur);
cos2_PU = dot(coord_Pr, coord_Ur);

theta2_SP = acosd(cos2_SP);
theta2_SU = acosd(cos2_SU);
theta2_PU = acosd(cos2_PU);

fprintf('2D embedded angles (should match neuron-space):\n');
fprintf('Self vs Chosen:   cos=%.3f, angle=%.1f°\n', cos2_SP, theta2_SP);
fprintf('Self vs Unchosen: cos=%.3f, angle=%.1f°\n', cos2_SU, theta2_SU);
fprintf('Chosen vs Unch.:  cos=%.3f, angle=%.1f°\n', cos2_PU, theta2_PU);

figure; hold on; axis equal; box on;

cSelf = [0    0.4470 0.7410];   % blue
cPrey = [0.8500 0.3250 0.0980]; % orange
cUnpr = [0.4940 0.1840 0.5560]; % purple

origin = [0 0];

quiver(origin(1), origin(2), coord_Sr(1), coord_Sr(2), 0, ...
    'Color', cSelf, 'LineWidth', 2, 'MaxHeadSize', 0.5);
quiver(origin(1), origin(2), coord_Pr(1), coord_Pr(2), 0, ...
    'Color', cPrey, 'LineWidth', 2, 'MaxHeadSize', 0.5);
quiver(origin(1), origin(2), coord_Ur(1), coord_Ur(2), 0, ...
    'Color', cUnpr, 'LineWidth', 2, 'MaxHeadSize', 0.5);

th = linspace(0,2*pi,200);
plot(cos(th), sin(th), 'k:');

xlabel('Axis dim 1'); ylabel('Axis dim 2');
title('Decoding axes with preserved angles');
legend({'Self','Chosen prey','Unchosen prey'}, 'Location','bestoutside');
set(gca,'FontSize',12);


%% Load chuncked data 
load('chunckForCCGP_40s_6BESTbinsPerClassGAZE.mat')

%% 2. Get decoding axes for Self, Chosen Prey, Unchosen Prey

w_S = get_decoding_axis(SelfItems,   ySelf);
w_P = get_decoding_axis(PreyItems,   yPrey);
w_U = get_decoding_axis(UnpreyItems, yUnprey);
w_G = get_decoding_axis(GazeItems, yGaze);

% normalize to unit length
uS = w_S / norm(w_S);
uP = w_P / norm(w_P);
uU = w_U / norm(w_U);
uG = w_G / norm(w_G);

%% 3. Cosines and angles between decoding axes (in neuron space)

cos_GP = dot(uG, uP);
cos_GU = dot(uG, uU);
cos_SG = dot(uS, uG);

theta_GP = acosd(cos_GP);
theta_GU = acosd(cos_GU);
theta_SG = acosd(cos_SG);

fprintf('Angles between SVM decoding axes (neuron space):\n');
fprintf('Gaze vs Chosen:   cos = %.3f, angle = %.1f°\n', cos_GP, theta_GP);
fprintf('Gaze vs Unchosen: cos = %.3f, angle = %.1f°\n', cos_GU, theta_GU);
fprintf('Self vs Gaze:  cos = %.3f, angle = %.1f°\n', cos_SG, theta_SG);

%%
% uS, uP, uU are unit vectors in neuron space: Nneurons x 1
uS = uS(:);
uP = uP(:);
uU = uU(:);
uG = uG(:);

U = [uS.'; uP.'; uU.'; uG'];   % 3 x Nneurons
G = U * U.';              % 3 x 3 Gram matrix, G_ij = dot(u_i, u_j)

% sanity check: these are exactly your cosines
disp('Gram matrix (cosines between decoding axes):');
disp(G);
[V, D] = eig(G);           % G = V * D * V'
[evals, order] = sort(diag(D), 'descend');
V = V(:, order);
D = diag(evals);

% coordinates in an isometric embedding
coords3 = V * sqrt(D);     % 3 x 3; rows = S, P, U
coords2 = coords3(:,1:2);  % take the top 2 dims -> 3 x 2

coord_S = coords2(1,:);
coord_P = coords2(2,:);
coord_U = coords2(3,:);
coord_G = coords2(4,:);

coord_S = coord_S / norm(coord_S);
coord_P = coord_P / norm(coord_P);
coord_U = coord_U / norm(coord_U);
coord_G = coord_G / norm(coord_G);

phi = atan2(coord_S(2), coord_S(1));
R = [cos(-phi) -sin(-phi); 
     sin(-phi)  cos(-phi)];

coord_Sr = (R * coord_S.').';
coord_Pr = (R * coord_P.').';
coord_Ur = (R * coord_U.').';
coord_Gr = (R * coord_G.').';

cos2_GP = dot(coord_Gr, coord_Pr);
cos2_GU = dot(coord_Gr, coord_Ur);
cos2_SG = dot(coord_Sr, coord_Gr);

theta2_GP = acosd(cos2_GP);
theta2_GU = acosd(cos2_GU);
theta2_SG = acosd(cos2_SG);

fprintf('2D embedded angles (should match neuron-space):\n');
fprintf('Gaze vs Chosen:   cos=%.3f, angle=%.1f°\n', cos2_GP, theta2_GP);
fprintf('Gaze vs Unchosen: cos=%.3f, angle=%.1f°\n', cos2_GU, theta2_GU);
fprintf('Self vs Gaze:  cos=%.3f, angle=%.1f°\n', cos2_SG, theta2_SG);

figure; hold on; axis equal; box on;

cSelf = [0    0.4470 0.7410];   % blue
cPrey = [0.8500 0.3250 0.0980]; % orange
cUnpr = [0.4940 0.1840 0.5560]; % purple
cGaze = [0.1 0.4 0.5560]; % idk

origin = [0 0];

quiver(origin(1), origin(2), coord_Gr(1), coord_Gr(2), 0, ...
    'Color', cGaze, 'LineWidth', 2, 'MaxHeadSize', 0.5);
quiver(origin(1), origin(2), coord_Ur(1), coord_Ur(2), 0, ...
    'Color', cUnpr, 'LineWidth', 2, 'MaxHeadSize', 0.5);
quiver(origin(1), origin(2), coord_Pr(1), coord_Pr(2), 0, ...
    'Color', cPrey, 'LineWidth', 2, 'MaxHeadSize', 0.5);
quiver(origin(1), origin(2), coord_Sr(1), coord_Sr(2), 0, ...
    'Color', cSelf, 'LineWidth', 2, 'MaxHeadSize', 0.5);

th = linspace(0,2*pi,200);
plot(cos(th), sin(th), 'k:');

xlabel('Axis dim 1'); ylabel('Axis dim 2');
title('Decoding axes with preserved angles');
legend({'Gaze','Chosen prey','Unchosen prey','Self' }, 'Location','bestoutside');
set(gca,'FontSize',12);


