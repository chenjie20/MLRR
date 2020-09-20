close all;
clear;
clc;

addpath('data');
load('uci_digit.mat');
gnd = uci_labels;
K = max(gnd); 

uci_data_views = cell(1, 3);
n = size(uci_fourier_data, 2);
uci_data_views{1} = uci_fourier_data;
X2 = uci_profile_data;
for i = 1 : n
    X2(:, i) = X2(:, i) ./ max(1e-12,norm(X2(:, i)));
end 
uci_data_views{2} = X2;
X3 = uci_kar_data;
for i = 1 : size(X3, 2)
    X3(:, i) = X3(:, i) ./ max(1e-12,norm(X3(:, i)));
end
uci_data_views{3} = X3;
nv = length(uci_data_views);

rand_enable = 0; % 0 false 1 true
if rand_enable ~= 0
    rand('state', 10200);
    rnd_idx = randperm(n);
    gnd = gnd(rnd_idx);
    for idx = 1 : nv
        uci_data_views(idx) = { uci_data_views{idx}(:, rnd_idx) };
    end
end

cluster_data = cell(1, K);
class_labels = zeros(1, K);
for idx =  1 : K
    class_labels(idx) = 200;
end

lambdas = [9e-2];
etas = [9e-3];
alphas = [5];

for lmd_idx = 1 : length(lambdas)
    lambda = lambdas(lmd_idx);
     for eta_idx = 1 : length(etas)
        eta = etas(eta_idx);
        Zs = zeros(n, n);
        [Zn, iter] = mlrr(uci_data_views, lambda, eta);     
        for idx = 1 : nv 
            Zs = Zs + Zn{idx}; 
        end
        [U s V] = svd(Zs);
        s = diag(s);
        r = sum(s>1e-6);

        U = U(:, 1 : r);
        s = diag(s(1 : r));
        V = V(:, 1 : r);

        M = U * s.^(1/2);
        mm = normr(M);
        rs = mm * mm';

        for alpha_idx = 1: length(alphas)
            alpha = alphas(alpha_idx);
            L = rs.^(2 * alpha);
            actual_ids = spectral_clustering(L, K);
            acc = 1 - calculate_err(gnd, actual_ids); 
            
            if(size(actual_ids, 2) == 1)
                actual_ids = actual_ids';
            end
            for idx =  1 : K
                 cluster_data(1, idx) = { gnd(actual_ids(1, :) == idx) };
            end
            [nmi, fmeasure, ri, ari] = calculate_results(class_labels, cluster_data);
            disp([rand_enable lambda eta alpha acc nmi fmeasure ri ari]);
            dlmwrite('uci_mlrr.txt', [rand_enable lambda eta alpha acc nmi fmeasure ri ari] , '-append', 'delimiter', '\t', 'newline', 'pc');
        end
    end
end
