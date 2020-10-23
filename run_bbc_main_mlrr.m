close all;
clear;
clc;

addpath('data');
addpath('utility');

load('BBC4view_685.mat');
n = size(truelabel{1}, 2);
nv = size(data, 2);
K = length(unique(truelabel{1}));
gnd = truelabel{1};

BBC_data_views = cell(1, nv);
new_BBC_data_views = cell(1, nv);

for nv_idx = 1 : nv
     BBC_data_views{nv_idx} = data{nv_idx};
end

for nv_idx = 1 : nv 
    for idx = 1 : n
        sample = BBC_data_views{nv_idx}(:, idx);
        BBC_data_views{nv_idx}(:, idx) = sample ./ max(1e-12, norm(sample));
    end    
    new_dim = 600;
    [eigen_vector, ~] = f_pca(BBC_data_views{nv_idx}, new_dim);
    new_BBC_data_views{nv_idx} = eigen_vector' *  BBC_data_views{nv_idx};
end
   
lambdas = [0.05];
etas = [0.1];
alphas = [1];

for lmd_idx = 1 : length(lambdas)
    lambda = lambdas(lmd_idx);
     for eta_idx = 1 : length(etas)
        eta = etas(eta_idx);
        Zs = zeros(n, n);
        [Zn, iter] = mlrr(new_BBC_data_views, lambda, eta);     
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
            [acc, nmi, purity, fmeasure, ri, ari] = mlrr_calculation(L, gnd, K);            
            disp([lambda, eta, alpha, acc, nmi, purity, fmeasure, ri, ari]);
            dlmwrite('mlrr_bbc_data.txt', [lambda, eta, alpha, acc, nmi, purity, fmeasure, ri, ari] , '-append', 'delimiter', '\t', 'newline', 'pc');
        end
    end
end

