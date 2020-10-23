close all;
clear;
clc;

addpath('data');
addpath('utility');

load('3sources.mat');
n = size(truelabel{1}, 2);
nv = size(data, 2);
K = length(unique(truelabel{1}));
gnd = truelabel{1};

tsources_data_views = cell(1, nv);
new_tsources_data_views = cell(1, nv);

for nv_idx = 1 : nv
     tsources_data_views{nv_idx} = data{nv_idx};
end

for nv_idx = 1 : nv 
    for idx = 1 : n
        sample = tsources_data_views{nv_idx}(:, idx);
        tsources_data_views{nv_idx}(:, idx) = sample ./ max(1e-12, norm(sample));
    end   
    new_dim = 160;
    [eigen_vector, ~] = f_pca(tsources_data_views{nv_idx}, new_dim);
    new_tsources_data_views{nv_idx} = eigen_vector' *  tsources_data_views{nv_idx};
end

lambdas = [0.5];
etas = [0.1];
alphas = [5];
    
for lmd_idx = 1 : length(lambdas)
    lambda = lambdas(lmd_idx);
     for eta_idx = 1 : length(etas)
        eta = etas(eta_idx);
        Zs = zeros(n, n);
        [Zn, iter] = mlrr(new_tsources_data_views, lambda, eta);     
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
            dlmwrite('mlrr_tsources_data.txt', [lambda, eta, alpha, acc, nmi, purity, fmeasure, ri, ari] , '-append', 'delimiter', '\t', 'newline', 'pc');
        end
    end
end

