close all;
clear;
clc;

addpath('data');
addpath('utility');

load('Caltech101-7.mat');
n = length(Y);
nv = size(X, 2);
K = length(unique(Y));
gnd = Y';

caltech_data_views = cell(1, nv);
new_caltech_data_views = cell(1, nv);

for nv_idx = 1 : nv
     caltech_data_views{nv_idx} = X{nv_idx}';
end

new_dim = 40;
for nv_idx = 1 : nv 
    for idx = 1 : n
        sample = caltech_data_views{nv_idx}(:, idx);
        caltech_data_views{nv_idx}(:, idx) = sample ./ max(1e-12, norm(sample));
    end    
    [eigen_vector, ~] = f_pca(caltech_data_views{nv_idx}, new_dim);
    caltech_data_views{nv_idx} = eigen_vector' *  caltech_data_views{nv_idx};
end

lambdas = [5e-2];
etas = [0.5];
alphas = [4];

for lmd_idx = 1 : length(lambdas)
    lambda = lambdas(lmd_idx);
     for eta_idx = 1 : length(etas)
        eta = etas(eta_idx);
        Zs = zeros(n, n);
        [Zn, iter] = mlrr(caltech_data_views, lambda, eta);     
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
            dlmwrite('caltech_mlrr.txt', [lambda, eta, alpha, acc, nmi, purity, fmeasure, ri, ari] , '-append', 'delimiter', '\t', 'newline', 'pc');
        end
    end
end
