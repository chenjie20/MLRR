close all;
clear;
clc;

addpath('data');
addpath('utility');

load('proteinFold_Kmatrix.mat');
n = length(Y);
nv = size(KH, 3);
K = length(unique(Y));
gnd = Y;

protein_data_views = cell(1, nv);
new_protein_data_views = cell(1, nv);

for nv_idx = 1 : nv
     protein_data_views{nv_idx} = KH(:, :, nv_idx)';
end

for nv_idx = 1 : nv 
    for idx = 1 : n
        sample = protein_data_views{nv_idx}(:, idx);
        protein_data_views{nv_idx}(:, idx) = sample ./ max(1e-12, norm(sample));
    end    
    new_dim = 100;
    [eigen_vector, ~] = f_pca(protein_data_views{nv_idx}, new_dim);
    new_protein_data_views{nv_idx} = eigen_vector' *  protein_data_views{nv_idx};
end

lambdas = [50];
etas = [0.01];
alphas = [2];

for lmd_idx = 1 : length(lambdas)
    lambda = lambdas(lmd_idx);
     for eta_idx = 1 : length(etas)
        eta = etas(eta_idx);
        Zs = zeros(n, n);
        [Zn, iter] = mlrr(new_protein_data_views, lambda, eta, K);     
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
            dlmwrite('protein_mlrr_data.txt', [lambda, eta, alpha, acc, nmi, purity, fmeasure, ri, ari] , '-append', 'delimiter', '\t', 'newline', 'pc');
        end          
    end
end
