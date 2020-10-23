close all;
clear;
clc;

addpath('data');
addpath('utility');

load('flower17_Kmatrix.mat');
n = length(Y);
nv = size(KH, 3);
K = length(unique(Y));
gnd = Y;

flower_data_views = cell(1, nv);
new_flower_data_views = cell(1, nv);

for nv_idx = 1 : nv
     flower_data_views{nv_idx} = KH(:, :, nv_idx)';
end

for nv_idx = 1 : nv 
%     for idx = 1 : n
%         sample = flower_data_views{nv_idx}(:, idx);
%         flower_data_views{nv_idx}(:, idx) = sample ./ max(1e-12, norm(sample));
%     end    
    new_dim = 100;
    [eigen_vector, ~] = f_pca(flower_data_views{nv_idx}, new_dim);
    new_flower_data_views{nv_idx} = eigen_vector' *  flower_data_views{nv_idx};
end

lambdas = [600];
etas = [0.1];
alphas = [1];

for lmd_idx = 1 : length(lambdas)
    lambda = lambdas(lmd_idx);
     for eta_idx = 1 : length(etas)
        eta = etas(eta_idx);
        
        Zs = zeros(n, n);
        [Zn, iter] = mlrr(new_flower_data_views, lambda, eta, K);     
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
            dlmwrite('mlrr_flower_data.txt', [lambda, eta, alpha, acc, nmi, purity, fmeasure, ri, ari] , '-append', 'delimiter', '\t', 'newline', 'pc');           
        end
            
    end
end

