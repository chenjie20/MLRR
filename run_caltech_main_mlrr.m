close all;
clear;
clc;

addpath('data');
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
%     new_reuters_data_views{new_ids(nv_idx)} = eigen_vector' *  reuters_data_views{nv_idx};
end
   
cluster_data = cell(1, K);
class_labels = zeros(1, K);
for idx =  1 : K
    class_labels(idx) = length(find(Y == idx));
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
            actual_ids = spectral_clustering(L, K);
            acc = 1 - calculate_err(gnd, actual_ids); 
            
            if(size(actual_ids, 2) == 1)
                actual_ids = actual_ids';
            end
            for idx =  1 : K
                tmp = { gnd(actual_ids(1, :) == idx) };
                if (size(tmp, 2) == 1)
                    cluster_data(1, idx) = tmp';
                else
                    cluster_data(1, idx) = tmp;
                end
            end
            [nmi, fmeasure, ri, ari] = calculate_results(class_labels, cluster_data);
            disp([lambda eta alpha acc nmi fmeasure ri ari]);
            dlmwrite('caltech_mlrr.txt', [lambda eta alpha acc nmi fmeasure ri ari] , '-append', 'delimiter', '\t', 'newline', 'pc');
        end
    end
end
