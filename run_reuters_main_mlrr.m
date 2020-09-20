close all;
clear;
clc;

addpath('data');
load('reuters.mat');
K = size(category, 2);
nv = size(X, 2);
reuters_data_views = cell(1, nv);
new_reuters_data_views = cell(1, nv);
num_each_class = 100;
new_dim = 600;
total_num = num_each_class * K;
gnd = zeros(1, total_num);
for nv_idx = 1 : nv 
     dim = size(X{nv_idx}, 2);
     reuters_data_views{nv_idx} = zeros(dim, num_each_class * K);
end
for idx = 1 : K
   view_ids = find(Y == (idx - 1));
   len = length(view_ids);
   rand('state', 2000);
   rnd_idx = randperm(len);
   new_view_ids = view_ids(rnd_idx(1 :num_each_class));
   current_ids = ((idx - 1) * num_each_class + 1) : idx * num_each_class;
   gnd(1, current_ids) = idx;
   for nv_idx = 1 : nv 
       reuters_data_views{nv_idx}(:, current_ids) = X{nv_idx}(new_view_ids, :)';
   end
end

for nv_idx = 1 : nv 
    for idx = 1 : total_num
        sample = reuters_data_views{nv_idx}(:, idx);
        reuters_data_views{nv_idx}(:, idx) = sample ./ max(1e-12, norm(sample));
    end
    
    [eigen_vector, ~] = f_pca(reuters_data_views{nv_idx}, new_dim);
    new_reuters_data_views{nv_idx} = eigen_vector' *  reuters_data_views{nv_idx};
%     new_reuters_data_views{new_ids(nv_idx)} = eigen_vector' *  reuters_data_views{nv_idx};
end
   
rand_enable = 0; % 0 false 1 true
if rand_enable ~= 0
    rand('state', 10200);
    rnd_idx = randperm(total_num);
    gnd = gnd(rnd_idx);
    for idx = 1 : nv
        new_reuters_data_views(idx) = { new_reuters_data_views{idx}(:, rnd_idx) };
    end
end

cluster_data = cell(1, K);
class_labels = zeros(1, K);
for idx =  1 : K
    class_labels(idx) = num_each_class;
end

lambdas = [0.8];
etas = [5e-3];
alphas = [2];

for lmd_idx = 1 : length(lambdas)
    lambda = lambdas(lmd_idx);
     for eta_idx = 1 : length(etas)
        eta = etas(eta_idx);
        Zs = zeros(total_num, total_num);
        [Zn, iter] = mlrr(new_reuters_data_views, lambda, eta);     
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
            dlmwrite('reuters_mlrr.txt', [rand_enable lambda eta alpha acc nmi fmeasure ri ari] , '-append', 'delimiter', '\t', 'newline', 'pc');
        end
    end
end
