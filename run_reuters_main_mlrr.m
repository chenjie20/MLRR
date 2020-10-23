close all;
clear;
clc;

addpath('data');
addpath('utility');

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

lambdas = [0.8];
etas = [5e-3];
alphas = [2];

for lmd_idx = 1 : length(lambdas)
    lambda = lambdas(lmd_idx);
     for eta_idx = 1 : length(etas)
        eta = etas(eta_idx);
        Zs = zeros(total_num, total_num);
        tic;
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
        cost = toc;

        for alpha_idx = 1: length(alphas)
            alpha = alphas(alpha_idx);
            L = rs.^(2 * alpha);
            [acc, nmi, purity, fmeasure, ri, ari] = mlrr_calculation(L, gnd, K);            
            disp([lambda, eta, alpha, acc, nmi, purity, fmeasure, ri, ari]);
            dlmwrite('mlrr_reuters_data.txt', [lambda, eta, alpha, acc, nmi, purity, fmeasure, ri, ari] , '-append', 'delimiter', '\t', 'newline', 'pc');
        end
    end
end
