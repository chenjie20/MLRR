close all;
clear;
clc;

addpath('data');
addpath('utility');

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
            [acc, nmi, purity, fmeasure, ri, ari] = mlrr_calculation(L, gnd, K);
            disp([lambda, eta, alpha, acc, nmi, purity, fmeasure, ri, ari]);            
            dlmwrite('mlrr_uci_data.txt', [lambda, eta, alpha, acc, nmi, purity, fmeasure, ri, ari] , '-append', 'delimiter', '\t', 'newline', 'pc');
        end            
    end
end
