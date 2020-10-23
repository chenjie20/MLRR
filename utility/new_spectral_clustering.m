function [ ids ] = new_spectral_clustering(W, numClusters)

D = diag(1./sqrt(sum(W, 2)+ eps));
W = D * W * D;
[U, s, V] = svd(W);
V = U(:, 1 : numClusters);
V = normr(V);

% ids = kmeans(V, numClusters, 'emptyaction', 'singleton', 'replicates', 1000, 'display', 'off');
ids = litekmeans(V, numClusters, 'MaxIter',100, 'Replicates', 1000);

end
