function [Zn, iter] = mlrr(data_views, lambda, eta)

    maxIter = 500;
    rho = 1.1;
    max_mu = 1e10;
    mu = 1e-2;
    tol = 1e-6;

    nv = length(data_views);
    Jn = cell(1, nv);
    Zn = cell(1, nv);
    Yn1 = cell(1, nv);
    Yn2 = cell(1, nv);
    Xtn = cell(1, nv);
    invXn = cell(1, nv);
    min_values = zeros(1, nv);
    min_1_values = zeros(1, nv);
    min_2_values = zeros(1, nv);
    
    % intialize
    for idx = 1 : nv
        [m, n] = size(data_views{idx});
        Jn{idx} = zeros(n, n);
        En{idx} = sparse(m, n);
        Zn{idx} = zeros(n, n);

        Yn1{idx} = zeros(m, n);
        Yn2{idx} = zeros(n, n);
        
        Xtn{idx} = data_views{idx}' * data_views{idx};
        invXn{idx} = inv(Xtn{idx} + (1 + (eta / mu) * (nv - 1) * eye(n)));
    end

    iter = 0;
    while iter < maxIter  
        iter = iter + 1;
        for idx = 1 : nv
            data_view = data_views{idx};
            [m, n] = size(data_view);
            % update Jn{idx}
            temp = Zn{idx} + Yn2{idx}/mu;
            temp1 = (temp + temp') / 2;
            [U s V] = svd(temp1);
            s = diag(s);
            r = sum(s > 1/mu); 
            Jn{idx} = U(:, 1 : r) * diag(s(1 : r) - 1/mu) * V(:, 1 : r)';
            
            % update Zn{idx}
            Zn_sum = zeros(size(Zn{idx}));
            for i = 1 : nv
                 if i ~= idx
                     Zn_sum = Zn_sum + Zn{i};
                 end
            end
            Zn{idx} =  invXn{idx} * (Xtn{idx} - data_view' * En{idx} + Jn{idx} + (data_view' * Yn1{idx} - Yn2{idx} + eta * Zn_sum)/mu);
             
            %  update En{idx}           
            xmaz = data_view - data_view * Zn{idx};
            tmp = xmaz + Yn1{idx}/mu;
            deta = lambda / mu;
            for i = 1 : n
                nw = norm(tmp(:, i));
                if deta < nw
                    En{idx}(:,i) = (nw - deta) * tmp(:, i) / nw;
                else
                    En{idx}(:,i)= zeros(length(tmp(:, i)),1);
                end
            end 
            leq1 = xmaz - En{idx};
            leq2 = Zn{idx} - Jn{idx};
            Yn1{idx} = Yn1{idx} + mu * leq1;
            Yn2{idx} = Yn2{idx} + mu * leq2;                       
        end
        mu = min(max_mu, mu * rho);
        for idx = 1 : nv                
            leq1 = data_views{idx} - data_views{idx} * Zn{idx} - En{idx};
            leq2 = Zn{idx} - Jn{idx};
            min_1_values(idx) = max(max(abs(leq1)));
            min_2_values(idx) = max(max(abs(leq2)));
            min_values(idx) = max(min_1_values(idx), min_2_values(idx));                      
        end
        finish = 0;
        for idx = 1 : nv 
            if min_values >= tol 
                finish = 1;
                break;
            end
        end
        if (iter == 1 || mod(iter, 50) == 0 || finish == 0)
            for idx = 1 : nv 
                disp(['iter ' num2str(iter) ', nv=' num2str(idx) ', mu=' num2str(mu, '%2.1e') ...
                     ', rank=' num2str(rank(Zn{idx},1e-3 * norm(Zn{idx},2))) ', min value=' num2str(min_values(idx),'%2.3e') ...
                     ', min_1=' num2str(min_1_values(idx), '%2.3e') ', min_2=' num2str(min_2_values(idx), '%2.3e')]);
            end
        end
        if finish == 0
            disp('mlrr done.');
            break;
        end
    end
end