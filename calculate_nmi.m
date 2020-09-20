function [result] = calculate_nmi(class_data, cluster_data)

    if(size(class_data, 2) == 1)
        class_data = class_data';
    end
    if(size(cluster_data, 2) == 1)
        cluster_data = cluster_data';
    end
    num_class = size(class_data, 2);
    num_cluster = size(cluster_data, 2);
    n = 0;
    m = 0;
    for idx = 1 : num_class
        data = class_data(idx);
        m = m + size(data{1}, 2);
    end
    for idx = 1 : num_cluster
        data = cluster_data(idx);
        n = n + size(data{1}, 2);
    end   
    
    ho = 0;
    for idx = 1 : num_class
        data = class_data(idx);
        num =  size(data{1}, 2);
        rs = num / m;
        ho = ho - rs * log2(rs);
    end
    
    ha = 0;
    hao = 0;
    for i = 1 : num_cluster
        current_cluster = cluster_data(i);
        num_i = size(current_cluster{1}, 2);
        rs = num_i / n;
        ha = ha - rs * log2(rs);
 
        hao_tmp = 0;
        for j = 1 : num_class                       
            num_j = length(find(current_cluster{1}(1, :) == j));            
            rs_tmp = num_j / num_i;
            if rs_tmp > 1e-6
                hao_tmp = hao_tmp + rs_tmp * log2(rs_tmp);
            end
        end
        hao = hao - rs * hao_tmp;        
    end
    result = 2 * (ho - hao) / (ho + ha);  
%     disp([ho, ha, hao]);
    
end

