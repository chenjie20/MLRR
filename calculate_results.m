function [nmi, fmeasure, ri, ari] = calculate_results(class_labels, cluster_data)
%class_labels: 1 * N vector
%cluster_data: 1 * N cells
    
    if(size(cluster_data, 2) == 1)
        cluster_data = cluster_data';
    end
    num_class = length(class_labels);
    num_cluster = size(cluster_data, 2);    
    n_1 = sum(class_labels);
    n = 0;
    for idx = 1 : num_cluster
        data = cluster_data(idx);
        n = n + size(data{1}, 2);
    end   
    if n_1 ~= n
        error('error: the numbers of sampes are not coincide.');
    end
    
    % nmi
    ho = 0;
    for idx = 1 : num_class
        rs = class_labels(idx) / n;
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
    nmi = 2 * (ho - hao) / (ho + ha);  
%     disp([ho, ha, hao]);
    
    % fmeasure
    fmeasure = 0;
    for i =  1 : num_class       
        fm = 0;
        for j = 1 : num_cluster
            current_cluster = cluster_data(j);            
            num = size(current_cluster{1}, 2);
            len = length(find(current_cluster{1}(1, :) == i));
            precison = len / num;
            recall = len / class_labels(i);
            re = 2 * precison * recall / (precison + recall);
            if re > fm
                fm = re;
            end            
        end   
        fmeasure = fmeasure + class_labels(i) / n * fm;
     end
    
    % ARI
    data = zeros(num_class, num_cluster);
    for i = 1 :num_class
        for j = 1 : num_cluster
            current_cluster = cluster_data(j);  
            data(i, j) = length(find(current_cluster{1}(1, :) == i));
        end
    end
    
    %nchoosek
    
%     rows_sum = sum(data,2);                                                                                                         
%     cols_sum = sum(data,1);                                                                                                         
%     a = 0;
%      for i = 1 :num_class
%         for j = 1 : num_cluster
%             if data(i, j) >= 2
%                 a = a + bincoeff(data(i, j), 2);
%             end
%         end
%      end
%     b0 = 0;
%     for i = 1 : length(rows_sum)
%         b0 = b0 + bincoeff(rows_sum(i), 2);
%     end
%     b = b0 - a;
%     c0 = 0;
%     for i = 1 : length(cols_sum)
%         c0 = c0 + bincoeff(cols_sum(i) ,2);  
%     end
%     c = c0 - a;                                                                                                           
%     total = bincoeff(n, 2);                                                                                                    
%     d = total - (a + b + c);                                                                                                  
% 
%     ri = (a + d) / total; 
%     ari = (a - b0 * c0 /total) / ((b0 + c0) / 2 - b0 * c0 / total); 
                                                                                                       
    rows_sum = sum(data,2);                                                                                                         
    cols_sum = sum(data,1);                                                                                                         
    a = 0;
     for i = 1 :num_class
        for j = 1 : num_cluster
            if data(i, j) >= 2
                a = a + nchoosek(data(i, j), 2);
            end
        end
     end
    b0 = 0;
    for i = 1 : length(rows_sum)
        if rows_sum(i) >= 2
            b0 = b0 + nchoosek(rows_sum(i), 2);
        end
    end
    b = b0 - a;
    c0 = 0;
    for i = 1 : length(cols_sum)
        if cols_sum(i) >= 2
            c0 = c0 + nchoosek(cols_sum(i) ,2);
        end
    end
    c = c0 - a;                                                                                                           
    total = nchoosek(n, 2);                                                                                                    
    d = total - (a + b + c);                                                                                                  

    ri = (a + d) / total; 
    ari = (a - b0 * c0 /total) / ((b0 + c0) / 2 - b0 * c0 / total); 
end

