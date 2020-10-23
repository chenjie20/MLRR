%------------------------------------------------------------------------
% Eigenface computing function
% train_data: each colomun represents a corresponding sample
%------------------------------------------------------------------------
function [eigen_vector, eigen_value, mean_value] = f_pca(train_data,  eigen_num)

[row_num, col_num] = size(train_data);
mean_value = mean(train_data, 2);  
train_data = train_data - mean_value * ones(1,col_num);

% not small sample size case, directly compute the matrix to obtain the result.
if row_num < col_num     
   cov_matrix = train_data * train_data' / (col_num - 1);    
   [e_vector, e_value]= find_k_max_eigen(cov_matrix, eigen_num);
   eigen_vector = e_vector;
   eigen_value = e_value;
else
    cov_matrix = train_data' * train_data/ (col_num - 1);   
    [e_vector, e_value]= find_k_max_eigen(cov_matrix, eigen_num);
    clear cov_matrix;
    eigen_value = e_value;
    tmp = e_vector * diag(e_value.^(-1/2));
    eigen_vector = (train_data /sqrt(col_num - 1)) * tmp;
    
%     cov_matrix = train_data' * train_data;  
%     [e_vector, e_value]= find_k_max_eigen(cov_matrix, eigen_num);
%     eigen_value = e_value;
%     eigen_vector = train_data  * e_vector * diag(e_value.^(-1/2));
    
    
%   disc_set=zeros(NN,Eigen_NUM);
%   clear R S;
%   Train_SET=Train_SET/sqrt(Train_NUM-1);
%   
%   for k=1:Eigen_NUM
%     a = Train_SET*V(:,k);
%     b = (1/sqrt(disc_value(k)));
%     disc_set(:,k)=b*a;
%   end
%     
end

end



function [eigen_vector,eigen_value] = find_k_max_eigen(cov_matrix, eigen_num)
%  the elements of eigen_value in descending order

% if the matrix is symmetrical, the following method can be used.
[v d] = eig(cov_matrix);
d1 = diag(d);
e_value = flipud(d1);
e_vector = fliplr(v);
eigen_value = e_value(1 : eigen_num);
eigen_vector = e_vector(:, 1 : eigen_num);

% the second method if the matrix is not symmetrical
% row_num = size(cov_matrix, 1);
% [v, d] = eig(cov_matrix);
% d = diag(d);
% [e_value,e_value_index] = sort (d,1,'descend');
% 
% eigen_vector = zeros(row_num, eigen_num);
% eigen_value = zeros(eigen_num, 1);
% for i = 1 : eigen_num
%     eigen_vector(:,i) =  v(:,e_value_index(i));
%     eigen_value(i) = e_value(i);
% end

end