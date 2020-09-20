function [ err ] = calculate_err(original_ids, actual_ids)
    if(size(original_ids, 2) == 1)
        original_ids = original_ids';
    end
    if(size(actual_ids, 2) == 1)
        actual_ids = actual_ids';
    end
    
    group_num = length(unique(original_ids));    
    rand('state', 1000000);
    permutation = perms(1 : group_num);

    errors = zeros(1, size(permutation, 1));
    for j = 1 : size(permutation, 1)
        errors(j) = sum(abs(actual_ids(1, :) - permutation(j, original_ids)) > 0.1);
    end

    err = min(errors)/length(actual_ids);
end

