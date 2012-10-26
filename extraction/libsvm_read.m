function [S classID]= libsvm_read(file, start_ind, end_ind, num_samples)
%%read libsvm format and return a sparse matrix (to save memory)
    fid = fopen(file, 'r');
    if (fid == -1)
        fprintf ('cant open %s for reading', file);
        return;
    end
    
    i=1;
    num_samples
    while  ~feof(fid) && (i <= num_samples)
        fprintf('line %d...\n',i);
        line = fgets (fid);
        remain = line;
        j_ind=2;
        [token remain] = strtok(remain, ' :'); %get class id
        %token(1)
        %token(2)
        %remain
        num = str2num(token);
        %num
        truth(i) = num;
        data_arr(i,j_ind) = num;  %store class id
        j_ind = j_ind+1;
        while true        
            [token remain] = strtok(remain, ' :'); %get feature index
            num1 = str2num(token);
            if (isempty(token))
                break;
            end            
            [token remain] = strtok(remain, ' :'); %get feature value
            num2 = str2num(token);
            if (num1 > end_ind) break; end
            if (num1 < start_ind) continue; end
            data_arr(i,j_ind) = num1;            
            j_ind = j_ind+1;
            data_arr(i,j_ind) = num2;            
            j_ind = j_ind+1;            
        end
        lengths(i) = j_ind-1;
        data_arr(i,1) = j_ind-1;
        i = i+1;
    end
    
    [N D] = size(data_arr);
    maxIndex = max(max(data_arr)) + 1; %max(max(data_arr)) - offset; % max index of any nonzero feature
    S = [];
    classID = zeros(N,1);
    % each iteration in the for loop below creates one sparse (matlab format) example
    for ind = 1:N
        fprintf('processing example %d\n',ind);
        exampleN = data_arr(ind,:);% a row vector consisting of all info for example # ind
        classID(ind) = exampleN(1,2); % class/cluster id is the second element in this row
        j = exampleN(1,3:2:D-1); % vector of column indices (they are at odd locations 3,5,7 etc).
        s = exampleN(1,4:2:D); % features are at even locations (4,6,8, etc)
        s = s(s~=0); % discard all zero valued features
        %% '+ 1' since index starts from 0 in data_arr, if it's not the case remove +1
        j = j(1:length(s)) - offset + 1; % we only care about the columns indices of features that are nonzero
        %i = ones(1,1:length(s)); % vector of all ones (will be our row index; since we are constructing a single row at a time, we used all 1's) 
        i = ones(1,length(s)); % vector of all ones (will be our row index; since we are constructing a single row at a time, we used all 1's) 
        % use i,j,s etc to construct a sparse feature vector for this example
        % and append in the existing matrix of previous examples.
        S = [S;sparse(i,j,s,1,maxIndex)];
    end
