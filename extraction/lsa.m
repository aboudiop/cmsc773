%%%perform and write LSA features
INF = 99999999999999;
[S_train classID_train]= libsvm_read('train.megam', 0, INF, INF);
[S_dev classID_dev]= libsvm_read('dev.megam', 0, INF, INF);
[S_test classID_test]= libsvm_read('test.megam', 0, INF, INF);
S = [S_train; S_dev; S_test];
classid = [classID_train; classID_dev; classID_test];

options.disp = 0;
[U Sigma V] = svds(S, 100, 'L', options);
data_arr = U;
labels = classid;

%change accordingly
train_indx = 1:6071;  %index of training samples
dev_indx = 6072:6071+1867; %index of dev samples
test_indx = 6071+1867+1:length(labels); %index of test samples
D = size(data_arr,2);

indx = train_indx;
fid = fopen('train_lsa100.megam', 'w');
if (fid == -1)
    fprintf ('cant open %s for reading', file);    
end
for i=1:length(indx)
    fprintf (fid, '%d\t', labels(indx(i)));
    for j=1:D
        if (data_arr(indx(i),j) > 0.00008)
            fprintf (fid, '%d %f\t', j, data_arr(indx(i),j));
        end
    end
    fprintf(fid, '\n');
end
fclose (fid);

indx = dev_indx;
fid = fopen('dev_lsa100.megam', 'w');
if (fid == -1)
    fprintf ('cant open %s for reading', file);    
end
for i=1:length(indx)
    fprintf (fid, '%d\t', labels(indx(i)));
    for j=1:D
        if (data_arr(indx(i),j) > 0.00008)
            fprintf (fid, '%d %f\t', j, data_arr(indx(i),j));
        end
    end
    fprintf(fid, '\n');
end
fclose (fid);

indx = test_indx;
fid = fopen('test_lsa100.megam', 'w');
if (fid == -1)
    fprintf ('cant open %s for reading', file);    
end
for i=1:length(indx)
    fprintf (fid, '%d\t', labels(indx(i)));
    for j=1:D
        if (data_arr(indx(i),j) > 0.00008)
            fprintf (fid, '%d %f\t', j, data_arr(indx(i),j));
        end
    end
    fprintf(fid, '\n');
end
fclose (fid);

