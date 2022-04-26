clear;clc;
load('CSdataset.mat');
[m,n]=size(trainset);
trains=[];
PCA_num=2:2:24;PCAn=length(PCA_num);
correct_rate=zeros(1,length(PCA_num));

figure
hold on
for ik=1:1:PCAn
    ia=2*ik;k=1;
    PCA_trainset=zeros(m,39*ia);PCA_trainlable=zeros(1,39*ia);
    for i=1:n
        if k<39 && k~=13
            if trainlable(i)==k
                trains=[trains trainset(:,i)];
            elseif trainlable(i-1)==k && trainlable(i)==k+1
                pca_coeff=pca(trains);
                training_pca=trains*pca_coeff;
                PCA_trainset(:,(k-1)*ia+1:k*ia)=training_pca(:,1:ia);
                PCA_trainlable(:,(k-1)*ia+1:k*ia)=k;
                if ia==2
                    plot(training_pca(:,1),training_pca(:,2),'*');
                end
                trains=[];k=k+1;
            end
        elseif k==13
            trains=[trains trainset(:,i)];
            if trainlable(i-1)==k && trainlable(i)==k+2
                pca_coeff=pca(trains);
                training_pca=trains*pca_coeff;
                PCA_trainset(:,(k-1)*ia+1:k*ia)=training_pca(:,1:ia);
                PCA_trainlable(:,(k-1)*ia+1:k*ia)=k;
                PCA_trainlable(:,(k)*ia+1:(k+1)*ia)=k+1;
                if ia==2
                    plot(training_pca(:,1),training_pca(:,2),'*');
                end
                trains=[];k=k+2;
            end
        elseif k==39
            trains=[trains trainset(:,i)];
            if i==n
                pca_coeff=pca(trains);
                training_pca=trains*pca_coeff;
                PCA_trainset(:,(k-1)*ia+1:k*ia)=training_pca(:,1:ia);
                PCA_trainlable(:,(k-1)*ia+1:k*ia)=k;
                if ia==2
                    plot(training_pca(:,1),training_pca(:,2),'*');
                end
            end
        end
    end

    [m,n1]=size(testset);
    testlb_r=zeros(size(testlable));
    correctnum=0;fm=39;S=ia;
    for num=1:1:n1
        y=testset(:,num);A=PCA_trainset;[~,N]=size(A);
        mse=zeros(fm,1);
        x_hat=omp(S,A,y);
        for num1=1:1:fm
            xv_hat=x_hat((num1-1)*S+1:num1*S);
            Av=A(:,(num1-1)*S+1:num1*S);
            y_hat=Av*xv_hat;
            squaredError=(double(y) - double(y_hat)) .^ 2;
            mse(num1)=sum(squaredError) / (N  * N);
        end
        testlb_r=find(mse==min(mse));
        if testlb_r==testlable(num)
            correctnum=correctnum+1;
        end
    end
    correct_rate(ik)=correctnum/n1;
end
hold off

figure
plot(PCA_num,correct_rate);
title('Used PCA Columns vs. Correct rate');
xlabel('Number of Columns');
ylabel('Correct rate');