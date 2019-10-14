%% Loading in the Appropriate Data
clear all

expt = 4;
cluster = 1;
preswitch = 1;

if expt == 1
    totalsubs = 63;
    if cluster == 1
        filename = 'ClusteredSubs_E1.csv';
        numsubs = 31;
        titleexp = 'Experiment 1 Clustered';
    elseif cluster == 2
        filename = 'NonClusteredSubs_E1.csv';
        numsubs = 32;
        titleexp = 'Experiment 1 Nonclustered';
    end
elseif expt == 2
     totalsubs = 64;
    if cluster == 1
        filename = 'ClusteredSubs_E2.csv';
        numsubs = 33;
        titleexp = 'Experiment 2 Clustered';
    elseif cluster == 2
        filename = 'NonClusteredSubs_E2.csv';
        numsubs = 31;
        titleexp = 'Experiment 2 Nonclustered';
    end
elseif expt == 3
    totalsubs = 58;
    if cluster == 1
        filename = 'ClusteredSubs_E3.csv';
        numsubs = 28;
        titleexp = 'Experiment 3 Clustered';
    elseif cluster == 2
        filename = 'NonClusteredSubs_E3.csv';
        numsubs = 30;
        titleexp = 'Experiment 3 Nonclustered';
    end
elseif expt == 4
    totalsubs = 65;
    if cluster == 1
        filename = 'ClusteredSubs_E4.csv';
        numsubs = 31;
        titleexp = 'Experiment 4 Clustered';
    elseif cluster == 2
        filename = 'NonClusteredSubs_E4.csv';
        numsubs = 34;
        titleexp = 'Experiment 4 Nonclustered';
    end
end

delimiter = ',';
startRow = 2;
if expt == 1
    formatSpec = '%f%f%s%f%f%f%f%f%f%f%f%f%f%s%s%f%f%s%s%f%f%f%s%[^\n\r]'; %e1
elseif expt == 2 || expt == 4
    formatSpec = '%f%f%s%f%f%f%f%f%f%f%f%f%f%s%f%f%s%s%f%f%f%f%s%[^\n\r]';
elseif expt == 3
    formatSpec = '%f%f%s%f%f%f%f%f%f%f%f%f%f%s%f%f%f%s%s%f%f%f%f%s%[^\n\r]'; %e3
end
fileID = fopen(filename,'r');
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'EmptyValue' ,NaN,'HeaderLines' ,startRow-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
fclose(fileID);

if expt == 1
    ResponseMatrix = dataArray{:, 17}; %E1
    Accuracy = dataArray{:, 20};
    RTuncorr = dataArray{:, 21};
    TypeOfResponse = dataArray{:, 3};
    subject = dataArray{:, 23};
elseif expt == 2 || expt == 4
    TypeOfResponse = dataArray{:, 3};
    ResponseMatrix = dataArray{:, 16};
    Accuracy = dataArray{:, 19};
    RTuncorr = dataArray{:, 20};
    subject = dataArray{:, 23};
elseif expt == 3
    TypeOfResponse = dataArray{:, 3};
    ResponseMatrix = dataArray{:, 17};
    Accuracy = dataArray{:, 20};
    RTuncorr = dataArray{:, 21};
    subject = dataArray{:, 24};
end

MatrixCounter = dataArray{:, 7};
TrialCounter = dataArray{:, 6};
clearvars filename delimiter startRow formatSpec fileID dataArray ans;

if preswitch == 1
    newfile = ['switchcostanalysis_' num2str(expt) '_' num2str(cluster) '_' num2str(preswitch) ];
else
    newfile = ['switchcostanalysis_' num2str(expt) '_' num2str(cluster) ];
end
%% Setting up the Trial Types to calculate the switch costs

TaskSetLearning = zeros(length(MatrixCounter),1);

for i = 1:length(MatrixCounter)
    if TrialCounter(i) ~= 0 %if not the first trial
        if ResponseMatrix(i) == 1
            if ResponseMatrix(i-1) == 2 %if A2-1, aka BiasedLearner
                TaskSetLearning(i) = 1.2;
            elseif ResponseMatrix(i-1) == 3 %if A3-1, aka NonbiasedLearner
                TaskSetLearning(i) = 2.3;
            elseif ResponseMatrix(i-1) == 4 %if A4-1, aka dble switch
                TaskSetLearning(i) = 3.1;
            elseif ResponseMatrix(i-1) == 1 %A1-1 aka repeat
                TaskSetLearning(i) = 4.1;
            end
         elseif ResponseMatrix(i) == 2
            if ResponseMatrix(i-1) == 1 %if A1-2, aka BiasedLearner
                TaskSetLearning(i) = 1.1;
            elseif ResponseMatrix(i-1) == 3 %if A3-2, aka dble switch
                TaskSetLearning(i) = 3.2;
            elseif ResponseMatrix(i-1) == 4 %if A4-2, aka NonbiasedLearner
                TaskSetLearning(i) = 2.4;
            elseif ResponseMatrix(i-1) == 2 %aka 2-2 repeat
                TaskSetLearning(i) = 4.2;
            end
         elseif ResponseMatrix(i) == 3
            if ResponseMatrix(i-1) == 1 %if A1-3, aka NonbiasedLearner
                TaskSetLearning(i) = 2.1;
            elseif ResponseMatrix(i-1) == 2 %if A2-3, aka dble switch
                TaskSetLearning(i) = 3.3;
            elseif ResponseMatrix(i-1) == 4 %if A4-3, aka BiasedLearner
                TaskSetLearning(i) = 1.4;
            elseif ResponseMatrix(i-1) == 3 % aka 3-3, repeat
                TaskSetLearning(i) = 4.3;
            end
         elseif ResponseMatrix(i) == 4
            if ResponseMatrix(i-1) == 1 %if A1-4, aka dble switch
                TaskSetLearning(i) = 3.4;
            elseif ResponseMatrix(i-1) == 2 %if A2-4, aka NonbiasedLearner
                TaskSetLearning(i) = 2.2;
            elseif ResponseMatrix(i-1) == 3 %if A3-4, aka BiasedLearner
                TaskSetLearning(i) = 1.3;
            elseif ResponseMatrix(i-1) == 4
                TaskSetLearning(i) = 4.4; %aka 4-4 repeat
            end
         end      
    else
        TaskSetLearning(i) = 0;
    end
end

%% Need to filter out the data - no filters for ACC, but for RT, take only correct responses at the right threshold
if expt == 3
    if preswitch == 1
        badTrials=(Accuracy==0 | RTuncorr < 200 | RTuncorr > 1250 | TrialCounter == 0 | TrialCounter > 59);
        idx = ~(badTrials);
        idx2 = ~(TrialCounter > 59);
        RT = RTuncorr(idx);
        TSLearn_RT = TaskSetLearning(idx);
        sub = subject(idx);
        subject = subject(idx2);
        TaskSetLearning = TaskSetLearning(idx2);
        Accuracy = Accuracy(idx2);
    else
        badTrials=(Accuracy==0 | RTuncorr < 200 | RTuncorr > 1250 | TrialCounter == 0);
        idx = ~(badTrials);
        RT = RTuncorr(idx);
        TSLearn_RT = TaskSetLearning(idx);
        sub = subject(idx);        
    end
else
    badTrials=(Accuracy==0 | RTuncorr < 200 | RTuncorr > 1250 | TrialCounter == 0);
    idx = ~(badTrials);
    RT = RTuncorr(idx);
    TSLearn_RT = TaskSetLearning(idx);
    sub = subject(idx);
end

%% Calculating Real Switch Costs

subcount = 1:totalsubs;
subcount = subcount';

%preallocation
RepeatTrials = zeros(length(TSLearn_RT),1);
DoubleSwitchTrials = zeros(length(TSLearn_RT),1);
UnbiasedTrials = zeros(length(TSLearn_RT),1);
BiasedTrials = zeros(length(TSLearn_RT),1);
TrialType = zeros(length(TSLearn_RT),1);

RepeatTrialsACC = zeros(length(TaskSetLearning),1);
DoubleSwitchTrialsACC = zeros(length(TaskSetLearning),1);
UnbiasedTrialsACC = zeros(length(TaskSetLearning),1);
BiasedTrialsACC = zeros(length(TaskSetLearning),1);
TrialTypeACC = zeros(length(TaskSetLearning),1);

meanRepeat = zeros(length(subcount),1);
meanDbleSwitch = zeros(length(subcount),1);
meanUnbiased = zeros(length(subcount),1);
meanBiased = zeros(length(subcount),1);

meanRepeatACC = zeros(length(subcount),1);
meanDbleSwitchACC = zeros(length(subcount),1);
meanUnbiasedACC = zeros(length(subcount),1);
meanBiasedACC = zeros(length(subcount),1);

stdRepeat = zeros(length(subcount),1);
stdDbleSwitch = zeros(length(subcount),1);
stdUnbiased = zeros(length(subcount),1);
stdBiased = zeros(length(subcount),1);

stdRepeatACC = zeros(length(subcount),1);
stdDbleSwitchACC = zeros(length(subcount),1);
stdUnbiasedACC = zeros(length(subcount),1);
stdBiasedACC = zeros(length(subcount),1);

switchcostRT_uncorr = zeros(length(subcount),1);
switchcostACC_uncorr = zeros(length(subcount),1);
switchcostSD = zeros(length(subcount),1);
switchcostSDACC = zeros(length(subcount),1);

%switch costs calculated by first averaging across the different iterations
%of trial types & then subtracting switch from repeat, and unbiased from
%biased to get a sense of switching from the dimension around which they
%structured their response rules

%3.1-3.4 = double switch trials (e.g., OF -> YM); 4.1-4.4 = feature repeat trials;
%1.1-1.4 = biased dimension (if clustered; if non-clustered, this is
%arbitrary, just one of the dimensions)
%2.1-2.4 = unbiased dimension (if clustered; if non-clustered, this is
%arbitrary, just one of the dimensions)

%determine RT and Accuracy for the repeat, switch, and biased/unbiased
%trials and mark those trial types for later aggregation
for i = 1:length(TSLearn_RT)
    if TSLearn_RT(i) == 4.1 || TSLearn_RT(i) == 4.2 || TSLearn_RT(i) == 4.3 || TSLearn_RT(i) == 4.4
        TrialType(i) = 4;
        RepeatTrials(i) = RT(i);
    elseif TSLearn_RT(i) == 3.1 || TSLearn_RT(i) == 3.2 || TSLearn_RT(i) == 3.3 || TSLearn_RT(i) == 3.4
        TrialType(i) = 3;
        DoubleSwitchTrials(i) = RT(i);
    elseif TSLearn_RT(i) == 2.1 || TSLearn_RT(i) == 2.2 || TSLearn_RT(i) == 2.3 || TSLearn_RT(i) == 2.4
        TrialType(i) = 2;
        UnbiasedTrials(i) = RT(i);
    elseif TSLearn_RT(i) == 1.1 || TSLearn_RT(i) == 1.2 || TSLearn_RT(i) == 1.3 || TSLearn_RT(i) == 1.4
        TrialType(i) = 1;
        BiasedTrials(i) = RT(i);
    end
end

%have to do it separately with Accuracy b/c the indices are different due
%to the filtering
for i = 1:length(TaskSetLearning)
    if TaskSetLearning(i) == 4.1 || TaskSetLearning(i) == 4.2 || TaskSetLearning(i) == 4.3 || TaskSetLearning(i) == 4.4
        TrialTypeACC(i) = 4;
        RepeatTrialsACC(i) = Accuracy(i);
    elseif TaskSetLearning(i) == 3.1 || TaskSetLearning(i) == 3.2 || TaskSetLearning(i) == 3.3 || TaskSetLearning(i) == 3.4
        TrialTypeACC(i) = 3;
        DoubleSwitchTrialsACC(i) = Accuracy(i);
    elseif TaskSetLearning(i) == 2.1 || TaskSetLearning(i) == 2.2 || TaskSetLearning(i) == 2.3 || TaskSetLearning(i) == 2.4
        TrialTypeACC(i) = 2;
        UnbiasedTrialsACC(i) = Accuracy(i);
    elseif TaskSetLearning(i) == 1.1 || TaskSetLearning(i) == 1.2 || TaskSetLearning(i) == 1.3 || TaskSetLearning(i) == 1.4
        TrialTypeACC(i) = 1;
        BiasedTrialsACC(i) = Accuracy(i);
    end      
end

%calculate the mean switch cost based on the trial types
for j = 1:length(subcount)
    meanRepeat(j) = nanmean(RepeatTrials(TrialType==4 & str2double(sub)==j));
    meanDbleSwitch(j) = nanmean(DoubleSwitchTrials(TrialType==3 & str2double(sub)==j));
    meanUnbiased(j) = nanmean(UnbiasedTrials(TrialType==2 & str2double(sub)==j));
    meanBiased(j) = nanmean(BiasedTrials(TrialType==1 & str2double(sub)==j));
    
    stdRepeat(j) = nanstd(RepeatTrials(TrialType==4 & str2double(sub)==j));
    stdDbleSwitch(j) = nanstd(DoubleSwitchTrials(TrialType==3 & str2double(sub)==j));
    stdUnbiased(j) = nanstd(UnbiasedTrials(TrialType==2 & str2double(sub)==j));
    stdBiased(j) = nanstd(BiasedTrials(TrialType==1 & str2double(sub)==j));  

    meanRepeatACC(j) = nanmean(RepeatTrialsACC(TrialTypeACC==4 & str2double(subject)==j));
    meanDbleSwitchACC(j) = nanmean(DoubleSwitchTrialsACC(TrialTypeACC==3 & str2double(subject)==j));
    meanUnbiasedACC(j) = nanmean(UnbiasedTrialsACC(TrialTypeACC==2 & str2double(subject)==j));
    meanBiasedACC(j) = nanmean(BiasedTrialsACC(TrialTypeACC==1 & str2double(subject)==j));

    stdRepeatACC(j) = nanstd(RepeatTrialsACC(TrialType==4 & str2double(sub)==j));
    stdDbleSwitchACC(j) = nanstd(DoubleSwitchTrialsACC(TrialType==3 & str2double(sub)==j));
    stdUnbiasedACC(j) = nanstd(UnbiasedTrialsACC(TrialType==2 & str2double(sub)==j));
    stdBiasedACC(j) = nanstd(BiasedTrialsACC(TrialType==1 & str2double(sub)==j));
        
    %could be (Unbiased - Repeat) - (Biased - Repeat) but this is the same
    switchcostRT_uncorr(j) = (meanUnbiased(j) - meanBiased(j));
    switchcostACC_uncorr(j) = (meanUnbiasedACC(j) - meanBiasedACC(j));
end

%absolute value the switch costs, since direction only indicates the rule
%dimension things organized around
switchcostRT = abs(switchcostRT_uncorr);
switchcostACC = abs(switchcostACC_uncorr);

%don't need to do this when using d-prime as the normalized statistic
%across individuals (u1 - u2 / sqrt(sigma_1^2 + sigma_2^2)
%switchcostSD = abs(switchcostSD);
%switchcostSDACC = abs(switchcostSDACC);

%get rid of the extras that came from running total of totalsubs mturk
%workers, but not all being assigned to either cluster or nonclustered
switchcostRT = switchcostRT(~isnan(switchcostRT));
switchcostACC = switchcostACC(~isnan(switchcostACC));
stdUnbiased = stdUnbiased(~isnan(stdUnbiased));
stdBiased = stdBiased(~isnan(stdBiased));
stdUnbiasedACC = stdUnbiasedACC(~isnan(stdUnbiasedACC));
stdBiasedACC = stdBiasedACC(~isnan(stdBiasedACC));

individuald_RT = switchcostRT./sqrt(0.5*((stdUnbiased.^2)+(stdBiased.^2)));
individuald_ACC = switchcostACC./sqrt(0.5*((stdUnbiasedACC.^2)+(stdBiasedACC.^2)));

%record all the switch costs for the individuals, so it can be used in the
%mixed models analysis
csvwrite([newfile 'switchcosts_Individ.csv'], [individuald_RT, individuald_ACC, sort((str2double(unique(subject))),'ascend')]);

%Make summaries of the data for stats
mCONTRAST_RT=mean(switchcostRT); %mean contrast
sCONTRAST_RT=std(switchcostRT); %std contrast
mCONTRAST_ACC=mean(switchcostACC); %mean contrast
sCONTRAST_ACC=std(switchcostACC); %std contrast

zCONTRAST_RT=(mCONTRAST_RT./sCONTRAST_RT)*sqrt(numsubs); %z-scored contrast for mean
zCONTRAST_ACC=(mCONTRAST_ACC./sCONTRAST_ACC)*sqrt(numsubs); %z-scored contrast

%% permute data

% Thank you, Khoi Vo, for making a more efficient version of the
% permutation code that I had

% Note: Here, we will run the permutations per subject -- the permutation
% will generate a switchCost_RT and _Acc distributions with sizes
% [nperm,numsubs]. Each row is a permutation and each col is a participant.

% In the original script, I calculated each ROW and ran the
% forloop nperm times...this was what caused such a slow down.

% getting the size for rtLabels and AccLabels for each participant
% generate permuted labels en masse based on each participant's unique data lengths
clustergrp = sort((unique(str2double(sub))),'ascend');
len_RTLabel = [];
len_AccLabel = [];
for j = 1:numsubs
    len_RTLabel = [len_RTLabel,sum(str2double(sub)==clustergrp(j))];
    len_AccLabel = [len_AccLabel,sum(str2double(subject)==clustergrp(j))];
end

% starting the permutation process by instantiating the distribution arrays
nperm = 10000;
switchCostRT_Dist = zeros(nperm,numsubs);
switchCostAcc_Dist = zeros(nperm,numsubs);
SC_SD_Dist_UB = zeros(nperm,numsubs);
SC_SD_Dist_B = zeros(nperm,numsubs);
SC_SDACC_Dist_UB = zeros(nperm,numsubs);
SC_SDACC_Dist_B = zeros(nperm,numsubs);

tic;
for j = 1:numsubs
    % here, the data labels are rounded to the nearest whole integer so we
    % don't have to go through an unnecessary forloop to define the data
    % labels based on the .1, .2, etc
    SubRTlabel = round(TSLearn_RT(str2double(sub)==clustergrp(j))',0);
    SubACClabel = round(TaskSetLearning(str2double(subject)==clustergrp(j))',0);
    
    % extracting only the necessary RT for the participant from bigger
    % matrices you've defined in preceding codes
    tempRT = RT(str2double(sub)==clustergrp(j));
    tempAcc = Accuracy(str2double(subject)==clustergrp(j));

    % Generate the randomized permutations for the data labels -- this just
    % generates a matrix of length [nperm,length(RT)] or [nperm,length(Acc)]
    % for the participant. Each row of the matrix just contains the permuted
    % trial number that we will then use to extract the corresponding label
    rand_RTLabel = cell2mat(arrayfun(@(x)randperm(len_RTLabel(j)),(1:nperm)','UniformOutput',0)); % randomizing condition identity for each subj
    rand_ACCLabel = cell2mat(arrayfun(@(x)randperm(len_AccLabel(j)),(1:nperm)','UniformOutput',0)); % randomizing condition identity for each subj

    % matrices to contain the means for each permutation -- final size is
    % [nperm,1]
    biasedRT = [];
    unbiasedRT = [];
    biasedAcc = [];
    unbiasedAcc = [];
    biasedSDall = [];
    unbiasedSDall = [];
    biasedSDallACC = [];
    unbiasedSDallACC = [];
    for i = 1:nperm
         % permuted data labels based on the permuted trial order generated above
        fake_RTLabel = SubRTlabel(1,rand_RTLabel(i,:))';
        fake_AccLabel = SubACClabel(1,rand_ACCLabel(i,:))';
        
        %individual SD for each permutation so I can calculate individual
        %Z-stats for each permutation for each subject
        tempRTsdbias = tempRT(fake_RTLabel==1);
        tempRTsdunbias = tempRT(fake_RTLabel==2);
        tempACCsdbias = tempAcc(fake_AccLabel==1);
        tempACCsdunbias = tempAcc(fake_AccLabel==2);
        
        biasedSD = nanstd(tempRTsdbias);
        unbiasedSD = nanstd(tempRTsdunbias);
        biasedSDACC = nanstd(tempACCsdbias);
        unbiasedSDACC = nanstd(tempACCsdunbias);
        
        % getting RT and Acc for labels 1 and 2 only
        tempRT2 = tempRT(fake_RTLabel<=2);
        tempAcc2 = tempAcc(fake_AccLabel<=2);
        
        % Design matrix to conduct regression (vectorized implementation)
        % for each design matrix, the first column is the constant so it's
        % all 1; the second column is a boolean for which trial belonged to
        % label 2. Thus, when you think of this as a regression, the fitted
        % intercept would be the mean for label 1 and the sum of the
        % intercept and the other fitted beta would be the mean for label 2
        desRT = [ones(length(tempRT2),1),fake_RTLabel(fake_RTLabel<=2)-1];
        desAcc = [ones(length(tempAcc2),1),fake_AccLabel(fake_AccLabel<=2)-1];
        
        % running regression based on design matrix & RT and Acc
        %   getting the means of biased & unbiased trials based on the
        %   permuted labels -- also getting rid of NaN's here
        betasRT = desRT(~isnan(tempRT2),:)\tempRT2(~isnan(tempRT2)); % beta(1) = label 1; sum(betas) = label 2
        betasAcc = desAcc(~isnan(tempAcc2),:)\tempAcc2(~isnan(tempAcc2)); % beta(1) = label 1; sum(betas) = label 2
        
        % logging the data
        biasedRT = [biasedRT;betasRT(1)];
        biasedAcc = [biasedAcc;betasAcc(1)];
        unbiasedRT = [unbiasedRT;sum(betasRT)];
        unbiasedAcc = [unbiasedAcc;sum(betasAcc)];
        
        biasedSDall = [biasedSDall;biasedSD];
        unbiasedSDall = [unbiasedSDall;unbiasedSD];
        biasedSDallACC = [biasedSDallACC;biasedSDACC];
        unbiasedSDallACC = [unbiasedSDallACC;unbiasedSDACC];
        
        clear fake_RTLabel fake_AccLabel tempRT2 tempAcc2 desRT desAcc betasRT betasAcc tempRTsdbias tempRTsdunbias tempACCsdbias tempACCsdunbias
    end
    
    % after running all permutations, calculate the switch costs
    switchCostRT = abs(unbiasedRT - biasedRT);
    switchCostAcc = abs(unbiasedAcc - biasedAcc);
    
    % concatenating the switch costs for the participant into main outcome
    % variable for the switch cost distributions
    switchCostRT_Dist(:,j) = switchCostRT;
    switchCostAcc_Dist(:,j) = switchCostAcc;
    SC_SD_Dist_UB(:,j) = unbiasedSDall;
    SC_SD_Dist_B(:,j) = biasedSDall;
    SC_SDACC_Dist_UB(:,j) = unbiasedSDallACC;
    SC_SDACC_Dist_B(:,j) = biasedSDallACC;
    
    clear SubRTlabel SubACClabel tempRT tempAcc rand_RTLabel rand_ACCLabel biasedRT unbiasedRT biasedAcc unbiasedAcc switchCostRT switchCostAcc
end
toc;

% summaries of the data -- based on what you had -- I just made sure the
% variable names have been changed accordingly

%group-level treats each subject as interchangeable, averaging across
%subjects to get 10k z-statistics for comparison
fakemCONTRAST_RT=mean(switchCostRT_Dist,2); %mean contrast
fakesCONTRAST_RT=std(switchCostRT_Dist,[],2); %std contrast
fakemCONTRAST_ACC=mean(switchCostAcc_Dist,2); %mean contrast
fakesCONTRAST_ACC=std(switchCostAcc_Dist,[],2); %std contrast

%now compare across all participants within the group with the z-score
fakezCONTRAST_RT=(fakemCONTRAST_RT./fakesCONTRAST_RT)*sqrt(numsubs); %z-scored contrast
fakezCONTRAST_ACC=(fakemCONTRAST_ACC./fakesCONTRAST_ACC)*sqrt(numsubs); %z-scored contrast
teststatisticRT = (zCONTRAST_RT>fakezCONTRAST_RT);
teststatisticACC = (zCONTRAST_ACC>fakezCONTRAST_ACC);

%individual treats each subject as having their own distribution, with 10k
%generated fake means and stdevs, to calculate their own z-stats
individualfaked_RT = switchCostRT_Dist./sqrt(0.5*((SC_SD_Dist_UB.^2) +  (SC_SD_Dist_B.^2)));
individualfaked_ACC = switchCostAcc_Dist./sqrt(0.5*((SC_SDACC_Dist_UB.^2) +  (SC_SDACC_Dist_B.^2)));

for j = 1:length(clustergrp)
    individdiffRT(:,j) = switchcostRT(j)>switchCostRT_Dist(:,j);
    individdiffACC(:,j) = switchcostACC(j)>switchCostAcc_Dist(:,j);
    individualdiffRT(:,j) = individuald_RT(j)>individualfaked_RT(:,j);
    individualdiffACC(:,j) = individuald_ACC(j)>individualfaked_ACC(:,j);
end

csvwrite([newfile 'individdiff_summary.csv'], [sum(individdiffRT,1); sum(individdiffACC,1)]);
csvwrite([newfile 'individdiff_summary_Z.csv'], [sum(individualdiffRT,1); sum(individualdiffACC,1)]);
csvwrite([newfile 'switchcosts.csv'], [switchcostRT, switchcostACC]);
csvwrite([newfile 'zstats_mainexpt.csv'], [zCONTRAST_RT, zCONTRAST_ACC]);
csvwrite([newfile 'permutation_all.csv'], [fakezCONTRAST_RT, fakezCONTRAST_ACC, teststatisticRT, teststatisticACC]);
csvwrite([newfile 'permutation_summary.csv'], [sum(teststatisticRT), sum(teststatisticACC)]);

