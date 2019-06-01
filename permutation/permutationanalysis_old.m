%% this old script was computationally ineffecient so I retired it

%% Loading in the Appropriate Data
clear all

expt = 1;
cluster = 1;
preswitch = 1;

if expt == 1
    totalsubs = 63;
    if cluster == 1
        filename = 'ClusteredSubs_E1.csv';
        numsubs = 31;
    elseif cluster == 2
        filename = 'NonClusteredSubs_E1.csv';
        numsubs = 32;
    end
elseif expt == 2
     totalsubs = 64;
    if cluster == 1
        filename = 'ClusteredSubs_E2.csv';
        numsubs = 33;
    elseif cluster == 2
        filename = 'NonClusteredSubs_E2.csv';
        numsubs = 31;
    end
elseif expt == 3
    totalsubs = 58;
    if cluster == 1
        filename = 'ClusteredSubs_E3.csv';
        numsubs = 28;
    elseif cluster == 2
        filename = 'NonClusteredSubs_E3.csv';
        numsubs = 30; 
    end
elseif expt == 4
    totalsubs = 65;
    if cluster == 1
        filename = 'ClusteredSubs_E4.csv';
        numsubs = 31; 
    elseif cluster == 2
        filename = 'NonClusteredSubs_E4.csv';
        numsubs = 34;
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
    RT = dataArray{:, 21};
    TypeOfResponse = dataArray{:, 3};
    subject = dataArray{:, 23};
elseif expt == 2 || expt == 4
    TypeOfResponse = dataArray{:, 3};
    ResponseMatrix = dataArray{:, 16};
    Accuracy = dataArray{:, 19};
    RT = dataArray{:, 20};
    subject = dataArray{:, 23};
elseif expt == 3
    TypeOfResponse = dataArray{:, 3};
    ResponseMatrix = dataArray{:, 17};
    Accuracy = dataArray{:, 20};
    RT = dataArray{:, 21};
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
    switchcostSD(j) = stdUnbiased(j) - stdBiased(j);
    switchcostSDACC(j) = stdUnbiasedACC(j) - stdBiasedACC(j);
end

%absolute value the switch costs, since direction only indicates the rule
%dimension things organized around
switchcostRT = abs(switchcostRT_uncorr);
switchcostACC = abs(switchcostACC_uncorr);
switchcostSD = abs(switchcostSD);
switchcostSDACC = abs(switchcostSDACC);

%get rid of the extras that came from running total of totalsubs mturk
%workers, but not all being assigned to either cluster or nonclustered
switchcostRT = switchcostRT(~isnan(switchcostRT));
switchcostACC = switchcostACC(~isnan(switchcostACC));
switchcostSD = switchcostSD(~isnan(switchcostSD));
switchcostSDACC = switchcostSDACC(~isnan(switchcostSDACC));

individualZ_RT = switchcostRT./switchcostSD;
individualZ_ACC = switchcostACC./switchcostSDACC;

%Make summaries of the data for stats
mCONTRAST_RT=mean(switchcostRT); %mean contrast
sCONTRAST_RT=std(switchcostRT); %std contrast
mCONTRAST_ACC=mean(switchcostACC); %mean contrast
sCONTRAST_ACC=std(switchcostACC); %std contrast

zCONTRAST_RT=(mCONTRAST_RT./sCONTRAST_RT)*sqrt(numsubs); %z-scored contrast for mean
zCONTRAST_ACC=(mCONTRAST_ACC./sCONTRAST_ACC)*sqrt(numsubs); %z-scored contrast

%% Data Permutation

%Note that this script took ~1001 seconds for 100 samples; it was not
%particularly computationally efficient; _kdv script is more
%computationally efficient, achieving the same goal for 10k samples within
%~70 seconds

tic;

%shuffle within one subject, then feed in through all subjects
%then has new data for all subjects
%calculate switch costs (absolute value these)
%repeat steps 1-3 10k (for loop
%average switch cost for 1 test statistic (see whether 2.5% beat my test
%statistic) -- do the majority have a mean at least as large as mine?

shuffle = @(v)v(randperm(numel(v)));
clustergrp = sort((unique(str2double(sub))),'ascend');

%define the variables outside the loop
meanpermutedswitchcostsRT = [];
sdpermutedswitchcostsRT = [];
zpermutedswitchcostsRT = [];
meanpermutedswitchcostsACC = [];
sdpermutedswitchcostsACC = [];
zpermutedswitchcostsACC = [];
ztcontrastRT = [];
ztcontrastACC = [];
nperm = 1000;

for k = 1:nperm
    %shuffle all the TrialType labels within each participant and then for
    %all participants
    fakeTSlabel_RT = [];
    fakeTSlabel_ACC = [];
    
    for s = 1:numsubs
       SubRTlabel = shuffle(TSLearn_RT(str2double(sub)==clustergrp(s)));
       SubACClabel = shuffle(TaskSetLearning(str2double(subject)==clustergrp(s)));
       fakeTSlabel_RT = [fakeTSlabel_RT; {SubRTlabel}];
       fakeTSlabel_ACC = [fakeTSlabel_ACC; {SubACClabel}]; 
    end

    %make the fake labels usable
    fakeTSlabel_RT = cell2mat(fakeTSlabel_RT);
    fakeTSlabel_ACC = cell2mat(fakeTSlabel_ACC);

    %preallocation
    fakeUnbiasedTrials = zeros(length(TSLearn_RT),1);
    fakeBiasedTrials = zeros(length(TSLearn_RT),1);
    fakeTrialType = zeros(length(TSLearn_RT),1);

    fakemeanUnbiased = zeros(length(subcount),1);
    fakemeanBiased = zeros(length(subcount),1);

    fakeUnbiasedTrialsACC = zeros(length(TaskSetLearning),1);
    fakeBiasedTrialsACC = zeros(length(TaskSetLearning),1);
    fakeTrialTypeACC = zeros(length(TaskSetLearning),1);

    fakemeanUnbiasedACC = zeros(length(subcount),1);
    fakemeanBiasedACC = zeros(length(subcount),1);

    fakeswitchcostRT_uncorr = zeros(length(subcount),1);
    fakeswitchcostACC_uncorr = zeros(length(subcount),1);

    %i basically just repeated the code from above
    for i = 1:length(TSLearn_RT)
        if fakeTSlabel_RT(i) == 2.1 || fakeTSlabel_RT(i) == 2.2 || fakeTSlabel_RT(i) == 2.3 || fakeTSlabel_RT(i) == 2.4
            fakeTrialType(i) = 2;
            fakeUnbiasedTrials(i) = RT(i);
        elseif fakeTSlabel_RT(i) == 1.1 || fakeTSlabel_RT(i) == 1.2 || fakeTSlabel_RT(i) == 1.3 || fakeTSlabel_RT(i) == 1.4
            fakeTrialType(i) = 1;
            fakeBiasedTrials(i) = RT(i);
        end
    end

    for i = 1:length(TaskSetLearning)
        if fakeTSlabel_ACC(i) == 2.1 || fakeTSlabel_ACC(i) == 2.2 || fakeTSlabel_ACC(i) == 2.3 || fakeTSlabel_ACC(i) == 2.4
            fakeTrialTypeACC(i) = 2;
            fakeUnbiasedTrialsACC(i) = Accuracy(i);
        elseif fakeTSlabel_ACC(i) == 1.1 || fakeTSlabel_ACC(i) == 1.2 || fakeTSlabel_ACC(i) == 1.3 || fakeTSlabel_ACC(i) == 1.4
            fakeTrialTypeACC(i) = 1;
            fakeBiasedTrialsACC(i) = Accuracy(i);
        end      
    end

    for j = 1:length(subcount)
        fakemeanUnbiased(j) = nanmean(fakeUnbiasedTrials(fakeTrialType==2 & str2double(sub)==j));
        fakemeanBiased(j) = nanmean(fakeBiasedTrials(fakeTrialType==1 & str2double(sub)==j));

        fakemeanUnbiasedACC(j) = nanmean(fakeUnbiasedTrialsACC(fakeTrialTypeACC==2 & str2double(subject)==j));
        fakemeanBiasedACC(j) = nanmean(fakeBiasedTrialsACC(fakeTrialTypeACC==1 & str2double(subject)==j));

        fakeswitchcostRT_uncorr(j) = (fakemeanUnbiased(j) - fakemeanBiased(j));
        fakeswitchcostACC_uncorr(j) = (fakemeanUnbiasedACC(j) - fakemeanBiasedACC(j));
    end

    fakeswitchcostRT = abs(fakeswitchcostRT_uncorr);
    fakeswitchcostACC = abs(fakeswitchcostACC_uncorr);

    fakeswitchcostRT = fakeswitchcostRT(~isnan(fakeswitchcostRT));
    fakeswitchcostACC = fakeswitchcostACC(~isnan(fakeswitchcostACC));

    %summaries of the data
    fakemCONTRAST_RT=mean(fakeswitchcostRT); %mean contrast
    fakesCONTRAST_RT=std(fakeswitchcostRT); %std contrast
    fakemCONTRAST_ACC=mean(fakeswitchcostACC); %mean contrast
    fakesCONTRAST_ACC=std(fakeswitchcostACC); %std contrast

    fakezCONTRAST_RT=(fakemCONTRAST_RT./fakesCONTRAST_RT)*sqrt(numsubs); %z-scored contrast
    fakezCONTRAST_ACC=(fakemCONTRAST_ACC./fakesCONTRAST_ACC)*sqrt(numsubs); %z-scored contrast

    %what i basically want at the end of the script are these mean switch
    %costs (might be best to use z-scores instead tho) b/c I will look at
    %the population as a whole and see whether my mean > permutation mean
    %at least 2.5% of the time
    meanpermutedswitchcostsRT = [meanpermutedswitchcostsRT; {fakemCONTRAST_RT}];
    sdpermutedswitchcostsRT = [sdpermutedswitchcostsRT; {fakesCONTRAST_RT}];
    zpermutedswitchcostsRT = [zpermutedswitchcostsRT; {fakezCONTRAST_RT}];
    meanpermutedswitchcostsACC = [meanpermutedswitchcostsACC; {fakemCONTRAST_ACC}];
    sdpermutedswitchcostsACC = [sdpermutedswitchcostsACC; {fakesCONTRAST_ACC}];
    zpermutedswitchcostsACC = [zpermutedswitchcostsACC; {fakezCONTRAST_ACC}];
    
end

meanpermutedswitchcostsRT = cell2mat(meanpermutedswitchcostsRT);
sdpermutedswitchcostsRT = cell2mat(sdpermutedswitchcostsRT);
zpermutedswitchcostsRT = cell2mat(zpermutedswitchcostsRT);
meanpermutedswitchcostsACC = cell2mat(meanpermutedswitchcostsACC);
sdpermutedswitchcostsACC = cell2mat(sdpermutedswitchcostsACC);
zpermutedswitchcostsACC = cell2mat(zpermutedswitchcostsACC);
teststatisticRT = (zCONTRAST_RT>zpermutedswitchcostsRT);
teststatisticACC = (zCONTRAST_ACC>zpermutedswitchcostsACC);

csvwrite([newfile 'switchcosts.csv'], [switchcostRT, switchcostACC]);
csvwrite([newfile 'zstats_mainexpt.csv'], [zCONTRAST_RT, zCONTRAST_ACC]);
csvwrite([newfile 'permutationstuff.csv'], [fakezCONTRAST_RT, fakezCONTRAST_ACC, teststatisticRT, teststatisticACC]);
toc