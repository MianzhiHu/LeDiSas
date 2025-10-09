function LEDIS1analysis

%% DATA ANALYSIS SCRIPT FOR BOSTRA1 TASK

% Adapted by M McKinney SP23 from J Irons AdaptChoiceAnalysis script

% 1. Ensure this file is in the same folder as the individual data files

% 2. Update the sublist of s numbers/codes: (e.g.,('1', '2', '3', ...)

% 3. Hit 'Run'

% 4. These text files will be created:

%       a. Data_BOSTRA1_allSubs_Summary:
%           

%% SETUP
sublist = [1:100]; %subject numbers/'names'
N = numel(sublist)
% sub data file column conditions
subNocond = 1; % subject number
group = 2;
Trial = 3;
block = 4; % 0 = practice
response = 11; % 2,3,4,5
acc = 13; % accuracy
points = 16; % points earned
RT = 17; % response time
optchoice = 18; % (0 or 1)
repsw = 19; % (1 or 2)

expname = 'LEDIS1';
%%
for s = 1:length(sublist)

    subNo = sublist(s);
    
    % set groups
    if rem(subNo,2)==1
        group=1; % Optimal High Reward group
    elseif rem(subNo,2)==0
        group=2; % Suboptimal High Reward group
    end

    subGroup(s,1)=group;

    % set up dataframes by group
    dfname = char(strcat('Data_',expname,'_',num2str(subNo),'.txt'));
    if ~isfile(dfname)
        warning('Missing file: %s', dfname);
        continue
    end
    df = dlmread(dfname,'',2,0);
%% EXCLUSION CRITERIA
    % exclude practice trials
    %%%%%%%%%%%%%%%%%%%%%%%%%
    % * exclude block 0 
    %%%%%%%%%%%%%%%%%%%%%%%%%
    df = df((df(:,block)~=0),:);
    
    Nblocks(s,1)=df(end,block);

    % NaNs for RTs beyond 3 STD from overall mean
    rawMeanRT = mean(df(df(:,acc)>0,RT));
    SDRT = std(df(df(:,acc)>0,RT));
    cutoffRT = rawMeanRT +3*SDRT;
    df(df(:,RT)<300,RT) = NaN;
    df(df(:,RT)>cutoffRT,RT) = NaN;
    df(df(:,acc)==0,RT) = NaN;

%% GET OVERALL MEANS

    % mean acc & RT
    meanRT(s,1) = mean(df(:,RT),"omitnan"); % RT
    meanacc(s,1) = mean(df(:,acc),"omitnan"); % acc
    
    % choice RTs
    meanRT_opt(s,1) = mean(df((df(:,optchoice)==1), RT), "omitnan");
    meanRT_subopt(s,1) = mean(df((df(:,optchoice)==0), RT), "omitnan");

    % proportion optimal choices
    propOpt(s,1) = (size(df((df(:,acc)==1)&(df(:,optchoice)==1),:),1))/...
        (size(df((df(:,acc)==1),:),1)); % for accurate trials only

    % proportion of switches
    Switches(s,1) = (size(df((df(:,repsw)==2),:),1))/ ...
        (size(df((df(:,repsw)>0),:),1)); % switch rate. exclude trials when %repsw = NaN (trial number 1, incorrect trials)

    % total points earned
    totalPoints(s,1) = df(end,points);
    
end % subNo
cd summaryStats/
% Export to summary files 
format long g
% summary file (critical analyses)
sublist = sublist';
printoutmeans = [sublist, subGroup, meanacc, meanRT, meanRT_opt, meanRT_subopt, propOpt, Switches, totalPoints, Nblocks];
outputfile1 = strcat('N',num2str(N),'Data_', expname,'_',date,'_Summary.txt');
save(strcat('N',num2str(N),'Data_',expname,'_',date,'Summary.mat'),'printoutmeans');
header = {'Sub','group','acc','RT','RTopt', 'RTsubopt','propOpt','Switches','totalPoints','Nblocks'};
txt=sprintf('%s\t',header{:});
txt(end)='';
dlmwrite(outputfile1,txt,'');
dlmwrite(outputfile1,printoutmeans, '-append','delimiter','\t','precision',6);
