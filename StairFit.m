function [MUNE,STAIR]=StairFit(data,D1,D2,inc,th)

% $$$ Documentation of StairFit  $$$
% Author:Maoqi Chen
% Email:hiei@mail.ustc.edu.cn  or  maoqi.chen@uhrs.edu.cn
% Update:2024.05.09
% Please cite:
% [1]Chen, M., Lu Z, Zong Y, Li X, and Zhou P A Novel Analysis of Compound Muscle Action Potential Scan: Staircase Function Fitting and StairFit Motor Unit Number Estimation. IEEE Journal of Biomedical and Health Informatics, 2023. 27(3): 1579-1587.
% [2]Chen, M. and Zhou, P. StairFit MUNE: A free and open source MatLab program for CMAP scan processing. Clinical Neurophysiology: Official Journal of the International Federation of Clinical Neurophysiology, 2024. 160: 111-112.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Full format: [MUNE,STAIR]=StairFit(data,D1,D2,inc,th);
%Suggested format : MUNE=StairFit(data,D1,D2,inc);
%INPUT:
%data,
% A N*2 matrix, the first column is the stimulation intensity (mA)(sorted from smallest to largest, or vice versa),
% the second column is the correspoding CMAP amplitude (mV). 
%D1, 
% The initial value of MUNE, a reasonable positive integer smaller than the estimated MUNE.(The easiest way is to just set D1 to 1)          
%D2, 
% The number of parallel operations per run (recommended to be set as the number of CPU cores).
%inc, 
% The increment (resolution). For CMAP scan with potentially large MUNE, it is recommended to set inc to 2 (or even greater) for efficiency.
%th,
% The fitting error threshold, Defined as 3 times the standard deviation of baseline noise.
% The noise level is generally around 0.005 (mV), so a default value of th is 0.015
% (mV). It is not recommended to change the threshold for robustness.
%%%%%%%%%%%%%%%%%%
%StairFit will simultaneously explore D2 models (D1-1+inc : inc : D1-1+D2*inc).
%If the threshold of fitting error is not reached, StairFit will automaticly set D1 to D1+D2*inc and explore the next set of D2 models until the threshold is reached.
%%%%%%%%%%%%%%%%%%
%OUTPUT:
%MUNE, a struct containing information about the model corresponding to MUNE.
% MUNE.number, the MUNE.
% MUNE.stair, the heights of stairs,the MU amplitudes can be obtained by "Amplitude=diff(MUNE.stair);".
% MUNE.point, the activation thresholds of MUs.
% MUNE.curve, the fitted CMAP scan.
% MUNE.runtime, the running time of StairFit.
%STAIR, a struct containing information about the fitted model of each motor unit number. 
% STAIR.height,the heights of stairs.
% STAIR.error, the fitting errors.


close all;
if nargin<5
th=0.015;
end
if data(1,1)>data(end,1)
    data=flipud(data);
end
label=0;
ts=clock;
STAIR.height=[];STAIR.error=[];
y=data(:,2);
while label==0
    parfor i=1:D2
        [stair{i},E(i)]=MUNE_STAIR_par(y,i*inc+D1);
        stair{i}=sort(stair{i});
    end
    STAIR.height=[STAIR.height,stair];
    STAIR.error=[STAIR.error,E];
    IN=find(E<th,1);
    label=1;
    if isempty(IN)
        label=0;
        D1=D2*inc+D1;
    end
end

MUNE.stair=stair{IN};figure;
[MUNE.point,MUNE.curve]=STAIR_estimate_par(MUNE.stair,data(:,1),data(:,2));
MUNE.number=length(MUNE.point);
title(['MUNE=',num2str(MUNE.number)]);
te=clock;
MUNE.runtime = etime(te,ts);
end

function [R,fit]=MUNE_STAIR_par(y,D)
        [~,initial_population] = kmeans(y,D);
        initial_population=sort(initial_population);
BL=ones(size(initial_population))*min(y);
BU=ones(size(initial_population))*max(y);
options1= optimoptions('patternsearch','PollOrderAlgorithm','Success','MaxIterations',10000, 'MaxFunctionEvaluations',1e10,'PollMethod','GSSPositiveBasis2N','UseCompletePoll',false,'AccelerateMesh',true,'MeshTolerance',1e-7,'UseParallel',false,'Display','off');%,'PlotFcn',{@psplotbestx,@psplotbestf,@psplotfuncount,@psplotmeshsize},'PlotInterval',1,'ConstraintTolerance',1e-10);%,'UseCompleteSearch',true,'SearchFcn',{@searchneldermead,1});%,'DisPlay','Diagnose');
[R,~]= patternsearch(@(r) fitness_step_y(r,y),initial_population,[],[],[],[],BL,BU,[],options1);
[D,~]= pdist2(R,y,'euclidean','Smallest',1);
fit=mean(D);
end

function [point,fun]=STAIR_estimate_par(stair,x,y)
r_y=max(y);stair=stair/r_y;
y=y/r_y;
r_x=max(x);x=x/r_x;
initial_population=zeros(length(stair)-1,1);
for i=1:length(initial_population)
    initial_population(i)=x(find(y>stair(i),1));
end

BL=x(3)*ones(size(initial_population));
BU=x(end-1)*ones(size(initial_population));
options1= optimoptions('patternsearch','PollOrderAlgorithm','Success','MaxIterations',10000, 'MaxFunctionEvaluations',1e10,'PollMethod','GSSPositiveBasis2N','UseCompletePoll',true,'AccelerateMesh',true,'MeshTolerance',1e-4,'UseParallel',true,'Display','off');%,'PlotFcn',{@psplotbestx,@psplotbestf,@psplotfuncount,@psplotmeshsize},'PlotInterval',1,'ConstraintTolerance',1e-10);%,'UseCompleteSearch',true,'SearchFcn',{@searchneldermead,1});%,'DisPlay','Diagnose');
[point,~]= patternsearch(@(r) fitness_step_x(r,stair,x,y),initial_population,[],[],[],[],BL,BU,[],options1);
point=sort(point);
stair=stair*r_y;
x=x*r_x;
y=y*r_y;
point=point*r_x;

fun=[x,stair(1)*ones(size(x))];
for i=1:length(point)
    fun(x>point(i),2)=stair(i+1);
end
scatter(x,y,'.');hold on
scatter(fun(:,1),fun(:,2),'.','linewidth',3);
end

function [fit]=fitness_step_x(point,stair,x,y)
point=sort(point);
fun=[x,stair(1)*ones(size(x))];
for i=1:length(point)
    fun(x>point(i),2)=stair(i+1);
end
D = pdist2(fun,[x,y],@weight_cityblock,'Smallest',1);
fit=sum(D);
end

function [fit]=fitness_step_y(stair,y)
[D,~] = pdist2(stair,y,'euclidean','Smallest',1);
fit=mean(D);
end

function D2 = weight_cityblock(ZI,ZJ)
M=ones(size(ZJ,1),1)*ZI-ZJ;
M(:,1)=M(:,1)*0.1;
D2=sum(abs(M),2);

end
