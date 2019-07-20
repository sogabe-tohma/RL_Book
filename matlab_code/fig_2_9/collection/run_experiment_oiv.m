
function cumulative_average=run_experiment_oiv(m1,m2,m3,N)
%BANDIT_MAIN Summary of this function1 goes here
%   Detailed explanation goes here
bandits{m1}=Bandit_initial_value;bandits{m2}=Bandit_initial_value;bandits{m3}=Bandit_initial_value;
data=[];
for a=1:N
    f=length(bandits);
    max_f=zeros(f,1);
    for c=1:length(bandits)
        max_f(c)=bandits{c}.mean;
    end
    [~,id]=max(max_f);
    d=bandits{id}.pull(id);
    bandits{id}.update(d) 
    data=[data,d];    
end
cumulative_average=cumsum(data)./linspace(1,N,N);
end

