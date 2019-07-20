function cumulative_average=run_experiment_ucb(m1,m2,m3,N)
bandits{m1}=ucb1;bandits{m2}=ucb1;bandits{m3}=ucb1;
bandits{m1}.init(m1);
bandits{m2}.init(m2);
bandits{m3}.init(m3);
data=[];
for a=1:N
    f=length(bandits);
    max_f=zeros(1,f);
    for c=1:length(bandits)
        max_f(c)=ucb(bandits{c}.mean,c+1,bandits{c}.N);
    end
    [~,id]=max(max_f);
    d=bandits{id}.pull();
    bandits{id}.update(d);
    data=[data,d];
end
cumulative_average=cumsum(data)./linspace(1,N,N);
% plot(cumulative_average)
% hold on
% plot(ones(1,N).*m1)
% plot(ones(1,N).*m2)
% plot(ones(1,N).*m3)
% set(gca, 'XScale', 'log')

end

function value=ucb(mean,n,nj)
    value=mean+sqrt(2*log(n)./(nj+1e-2));
end
