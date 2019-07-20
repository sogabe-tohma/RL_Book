function epsilon_compare_main() 
    Nexp=50;
    Npulls=3000; 
    global iterate;
    iterate=0;
    global summation;
    summation=zeros(1,Npulls);  
    avg_outcome_eps0p0=zeros(1,Npulls);
    avg_outcome_eps0p05=zeros(1,Npulls);
    avg_outcome_eps0p1=zeros(1,Npulls);
    avg_outcome_eps0p4=zeros(1,Npulls);
    avg_outcome_eps0p8=zeros(1,Npulls);
   
    for n =1:Nexp
       bandit=Bandit;
       bandit.init(10);
       avg_outcome_eps0p0=avg_outcome_eps0p0+run_experiment(bandit,Npulls,0.0);
       bandit=Bandit;
       bandit.init(10);
       avg_outcome_eps0p05=avg_outcome_eps0p05+run_experiment(bandit,Npulls,0.05);
       bandit=Bandit;
       bandit.init(10);
       avg_outcome_eps0p1=avg_outcome_eps0p1+run_experiment(bandit,Npulls,0.1);
       bandit=Bandit;
       bandit.init(10);
       avg_outcome_eps0p4=avg_outcome_eps0p4+run_experiment(bandit,Npulls,0.4);
       bandit=Bandit;
       bandit.init(10);
       avg_outcome_eps0p8=avg_outcome_eps0p8+run_experiment(bandit,Npulls,0.8);       
    end
    avg_outcome_eps0p0= avg_outcome_eps0p0/Nexp;
    avg_outcome_eps0p05= avg_outcome_eps0p05/Nexp;
    avg_outcome_eps0p1= avg_outcome_eps0p1/Nexp;
    avg_outcome_eps0p4= avg_outcome_eps0p4/Nexp;
    avg_outcome_eps0p8= avg_outcome_eps0p8/Nexp;
    plot(avg_outcome_eps0p0)
    hold on
    plot(avg_outcome_eps0p05)
    plot(avg_outcome_eps0p1)
    plot(avg_outcome_eps0p4)
    plot(avg_outcome_eps0p8)
    legend({'eps-0.0','eps-0.05','eps-0.1','eps-0.4','eps-0.8',},'Location','southeast')
 
end

function history=run_experiment(bandit,Npulls,epsilon)
    history=[];
    global summation;
    global iterate  ;
    iterate=iterate+1;
   %color=['m','g','c','g','m','c','g','m','c','k'];

    for a=1:Npulls
       
        action=bandit.choose_eps_greedy(epsilon);
        R=bandit.get_reward(action);       
        bandit.update_est(action,R);
        history=[history,R];
        %{
         hold on 
        subplot(311)        
        scatter(a,bandit.update_est(action,R),'filled',color(action));
        pause(0.001);
        hold on
        subplot(312)
        scatter(a,bandit.get_reward(action),'filled',color(action));
      
        hold on  
        subplot(313)  
        summation(a)=summation(a)+history(a);  
        scatter(Npulls*(iterate-1)+a,summation(a)/(1),'filled',set_color(summation(a)));
        pause(0.001);
       %}
       
    end
       %hold off;  
end
% function clor= set_color(l)
%     if l<=0
%         clor='r';
%     else
%         clor='b';
%     end
%    
% end

