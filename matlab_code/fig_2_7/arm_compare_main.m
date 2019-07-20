function arm_compare_main() 
    Nexp=50;
    Npulls=3000; 
    global iterate;
    iterate=0;
    global summation;
    summation=zeros(1,Npulls);  
    avg_outcome_eps0a1=zeros(1,Npulls);
    avg_outcome_eps0a5=zeros(1,Npulls);
    avg_outcome_eps0a10=zeros(1,Npulls);
    avg_outcome_eps0a100=zeros(1,Npulls);
    avg_outcome_eps0a10000=zeros(1,Npulls);
   
    for n =1:Nexp
       bandit=Bandit;
       bandit.init(1);
       avg_outcome_eps0a1=avg_outcome_eps0a1+run_experiment(bandit,Npulls,0.05);
       bandit=Bandit;
       bandit.init(5);
       avg_outcome_eps0a5=avg_outcome_eps0a5+run_experiment(bandit,Npulls,0.05);
       bandit=Bandit;
       bandit.init(10);
       avg_outcome_eps0a10=avg_outcome_eps0a10+run_experiment(bandit,Npulls,0.05);
       bandit=Bandit;
       bandit.init(100);
       avg_outcome_eps0a100=avg_outcome_eps0a100+run_experiment(bandit,Npulls,0.05);
       bandit=Bandit;
       bandit.init(10000);
       avg_outcome_eps0a10000=avg_outcome_eps0a10000+run_experiment(bandit,Npulls,0.05);       
    end
    avg_outcome_eps0a1= avg_outcome_eps0a1/Nexp;
    avg_outcome_eps0a5= avg_outcome_eps0a5/Nexp;
    avg_outcome_eps0a10= avg_outcome_eps0a10/Nexp;
    avg_outcome_eps0a100= avg_outcome_eps0a100/Nexp;
    avg_outcome_eps0a10000= avg_outcome_eps0a10000/Nexp;
    plot(avg_outcome_eps0a1)
    hold on
    plot(avg_outcome_eps0a5)
    plot(avg_outcome_eps0a10)
    plot(avg_outcome_eps0a100)
    plot(avg_outcome_eps0a10000)
    legend({'Arm-1','Arm-5','Arm-10','Arm-100','Arm-10000',},'Location','southeast')
 
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

