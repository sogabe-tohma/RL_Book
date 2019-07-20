function fig_2_9_main()
    addpath('./collection/')
    Nexp=50;
    Npulls=3000; 
    global iterate;
    iterate=0;
    global summation;
    summation=zeros(1,Npulls);
   
    avg_outcome_eps0p0=zeros(1,Npulls);
    avg_outcome_eps0p1=zeros(1,Npulls);
    avg_outcome_eps0p2=zeros(1,Npulls);
    avg_outcome_eps0p3=zeros(1,Npulls);
    
    for n =1:Nexp     
       bandit=Bandit;
       bandit.init(10);
       avg_outcome_eps0p2=avg_outcome_eps0p2+run_experiment(bandit,Npulls);
       bandit1=Bandit_BA;
       bandit1.init(10);
       avg_outcome_eps0p3=avg_outcome_eps0p3+experiment_bayes(bandit1,Npulls);
       bandit2=BanditE;
       bandit2.init(10);
       avg_outcome_eps0p1=avg_outcome_eps0p1+experiment_eps(bandit2,Npulls);
       bandit3=BanditI;
       bandit3.init(10);
       avg_outcome_eps0p0=avg_outcome_eps0p0+experiment_ini(bandit3,Npulls);
       
    end
    avg_outcome_eps0p0= avg_outcome_eps0p0/Nexp;
    avg_outcome_eps0p1= avg_outcome_eps0p1/Nexp;
    avg_outcome_eps0p2= avg_outcome_eps0p2/Nexp;
    avg_outcome_eps0p3= avg_outcome_eps0p3/Nexp;
    plot(avg_outcome_eps0p1,'LineWidth',1)
    hold on
    plot(avg_outcome_eps0p0,'LineWidth',1)
    plot(avg_outcome_eps0p2,'LineWidth',1)
    plot(avg_outcome_eps0p3,'LineWidth',1)
    legend({'e-greedy,eps=0.05','optimal inital value','UCB1','Bayes sampling'},'Location','southeast')
 
end

function history=run_experiment(bandit,Npulls)
    history=[];
    global summation;
    global iterate  ;
    iterate=0;
    %color=['m','g','c','g','m','c','g','m','c','k'];

    
    for a=1:Npulls
        iterate=iterate+1;
        action=bandit.choose_eps_greedy(iterate);
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
        pause(0.01);
        %}
       
    end
      % hold off;  
end
function clor= set_color(l)
    if l<=0
        clor='r';
    else
        clor='b';
    end
   
end

