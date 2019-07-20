function bandit_10_epsilon_0() 
    global iterate;
    iterate=0;
    global Nexp;
    Nexp=3;
    global Npulls;
    Npulls=150;    
    global summation;
    summation=zeros(1,Npulls);
    figure(1);
    set(gcf,'Position',[0,0,500,1200])
    
    avg_outcome_eps0p3=zeros(1,Npulls);
   
    for n =1:Nexp 
        clf()
       bandit=Bandit;
       bandit.init(10,10);
       avg_outcome_eps0p3=avg_outcome_eps0p3+run_experiment(bandit,Npulls,0);
    end
   clf
   figure(1);
   set(gcf,'Position',[0,0,500,500])
   plot(avg_outcome_eps0p3)
   legend({'eps=0'},'Location','southeast')
end

function history=run_experiment(bandit,Npulls,epsilon)
    history=[];
    color=['m','m','m','g','g','g','c','c','c','k'];
    global summation;
    global iterate;
    iterate=iterate+1;
    
    for a=1:Npulls
        action=bandit.choose_eps_greedy(epsilon);
        R=bandit.get_reward(action);
        bandit.update_est(action,R);
        history=[history,R];        
        hold on 
        subplot(311)        
        scatter(a,bandit.update_est(action,R),'filled',color(action));
        pause(0.01);
        hold on
        subplot(312)
        scatter(a,bandit.get_reward(action),'filled',color(action));
        pause(0.01);
        hold on  
        subplot(313)  
        summation(a)=summation(a)+history(a);  
        scatter(Npulls*(iterate-1)+a,summation(a)/(1),'filled',set_color(summation(a)));
        pause(0.01);
    end
    hold off
        
end
function clor= set_color(l)
    if l<0
        clor='r';
    else
        clor='b';
    end
   
end

