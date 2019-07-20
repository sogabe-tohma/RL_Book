function Bandit_ucb1_main() 
    global iterate;
    iterate=0;
    Nexp=1;
    Npulls=150;    
    global summation;
    summation=zeros(1,Npulls);
    figure(1);
    set(gcf,'Position',[0,0,900,900])
    avg_outcome_eps0p2=zeros(1,Npulls);
    for n =1:Nexp     
       bandit=Bandit;
       bandit.init(4,0.0);
       avg_outcome_eps0p2=avg_outcome_eps0p2+run_experiment(bandit,Npulls);
    end
    clf
    figure(1);
    set(gcf,'Position',[0,0,500,500])
    avg_outcome_eps0p2= avg_outcome_eps0p2/Nexp;
    plot(avg_outcome_eps0p2,'b-o')
    legend({'UCB1'},'Location','northwest')
    
end

function history=run_experiment(bandit,Npulls)
    history=[];
    color=['b','g','r','c','m','y','k','g','m','c'];
    global summation;
    global iterate;
    iterate_1=0;
    iterate_1=iterate_1+1;
    for a=1:Npulls
        iterate=iterate+1;
        action=bandit.choose_eps_greedy(iterate);
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
        scatter(Npulls*(iterate_1-1)+a,summation(a)/(1),'filled',color(4));
        pause(0.01);
    end
    hold off;
       
end
function clor= set_color(l)
    if l<0
        clor="r";
    else
        clor="b";
    end
   
end

