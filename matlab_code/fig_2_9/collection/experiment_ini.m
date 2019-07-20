
function history=experiment_ini(bandit,Npulls)
    history=[];
    global summation;
    global iterate  ;
    iterate=iterate+1;
    %color=['m','g','c','g','m','c','g','m','c','k'];

    
    for a=1:Npulls
        action=bandit.choose_eps_greedy();
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
