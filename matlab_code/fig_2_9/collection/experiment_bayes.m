
function history=experiment_bayes(bandit,Npulls)
    history=[];
    global summation;
    global iterate;
    iterate=iterate+1;
   
    
    for a=1:Npulls
        action=bandit.choose_eps_greedy();
        R=bandit.get_reward(action);
        bandit.update_est(action,R);
        history=[history,R];
       
    end

end
