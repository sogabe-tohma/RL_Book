function [state,count]=play_game(policy,count)
    gamma = 0.9;
    width=4;
    height=3;
    start=[3,4];
    goal=[1,1];
    N_goal=[2,1];
    barrier=[2,3];
    game=grid(); 
    game.initialization(width,height,start,goal,barrier,N_goal);
    a=policy(game.start(1),game.start(2)); 
    action=random_action(a);
    count(game.start(1),game.start(2),action)=count(game.start(1),game.start(2),action)+1;
    r=0;
    all_SAR=[];
    all_SAR=[all_SAR;{start,action,r}];
    while (1)
        out=game.step(action);
        n_state=out{1};
        r=out{2};
        done=out{3};
        if done==1
             all_SAR=[all_SAR;{n_state,0,r}];
            break;
        end
        a=policy(game.current(1),game.current(2));
        action=random_action(a);
        all_SAR=[all_SAR;{n_state,action,r}];
        count( n_state(1), n_state(2),action)=count(  n_state(1), n_state(2),action)+1;
    end
    t=length(all_SAR);
    G=0;
    t_SAG=[];
    for steps=1:t
        r_out=all_SAR((t-(steps-1)),:);
        states=r_out{1};
        actions=r_out{2};
        reward=r_out{3};
        if steps==1
            do=0;
        else
            t_SAG=[t_SAG;{states,actions,G}];
        end
        G=round((reward+gamma*G),3);
        
    end
    state=t_SAG;  
end
function action = random_action(a)
    p=rand(1);
    eps=0.15;
    if p>(1-eps)
        action=a;
    else
        action=randi([1,4],1);
    end
end