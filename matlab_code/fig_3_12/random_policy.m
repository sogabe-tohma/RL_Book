function V=random_policy(row,col,action,barrier,goal,N_goal)
    policy=zeros(row,col);
    v_action=make_action(row,col,action,barrier,goal,N_goal);
    for r=1:row
        for c=1:col 
            if sum(v_action(r,c,:))>=1
                x=find(v_action(r,c,:));
                policy(r,c)=randsample(x,1);
            else
                 policy(r,c)=0;
            end
        end
    end
    V=policy;
end
