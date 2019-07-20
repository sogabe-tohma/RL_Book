function v=random_v(row,col,action,barrier,goal,N_goal)
    random_v=zeros(row,col);
    v_action=make_action(row,col,action,barrier,goal,N_goal);
    for r=1:row
        for c=1:col 
            if sum(v_action(r,c,:))>=1
                random_v(r,c)=round(rand(1),1);
            else
                 random_v(r,c)=0;
            end
        end
    end
    v=random_v;
end

