function fig2_16()
row = 3;
col = 4;

start.row = 3;
start.col = 4;
start;

goal.row= 1;
goal.col = 1;
goal;

N_goal.row=2;
N_goal.col=1;
N_goal;

barrier.row=2;
barrier.col=3;
barrier;

rewards=zeros(row,col);
rewards(barrier.row,barrier.col)=0;
rewards(goal.row,goal.col)=100;
rewards(N_goal.row,N_goal.col)=-100;
v=random_v(row,col,4,barrier,goal,N_goal);
disp("This is random v")
disp(v)
policy=random_policy(row,col,4,barrier,goal,N_goal);
disp("This is random policy")
print(policy,row,col)
small_enough=0.001;
gamma = 0.9;
disp("This is reward")
print(rewards,row,col)
    while (1)
        biggest_change=0;
         for r=1:row
            for c=1:col
                old_v=v(r,c);
                if ~((barrier.row==r&& barrier.col==c)||(goal.row==r&&...
                                    goal.col==c)||(N_goal.row==r&& N_goal.col==c))
                     new_v=-0.000001;
                     for a =1:4
                         state=[r,c];
                         n_state=getNext(state, a,row,col);
                         reward=rewards(n_state(1),n_state(2));
                         n_v=reward+(gamma*v(n_state(1),n_state(2)));
                         if n_v>new_v
                             new_v=n_v;
                         end
                     end
                     v(r,c)=new_v;
                     biggest_change=max(biggest_change,abs(old_v-v(r,c)));
                end

            end
         end
         if biggest_change<small_enough
                   break;
         end
    end
    for r=1:row
        for c=1:col
            if ~((barrier.row==r&& barrier.col==c)||(goal.row==r&&...
                            goal.col==c)||(N_goal.row==r&& N_goal.col==c))
                 best_a=0.0;
                 best_value=-0.000001;
                 for a =1:4
                     state=[r,c];
                     n_state=getNext(state, a,row,col);
                     reward=rewards(n_state(1),n_state(2));
                     n_v=reward+(gamma*v(n_state(1),n_state(2)));
                     if n_v>best_value
                         best_value=n_v;
                         best_a=a;
                     end
                 end
                 policy(r,c)=best_a;
            end
        end
    end
    disp("final values")
    print(v,row,col)
    disp("final policy")
    print(policy,row,col)
end


function c= make_action(row,col,action_size,barrier,goal,N_goal)
    v=zeros(row,col,action_size);
    action=[1,1,1,1];
    for r=1:row
        for c=1:col           
            A=1;
            while (A<=action_size)
                recent=[r,c];
                if A==1
                    recent(1)=recent(1)-1;
                    recent(2)=recent(2);
                elseif A==2
                    recent(1)=recent(1)+1;
                    recent(2)=recent(2);
                elseif A==3
                    recent(1)=recent(1);
                    recent(2)=recent(2)+1;                   
                elseif A==4
                    recent(1)= recent(1);
                    recent(2)=recent(2)-1;
                end
                if (recent(1)>0 && recent(1)<=row)
                    if (recent(2)>0 && recent(2)<=col)
                        if ~((barrier.row==recent(1)&& barrier.col==recent(2))||(goal.row==recent(1)&&...
                                goal.col==recent(2))||(N_goal.row==recent(1)&& N_goal.col==recent(2)))
                            v(recent(1),recent(2),A)=action(A);
                        end
                    end
                end
                A=A+1;
            end
        end
    end 
    c=v;
end
                    
function pos = getNext(now, action,row,col)
    current.row=now(1);
    current.col=now(2);
    last=current;
    pos.row=now(1);
    pos.col=now(2);
    switch action
        case 3 %east
            pos.col= current.col + 1;
        case 2 %south
            pos.row= current.row + 1;
        case 4 %west
            pos.col= current.col - 1;
        case 1 %north
            pos.row= current.row - 1;
    end
    if (pos.col==3 && pos.row==2)
        pos.row=last.row;
        pos.col=last.col;
    end
    if(pos.col <= 0)
        pos.col = 1; end
    if(pos.col > col)
        pos.col = col; end
    if(pos.row <= 0)
        pos.row = 1; end
    if(pos.row > row)
        pos.row = row; end
    pos=[pos.row,pos.col];
end
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
function V=random_policy(row,col,action,barrier,goal,N_goal)
    policy=zeros(row,col);
    v_action=make_action(row,col,action,barrier,goal,N_goal);
    for r=1:row
        for c=1:col 
            if sum(v_action(r,c,:))>=1
                policy(r,c)=randi([1,4],1);
            else
                 policy(r,c)=0;
            end
        end
    end
    V=policy;
end
function print(v,row,col)
     for r=1:row
         disp("---------------------------------")
         d=zeros(4);
        for c=1:col
            d(c)=v(r,c);
        end
        fprintf("%5d   %5d    %5d    %5d\n",round(d(1)),round(d(2)),round(d(3)),round(d(4)))
     end 
     disp("---------------------------------")   
end

