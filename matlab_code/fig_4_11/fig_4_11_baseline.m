function fig_4_11_baseline()  
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
max_episode=2000;
return_Q=zeros(row,col,4,2);
policy_value=zeros(row,col);
policy1=zeros(row,col);
policy=random_policy(row,col,4,barrier,goal,N_goal);
Q= zeros(row,col,4);
delta=[];
g_world=grid();
g_world.initialization(4,3,[3,4],[1,1],[2,3],[2,1]);
t1=1;
t2=1;
Alpha=0.01;
gamma=0.9;
alpha=0.01;
delta=[];
for times=1:max_episode
    biggest_change=0;
    state_G=play_game(policy);
    len_stateG=length(state_G);
    seen_state_action=zeros(row,col,4);
    for tin=1:len_stateG
        sag=state_G(tin,:);
        s=sag{1};
        a=sag{2};
        G=sag{3};
        if seen_state_action(s(1),s(2),a)==0
           old_q=Q(s(1),s(2),a);
           return_Q(s(1),s(2),a,1)=return_Q(s(1),s(2),a,1)+1;
           return_Q(s(1),s(2),a,2)=return_Q(s(1),s(2),a,2)+G;
           Q(s(1),s(2),a)=return_Q(s(1),s(2),a,2)/return_Q(s(1),s(2),a,1);
           Q(s(1),s(2),a)=Q(s(1),s(2),a)-mean(Q(s(1),s(2)));
           biggest_change=max(biggest_change,abs(old_q-Q(s(1),s(2),a)));
           seen_state_action(s(1),s(2),a)=G;
        else
            do=0;
        end  
    delta=[delta,biggest_change];
    end
    for r=1:row
        for c=1:col
            if policy(r,c)~=0
            [val,in] =max(Q(r,c,:));  
            policy(r,c)=in;
            policy_value(r,c)=val;
            end
        end
   end
end
plot(delta)
pause(5)
disp("this is optimal policy")
draw_a_grid(policy, row, col,start,goal,barrier,N_goal)
disp(policy_value);
end


