function fig_3_12_q_learning()  
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
max_episode=100000;
return_Q=zeros(row,col,4,2);
policy_value=zeros(row,col);
policy1=zeros(row,col);
Q= zeros(row,col,4);
delta=[];
model=Model1();
g_world=grid();
policy=random_policy(row,col,4,barrier,goal,N_goal);
g_world.initialization(4,3,[3,4],[1,1],[2,3],[2,1]);
t1=1;
t2=1;
Alpha=0.01;
gamma=0.9;
alpha=0.01;
for times=1:max_episode
    if rem(times,100)==0
        t1=t1+0.01;
        t2=t2+0.01;
    end
    if rem(times,1000)==0 
        alpha=Alpha/t2;
    end
    qs=g_world.reset();
    qa=getqs(model,g_world.current);
    [~,a]=max(qa);
    biggest_change=0;
    done=0;
    l_state=qs;
    while ~(done)
        action1= random_action(a,t1);
        out=g_world.step(action1);
        n_state=out{1};
        r=out{2};
        done=out{3};
        old_theta=model.theta;
        if done
            model.theta=model.theta+(alpha*(r- model.predict(l_state,action1))*model.grad(l_state,action1));
        else
            qa2=getqs(model,n_state);
            [qmax_s2a2,a]=max(qa2);
            action= random_action(a,t1);
            
            model.theta=model.theta+(alpha*(r+ gamma*qmax_s2a2-...
            model.predict(l_state,action1))*model.grad(l_state,action1));
            l_state=n_state;
           
        end
        biggest_change=max(biggest_change,abs(sum(model.theta-old_theta)));
    end
    delta=[delta; biggest_change];
   for r=1:row
    for c=1:col
        if policy(r,c)~=0
            Qs=getqs(model,[r,c]);
            [val,in] =max(Qs);  
            policy1(r,c)=in;
            policy_value(r,c)=val;
        end
    end
   end
end
plot(delta)
pause(1)
disp("this is optimal policy")
draw_a_grid(policy1, row, col,start,goal,barrier,N_goal)
disp(policy_value);
end
function xs=getqs(model,s)
    qsa=[];
    for a =1:4
     tt= model.predict(s,a);
     qsa=[qsa,tt];
    end
    xs=qsa;
end

function action = random_action(a,t)
    p=rand(1);
    eps=0.5/t;
    if p<(1-eps)
        action=a;
    else
        action=randi([1,4],1);
    end
end


