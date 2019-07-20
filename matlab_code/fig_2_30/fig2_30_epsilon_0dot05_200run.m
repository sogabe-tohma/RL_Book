function fig2_30_epsilon_0dot05_200run()
rng(321,'v4')
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
max_episode=200;
policy=zeros(row,col);
value=zeros(row,col);
global Q;
Q= zeros(row,col,4);
delta=[];
ts=1;
gamma=0.9;
alpha =0.1;
eps=0.05;
game=grid(); 
game.initialization(4,3,[3,4],[1,1],[2,3],[2,1]);
for times=1:max_episode
    if rem(times,100)==0
        ts=ts+0.01;
    end
    biggest_change=0;
    a=randi([1,4],1);
    s=[0,0];
    s=game.reset();
    done=0;
    while ~(done)
        a=random_action(a,eps);
        next=game.step(a);
        s2=next{1};
        r=next{2};
        done=next{3};
        [max_q,~]= max(Q(s2(1),s2(2),:));
        old_qsa=Q(s(1),s(2),a);
        Q(s(1),s(2),a)=Q(s(1),s(2),a)+alpha*(r+gamma*max_q-Q(s(1),s(2),a));
        biggest_change=max(biggest_change,abs(old_qsa-Q(s(1),s(2),a)));
        s=s2;
    end
    delta=[delta; biggest_change];
    for R=1:row
        for c=1:col           
            [val,in] =max(Q(R,c,:));  
            policy(R,c)=in;
            value(R,c)=val;
        end
    end  
end
plot(delta)
% disp("this is value")
% disp(value)
% disp("this is optimal policy")
% draw_a_grid(policy, row, col,start,goal,barrier,N_goal)
end
function action = random_action(a,eps)
    action=NaN;
    p=randn(1);
    if p<(1-eps)
        action=a;
    else
        action=randsample([4,1,2,3],1);
    end
end

