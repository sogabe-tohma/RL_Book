function fig2_27()
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
max_episode=5000;
update_count_sa=ones(row,col,4);
update_count=zeros(row,col);
policy=zeros(row,col);
value=zeros(row,col);
Q= zeros(row,col,4);
delta=[];
ts=1;
gamma=0.9;
alpha =0.1;
game=grid(); 
game.initialization(4,3,[3,4],[1,1],[2,3],[2,1]);
for times=1:max_episode
    Q(1,1,:)=0;
    Q(2,1,:)=0;
    Q(2,3,:)=0;
    if rem(times,100)==0
        ts=ts+0.01;
    end
    biggest_change=0;
    a=randi([1,4],1);
    s=game.reset();
    done=0;
    while ~(done)
        next=game.step(a);
        s2=next{1};
        r=next{2};
        done=next{3};
        [~,a2]= max(Q(s2(1),s2(2),:));
        a2=random_action(a2,ts);
        update_count_sa(s(1),s(2),a) = update_count_sa(s(1),s(2),a)+0.005;
        old_qsa=Q(s(1),s(2),a);
        Q(s(1),s(2),a)=Q(s(1),s(2),a)+alpha*(r+(gamma*Q(s2(1),s2(2),a2)-Q(s(1),s(2),a)));
        biggest_change=max(biggest_change,abs(old_qsa-Q(s(1),s(2),a)));
        update_count(s(1),s(2))= update_count(s(1),s(2))+1;
        a=a2;
        s=s2;
    end
    
    for R=1:row
        for c=1:col           
            [val,in] =max(Q(R,c,:));  
            policy(R,c)=in;
            value(R,c)=val;
        end
    end  
end
disp("this is value")
disp(value)
disp("this is optimal policy")
draw_a_grid(policy, row, col,start,goal,barrier,N_goal)
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

