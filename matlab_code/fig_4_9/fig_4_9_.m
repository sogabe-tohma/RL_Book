function fig_4_2_()  
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
max_episode=500;
policy_value=zeros(row,col);
pi=zeros(row,col,4);
policy=random_policy(row,col,4,barrier,goal,N_goal);
delta=[];
g_world=grid();
g_world.initialization(4,3,[3,4],[1,1],[2,3],[2,1]);
ALPHA=0.01;
for times=1:max_episode
    pi(goal.row,goal.col,:)=0;
    pi(N_goal.row,N_goal.col,:)=0;
    
    pi(barrier.row,barrier.col,:)=0;
    biggest_change=0;
    count=zeros(row,col,4);
    [state_G,r_count]=play_game(policy,count);
    len_stateG=length(state_G);
    seen_state_action=zeros(row,col,4);
    for tin=1:len_stateG
        sag=state_G(tin,:);
        s=sag{1};
        a=sag{2};
        G=sag{3};
        gt=[0,0,0,0];
        gt(a)=G;
        data=zeros(1,4);
        baseline=0;
        if seen_state_action(s(1),s(2),a)==0
           old_pi=pi(s(1),s(2),a);
           for act=1:4
               delt=r_count(s(1),s(2),act)-sum(r_count(s(1),s(2),:))*soft_max(pi(s(1),s(2),act),pi(s(1),s(2),:))/sum(r_count(:));
               data(1,act)=delt;  
           end
           for ac=1:4
               pi(s(1),s(2),ac)=pi(s(1),s(2),ac)+(ALPHA* data(1,ac))*(gt(ac)-baseline);
           end
           biggest_change=max(biggest_change,abs(old_pi-pi(s(1),s(2),a)));
           seen_state_action(s(1),s(2),a)=G;
        else
            do=0;
        end  
    delta=[delta,biggest_change];
    end
   if times==1||times==10 ||times==100 ||times==500
       fprintf("this is result of episode %d \n",times);
       fprintf("                   UP      DOWN    LEFT    RIGHT\n");
        for r=1:row
            for c=1:col
                if policy(r,c)~=0
                    prob=zeros(1,4);
                    for items=1:4
                        s_max=soft_max(pi(r,c,items),pi(r,c,:));  
                        prob(1,items)=s_max;
                    end
                   fprintf("place ( %d, %d ) : %5.2f    %5.2f   %5.2f   %5.2f \n",r,c,prob(1,1),prob(1,2),prob(1,3),prob(1,4));
                end
                
            end
        end
   end
   for r=1:row
        for c=1:col
            if policy(r,c)~=0
                [val,in] =max(pi(r,c,:));  
                policy(r,c)=in;
                policy_value(r,c)=val;
            end
        end
    end
end
draw_a_grid(policy, row, col,start,goal,barrier,N_goal)
end

function y =soft_max(x,d_x)
    sum1=0;
    beta=100;
    max1=0;
    if max1<max(d_x)
        max1=max(d_x);
    end
    exp1=exp(beta*(x-max1));
    for t_=1:length(d_x)
        sum1=sum1+exp(beta*(d_x(t_)-max1));
    end
    y=(exp1/sum1);
end
