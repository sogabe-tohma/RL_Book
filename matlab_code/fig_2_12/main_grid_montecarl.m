function main_grid_montecarl()
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
gamma = 1;
rewards=zeros(row,col);
rewards(barrier.row,barrier.col)=0;
rewards(goal.row,goal.col)=100;
rewards(N_goal.row,N_goal.col)=-100;
iteration = 1;
count=0;
iterate=0;
v_action=make_action(row,col,4,barrier,goal,N_goal);
disp(v_action)
color=['m','g','c','r','y','b','k','m','c','k','m','g','c'];
v=zeros(row,col);
figure(1);
set(gcf,'Position',[0,0,500,500])
while (count<5)
    count=count+1;
    v(goal.row,goal.col)=0;
    v(N_goal.row,N_goal.col)=0;
    v(barrier.row,barrier.col)=0;
    b_change=0;
    for r=1:row
        for c=1:col  
            if ~((barrier.row==r&& barrier.col==c)||(goal.row==r&&...
                                goal.col==c)||(N_goal.row==r&& N_goal.col==c))
                old_v=v(r,c);
                iterate=iterate+1;
                new_v=0;
                pa=1/sum(v_action(r,c,:));
                a=v_action(r,c,:);
                b=find(a);
                for A=1:length(b)
                    v(goal.row,goal.col)=0;
                    v(N_goal.row,N_goal.col)=0;
                    next_state=getNext([r,c],b(A),row,col);
                    disp( next_state)
                    reward=rewards(next_state(1),next_state(2));
                    %disp(reward)
                    new_v=new_v+(pa*1.0*(reward+(gamma*(v(next_state(1),next_state(2))))));
                    %fprintf('new_v: %d\n', new_v)
                end
                hold on
                scatter(iterate,v(3,1),'filled',color(1));
                scatter(iterate,v(1,2),'filled',color(2));
                scatter(iterate,v(2,2),'filled',color(3));
                scatter(iterate,v(3,2),'filled',color(4));
                scatter(iterate,v(1,3),'filled',color(5));
                scatter(iterate,v(3,3),'filled',color(6));
                scatter(iterate,v(1,4),'filled',color(7));
                scatter(iterate,v(2,4),'filled',color(8));
                scatter(iterate,v(3,4),'filled',color(9));
                v(r,c)=new_v;
                %fprintf('Episode: %d\n', iterate)
                b_change=max(b_change,old_v-v(r,c));
            end
        end
        v(goal.row,goal.col)=0;
        v(N_goal.row,N_goal.col)=0;
        v(barrier.row,barrier.col)=0;
    end
    disp(v)
end
% disp(iterate)
disp(v)
end

function c= make_action(row,col,action_size,barrier,goal,N_goal)
    v=zeros(row,col,action_size);
    action=[1,1,1,1];
    for r=1:row
        for c=1:col 
            if ~((barrier.row==r&& barrier.col==c)||(goal.row==r&&...
                                goal.col==c)||(N_goal.row==r&& N_goal.col==c))
                A=1;
                recent(1)=0;
                recent(2)=0;
                while (A<=action_size)
                    recent(1)=r;
                    recent(2)=c;
                    if A==1
                        recent(1)=recent(1)+1;
                        recent(2)=recent(2);
                    elseif A==2
                        recent(1)=recent(1)-1;
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
                            else
                                 v(r,c,A)=0;
                            end
                        end
                    end
                    A=A+1;
                end
            end
        end
        v(3,1,1)=1;
        v(1,2,3)=1;
        v(2,2,3)=1;
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
    disp([pos.row,pos.col])
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


