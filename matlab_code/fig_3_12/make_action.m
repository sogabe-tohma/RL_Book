
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
 