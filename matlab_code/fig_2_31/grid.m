classdef grid <handle
    %actions=["up","down","right","left"]
    % grid is initialize with grid initialization
    properties
        x=0;
        y=0;
        goal=[];
        start=[];
        barrier=[];
        current=[]
        N_goal=[];
        rewards=[];
    end
    methods
        function self=initialization(self,width,height,start,goal,barrier,N_goal)
            self.x=width; %g.initialization(4,3,[3,4],[1,1],[2,3],[2,1])
            self.y=height;
            self.start=start;
            self.current=start;
            self.goal=goal;
            self.barrier=barrier;
            self.N_goal=N_goal;
            self.rewards=zeros(height,width);
            self.rewards(self.goal(1),self.goal(2))=100;
            self.rewards(self.N_goal(1),self.N_goal(2))=-100;
            
        end
        function current= reset(self)
             current=[0,0];
             current=self.start;
             self.current=current;
        end
        function pos=next_state(self,action)
            last=[];
            last(1)=self.current(1);
            last(2)=self.current(2);
            pos.x=self.current(1);
            pos.y=self.current(2);
            switch action
                case 1 %up
                    pos.x= self.current(1) - 1;
                case 2 %down
                    pos.x= self.current(1) + 1;
                case 3 %left
                    pos.y= self.current(2) - 1;
                case 4 %right
                    pos.y=self.current(2) + 1;
            end
           if (pos.x==self.barrier(1) && pos.y==self.barrier(2))
                pos.x=last(1);
                pos.y=last(2);
            end
            if(pos.y <= 0)
                pos.y = 1; end
            if(pos.y > self.x)
                pos.y = self.x; end
            if(pos.x <= 0)
                pos.x = 1; end
            if(pos.x > self.y)
                pos.x = self.y; end
           
            self.current=[pos.x,pos.y];
            pos=[pos.x,pos.y];
        end
        function go=terminate(self,next)
            if ((self.barrier(1)==next(1)&& self.barrier(2)==next(2))||(self.goal(1)==next(1)...
                    && self.goal(2)==next(2))||(self.N_goal(1)==next(1)&& self.N_goal(2)==next(2)))
                go=1;
            else
                go=0;
            end           
        end
        function future_s=step(self,action)
            next=self.next_state(action);
            reward=self.rewards(next(1),next(2));
            done=self.terminate(next);
            future_s={[next(1),next(2)],reward,done};
        end
    end
    
end

