classdef pendulum <handle
    properties(SetAccess=private)
        max_speed=8;
        max_torque=2;
        t=0.05;
        state=[pi,0];
        maxite=1000;
        count=0; 
    end 
   
    methods
                
        function [next,done,reward]= forward(self,st,action)
            self.pendulumPlot();
            self.count=self.count+1;
            if self.count==self.maxite
                done=1;
                self.count=1;
            else
                done=0;
            end
            self.state=st;         
            th = self.state(1); 
            thdot = self.state(2); 
            g = 10;
            m = 1;
            l = 1;
            dt = self.t;
            if action<-self.max_torque
                action=-2;
            elseif action>self.max_torque
                action=2;
            else
                action=action;
            end 
            ax=angle_normalize(th);
            reward= ax^2 + 0.1*thdot^2 +0.001*(action^2);
            
            newthdot = thdot + (-3*g/(2*l) * sin(th + pi) + 3./(m*l^2)*action) * dt;
            newth = th + newthdot*dt;
            if newthdot<-self.max_speed
                newthdot=-8;
            elseif newthdot>self.max_speed
                newthdot=8;
            else
               newthdot=newthdot;
            end  
            next=[newth, newthdot];
            self.state=next;                            
        end
       
    end
end
panel = figure;
panel.Position = [300 200 300 200];
panel.Color = [1 1 1];
subplot(1,4,1)
hold on
f = plot(0,0,'b','LineWidth',10);
axPend = f.Parent;
axPend.Visible='off';
axPend.Position = [0.2 0.2 0.3 0.3]; 

axPend.Clipping = 'off';
axis equal
axis([-1.2679 1.2679 -1 1]);
plot(0.001,0,'.k','MarkerSize',50); 
hold off;

function X= angle_normalize(x)
         X=(((x+pi) / (2*pi)) - pi);      
end
function pendulumPlot(x,)
            
end
        
        
