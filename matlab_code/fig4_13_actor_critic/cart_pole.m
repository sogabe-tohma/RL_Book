
classdef cart_pole < handle 
    %{
Parameters for simulation
g=9.8;              %Gravity
Mass_Cart=1.0;      %Mass of the cart is assumed to be 1Kg
Mass_Pole=0.1;      %Mass of the pole is assumed to be 0.1Kg
Total_Mass=Mass_Cart+Mass_Pole;
Length=0.5;         %Half of the length of the pole 
PoleMass_Length=Mass_Pole*Length;
Force_Mag=10.0;
Tau=0.02;           %Time interval for updating the values;
    
   %}
    properties(SetAccess=private)  
        gravity = 9.8
        masscart = 1.0
        masspole = 0.1
        total_mass = 0.1+1;
        length = 0.5 
        polemass_length = 0.1*0.5;
        force_mag = 9.8
        tau = 0.02 
        theta_threshold_radians = 12 * 2 * pi / 360
        x_threshold = 2.4
        steps_beyond_done=0;
        state=random('Uniform',-0.05, 0.05, [1,4])
        defult=[ 0.5 0 0 0];
        max_steps=500;
        count=0;
      
    end   
    methods        
        function [next,reward,done]= forward(self,action,epi)
%            rng(242)
            if self.count==self.max_steps
                self.count=0; % if max_step count reset
            end
            self.count=self.count+1;
            x= self.state(1);
            x_dot=self.state(2);
            theta=self.state(3);
            theta_dot = self.state(4);
            %action apply as force            
            if action==1
                force = self.force_mag ;
            else
                force =-self.force_mag;
            end
            %next state calculation
            costheta = cos(theta);
            sintheta = sin(theta);
            temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass;
            thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass));
            xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass;
            x  = x + self.tau * x_dot;
            x_dot = x_dot + self.tau * xacc;
            theta = theta + self.tau * theta_dot;
            theta_dot = theta_dot + self.tau * thetaacc;
            next= [x,x_dot,theta,theta_dot];                   
            if x < -self.x_threshold || x > self.x_threshold ||theta < -self.theta_threshold_radians || theta > self.theta_threshold_radians
                done = 1;
            else
                done=0;
            end
            if done==0
               reward=-10;
            else
                reward=0.1;
            end
            if done==1
                next=self.defult;
                self.count=0;
            end   
            self.state =next;    
            self.CartPlot(action,epi)
        end
        function init_state=re_set(self)
            init_state=random('Uniform',-0.05, 0.05, [1,4]);
            self.state=init_state;
        end
        function CartPlot(self,acton,ep)
            if ep>100
                x     = self.state(1);
                theta = self.state(3);
                l= 2 ;     %pole's Length for ploting it can be different from the actual length
                pxg = [x+1 x-1 x-1 x+1 x+1];
                pyg = [0.25 0.25 1.25 1.25 0.25];
                pxp=[x x+l*sin(theta)];
                pyp=[1.25 1.25+l*cos(theta)];
                subplot(1,1,1);
                %Car 
                fill(pxg,pyg,[.6 .6 .5],'LineWidth',2);  %car
                hold on
                title(['Steps: ',int2str(self.count)])
                %Car Wheels
                plot(x-0.5,0.25,'rO','LineWidth',2,'Markersize',20,'MarkerEdgeColor','k','MarkerFaceColor',[0.5 0.5 0.5]);
                plot(x+0.5,0.25,'rO','LineWidth',2,'Markersize',20,'MarkerEdgeColor','k','MarkerFaceColor',[0.5 0.5 0.5]);
                %Pendulum
                plot(pxp,pyp,'-r','LineWidth',3);
                plot(pxp(1),pyp(1),'.r','LineWidth',2,'Markersize',10,'MarkerEdgeColor','k','MarkerFaceColor','r');
                %text(x + arrowfactor_x - 0.5 ,0.8,text_arrow);
                axis([-6 6 0 6])
                %grid
                box off
                drawnow;
                hold off
            end
        end
    end
        
end

        
