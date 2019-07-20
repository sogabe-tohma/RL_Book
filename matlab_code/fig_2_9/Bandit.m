classdef Bandit <handle
    %   Detailed explanation goes here
    properties
        N_arm;
        mean_rate;
        arm_values;
        k,est_values;
        
    end    
    methods
        function self=init(self,N_arm)
            self.N_arm=N_arm;
            self.mean_rate=0.0;
            self.arm_values=normrnd(0,1,N_arm);
            self.k=zeros(1,N_arm);
            self.est_values=ones(1,N_arm)*self.mean_rate;           
        end
       
        function reward = get_reward(self,action)
            noise= normrnd(0,1);
            reward=self.arm_values(action)+noise;
         
        end
        function u_val=ucb(self,n)
            u_val=self.est_values+sqrt(2*log(n)./(self.k+1e-2));
            
        end
        function idx=choose_eps_greedy(self,n)
%             fprintf('test print 1 is :[ %3f %3f %3f %3f %3f %3f %3f %3f %3f %3f ]\n',self.ucb(n));
%             fprintf(' n :%d \n',n);
            [~,idx]=max(self.ucb(n));
           
        end    
        function val=update_est(self,action,reward)
            self.k(action) = self.k(action)+1;
            alpha=1./self.k(action);
            self.est_values(action)=self.est_values(action)+alpha*(reward -self.est_values(action));
            val=self.est_values(action);          
        end
    end
end

