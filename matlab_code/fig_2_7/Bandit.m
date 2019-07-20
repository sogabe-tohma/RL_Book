classdef Bandit <handle
    %   Detailed explanation goes here
    properties
        N_arm;
        arm_values;
        k,est_values;
       
    end    
    methods
        function self=init(self,N_arm)
            self.N_arm=N_arm;
            self.arm_values=normrnd(0,1,N_arm);
            self.k=zeros(1,N_arm);
            self.est_values=zeros(1,N_arm);           
        end
       
        function reward = get_reward(self,action)
            noise= normrnd(0,1);
            reward=self.arm_values(action)+noise;
         
        end
        function idx=choose_eps_greedy(self,epsilon)
            rand_num=rand(1);
            if epsilon>rand_num
                idx=randi([1,self.N_arm],1,1);
            else
                [~,idx]=max(self.est_values);
            end
           
        end    
        function val=update_est(self,action,reward)
            self.k(action) = self.k(action)+1;
            alpha=1./self.k(action);
            self.est_values(action)=self.est_values(action)+alpha*(reward -self.est_values(action));
            val=self.est_values(action);          
        end
    end
end

